"""Actual TabbyAPI server with external observation/admin diagnostics only.

Run with TabbyAPI's Python -I, cwd D:/tabbyAPI. Does not import this checkout,
alter scheduling, replace routing, or bypass native eligibility guards.
"""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import threading
import time
from collections import Counter

TABBY=Path('D:/tabbyAPI');ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'native-results/nemotron-wheel-test'
sys.path.insert(0,str(TABBY))
import torch
import exllamav3
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import native_moe
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.generator.generator import Generator
from exllamav3.model.model import Model
assert Path(exllamav3.__file__).is_relative_to(TABBY/'venv')
assert Path(ext.__file__).is_relative_to(TABBY/'venv')
assert ext.native_moe_assignment_abi==1
assert os.environ['CUDA_VISIBLE_DEVICES']=='2' and torch.cuda.device_count()==1
assert 'RTX PRO 6000' in torch.cuda.get_device_name(0)

from common import model as serving
from common.auth import check_admin_key
from common.tabby_config import config
from fastapi import Depends,HTTPException
from endpoints import server
import uvicorn
from loguru import logger
# Tabby's normal startup prints credentials; keep validation logs shareable.
def redact_startup(record):
    if 'Your API key is:' in record['message']:
        record['message']='Existing authentication credentials loaded (redacted in validation logs).'
logger.configure(patcher=redact_startup)

state=dict(mode='off',name='startup',frames=[],forwards=[],jobs={},moe=Counter(),dispatch=Counter(),
           shadow=[],seen=set(),refreshes=[],threads=Counter(),current=None)


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def stock_routed(m,y,selected,weights):
    """Diagnostic oracle: stock all-fused fallback, same inputs/weights/tile."""
    flat=selected.flatten();order=flat.argsort();e=m.num_experts
    counts=torch.bincount(flat,minlength=e+1);host=counts.tolist()
    starts=counts.cumsum(0)-counts
    inverse=torch.empty_like(order).scatter_(0,order,torch.arange(order.numel(),device=y.device))
    tokens=torch.arange(y.shape[0],device=y.device).repeat_interleave(selected.shape[1])[order]
    packed=weights.flatten()[order]
    output=torch.zeros_like(y,dtype=torch.float);scratch=torch.empty((order.numel(),y.shape[1]),device=y.device)
    up,down=m.multi_up,m.multi_down;b=m.fused_mode_buffers
    ext.exl3_moe(y,output,counts,tokens,packed,b.temp_state_g,b.temp_state_u,b.temp_intermediate_g,b.temp_intermediate_u,
        m.activation_fn_idx,up.K,up.K,down.K,up.ptrs_trellis,up.ptrs_suh,up.ptrs_svh,
        up.ptrs_trellis,up.ptrs_suh,up.ptrs_svh,down.ptrs_trellis,down.ptrs_suh,down.ptrs_svh,
        up.mcg,up.mul1,up.mcg,up.mul1,down.mcg,down.mul1,m.act_limit,
        sum(x>0 for x in host[:e]),scratch,starts,1,m.fused_rows,16)
    kind=(counts>0).long();kind[-1]=0
    ext.exl3_moe_gather(output,scratch,flat,inverse,starts[:e],starts[:e],kind[:e],packed)
    return output


original_run=native_moe.run
def observed_run(m,y,selected,weights,params):
    out=original_run(m,y,selected,weights,params)
    if state['mode']=='audit' and out is not None:
        rows=y.shape[0];state['dispatch'][str(rows)]+=1
        state['threads'][threading.current_thread().name]+=1
        key=(m.key,rows)
        if key not in state['seen']:
            reference=stock_routed(m,y,selected,weights)
            exact=torch.equal(reference,out)
            state['shadow'].append(dict(layer=m.key,rows=rows,exact=exact,
                default_stream=torch.cuda.current_stream()==torch.cuda.default_stream()))
            state['seen'].add(key)
            assert exact,(m.key,rows,'routed result mismatch')
    return out

original_moe=BlockSparseMLP.forward
def observed_moe(self,x,params,*args,**kwargs):
    if state['mode']=='audit':state['moe'][str(x.numel()//x.shape[-1])]+=1
    return original_moe(self,x,params,*args,**kwargs)

original_forward=Model.forward
def observed_forward(self,input_ids,params=None):
    c=serving.container
    target=c is not None and self is c.model
    if state['mode']=='off' or not target:return original_forward(self,input_ids,params)
    t=time.perf_counter();out=original_forward(self,input_ids,params)
    r=dict(shape=list(input_ids.shape),seconds=time.perf_counter()-t,
           thread=threading.current_thread().name,default_stream=torch.cuda.current_stream()==torch.cuda.default_stream())
    if state['mode']=='audit' and input_ids.numel()<=16:
        r.update(ids=input_ids.tolist(),logits_sha256=digest(out),positions=params.get('cache_seqlens').tolist())
    state['forwards'].append(r)
    if state['current'] is not None:state['current']['shapes'].append(r['shape'])
    return out
Model.forward=observed_forward

original_iterate=Generator.iterate
def observed_iterate(self,*args,**kwargs):
    if state['mode']=='off':return original_iterate(self,*args,**kwargs)
    t=time.perf_counter();before={str(j.serial_number):j.new_tokens for j in self.active_jobs}
    frame=dict(start=t,shapes=[],before=before)
    state['current']=frame
    try:results=original_iterate(self,*args,**kwargs)
    finally:state['current']=None
    frame.update(end=time.perf_counter(),after={str(j.serial_number):j.new_tokens for j in self.active_jobs},states=[])
    for j in self.active_jobs:
        if not j.is_prefill_done():continue
        s,seq=j.recurrent_state.target,j.sequences[0]
        row=dict(serial=j.serial_number,position=s.position,kv=seq.kv_position,
                 length=len(seq.sequence_ids),history=s.last_history,generated=j.new_tokens)
        assert s.last_history==0 and s.position==seq.kv_position==len(seq.sequence_ids)-1,row
        frame['states'].append(row)
    for result in results:
        j=result.get('job')
        if j is None:continue
        key=str(j.serial_number)
        if key not in state['jobs']:
            ids=j.sequences[0].input_ids.torch()
            state['jobs'][key]=dict(prompt_length=ids.shape[-1],prompt_sha256=digest(ids))
        if result.get('eos'):
            record=state['jobs'][key]
            record.update({k:result.get(k) for k in ('prompt_tokens','cached_tokens','new_tokens','time_enqueued',
                'time_prefill','time_generate','accepted_draft_tokens','rejected_draft_tokens','eos_reason')})
            tokens=j.sequences[0].sequence_ids.torch()[0,record['prompt_length']:].tolist()
            record.update(tokens=tokens,token_sha256=hashlib.sha256(str(tokens).encode()).hexdigest())
    state['frames'].append(frame)
    return results
Generator.iterate=observed_iterate


def install_refresh(c):
    if c.draft_model is None or hasattr(c.draft_model,'_wheel_test_refresh'):return
    original=c.draft_model.prefill
    def refresh(input_ids,params=None):
        ids=input_ids
        sample=(state['mode']=='audit' and ids.shape[1]<=3 and
                not any(r.get('repeat_exact') and r['shape']==list(ids.shape) for r in state['refreshes']))
        tensors=[t for l in c.draft_cache.layers.values() for t in l.get_tensors()] if sample else []
        before=[t.clone() for t in tensors]
        out=original(ids,params)
        if state['mode']=='audit':
            r=dict(shape=list(ids.shape),positions=params['cache_seqlens'].tolist())
            if sample:
                first=[t.clone() for t in tensors]
                mask=torch.zeros(tensors[0].shape[:2],dtype=torch.bool,device=tensors[0].device)
                for row,pos in enumerate(r['positions']):
                    for q in range(ids.shape[1]):
                        at=pos+q;mask[int(params['block_table'][row,at//256]),at%256]=True
                r['outside_exact']=all(torch.equal(x[~mask].view(torch.uint8),y[~mask].view(torch.uint8)) for x,y in zip(before,first))
                original(ids,params)
                r['repeat_exact']=all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(first,tensors))
                assert r['outside_exact'] and r['repeat_exact']
            state['refreshes'].append(r)
        return out
    c.draft_model.prefill=refresh;c.draft_model._wheel_test_refresh=True


def status():
    c=serving.container
    g=c.generator.generator if c and c.generator else None
    def cache_bytes(cache):
        return sum(t.numel()*t.element_size() for l in cache.layers.values() for t in l.get_tensors()) if cache else 0
    return dict(version=md.version('exllamav3'),package=exllamav3.__file__,extension=ext.__file__,abi=ext.native_moe_assignment_abi,
        ir=native_moe.enabled,mode=state['mode'],thread=threading.current_thread().name,gpu=torch.cuda.get_device_name(0),
        visible=os.environ.get('CUDA_VISIBLE_DEVICES'),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),free=torch.cuda.mem_get_info()[0],
        model_loaded=bool(c and c.loaded),active=len(g.active_jobs) if g else 0,pending=len(g.pending_jobs) if g else 0,
        generator=dict(batch=g.max_batch_size,drafts=g.num_draft_tokens,dynamic=g.dynamic_draft,cap=g.draft_max_rows,
                       cache_tokens=g.cache.max_num_tokens) if g else None,
        kv_bytes=dict(target=cache_bytes(c.cache),draft=cache_bytes(c.draft_cache)) if c and c.loaded else None,
        max_seq_len=c.max_seq_len if c else None,arenas=[dict(key=str(k),bytes=sum(t.numel()*t.element_size() for t in
            [w.counts,w.tables,w.tokens,w.weights,w.inverse,w.output,w.scratch,*w.buffers])) for k,w in native_moe._arenas.items()])

original_setup=server.setup_app
def setup_app(*args,**kwargs):
    app=original_setup(*args,**kwargs)
    @app.get('/__validation/status',dependencies=[Depends(check_admin_key)])
    async def get_status():return status()
    @app.post('/__validation/control',dependencies=[Depends(check_admin_key)])
    async def control(body:dict):
        c=serving.container
        if not c or not c.loaded:raise HTTPException(409,'no model')
        if c.active_job_ids or c.generator.jobs:raise HTTPException(409,'requests still active')
        if body.get('reset_cache'):await c.create_generator()
        native_moe.enabled=bool(body.get('ir',True))
        c.generator.generator.draft_max_rows=int(body.get('draft_cap',0))
        mode=body.get('mode','measure');assert mode in ('audit','measure','off')
        state.update(mode=mode,name=body.get('name','run'),frames=[],forwards=[],jobs={},moe=Counter(),dispatch=Counter(),
                     shadow=[],seen=set(),refreshes=[],threads=Counter(),current=None)
        native_moe.run=observed_run if mode=='audit' else original_run
        BlockSparseMLP.forward=observed_moe if mode=='audit' else original_moe
        install_refresh(c)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        return status()
    @app.get('/__validation/results',dependencies=[Depends(check_admin_key)])
    async def results():
        result={k:v for k,v in state.items() if k not in ('seen','current')}
        result['status']=status()
        return result
    @app.post('/__validation/shutdown',dependencies=[Depends(check_admin_key)])
    async def shutdown():
        server_instance.should_exit=True
        return {'shutdown':True}
    return app
server.setup_app=setup_app
original_server_init=uvicorn.Server.__init__
def server_init(self,*args,**kwargs):
    global server_instance
    server_instance=self;original_server_init(self,*args,**kwargs)
uvicorn.Server.__init__=server_init

if __name__=='__main__':
    import main
    main.entrypoint()
