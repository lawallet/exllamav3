"""Isolated TabbyAPI diagnostics; run with Tabby's Python -I, from D:/tabbyAPI.

The installed backend executes normal HTTP requests. Controls require the existing
admin key and reject changes while requests are active. No persistent config edits.
"""
import cProfile
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['DS41_RESULTS'])
OUT.mkdir(parents=True, exist_ok=True)
if os.environ.get('DS41_EXTENSION_DIR'): sys.path.insert(0,os.environ['DS41_EXTENSION_DIR'])
if os.environ.get('DS41_PACKAGE_ROOT'): sys.path.insert(0,os.environ['DS41_PACKAGE_ROOT'])
sys.path.insert(0, 'D:/tabbyAPI')
import torch
import exllamav3
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import native_plan
from exllamav3.generator.generator import Generator
from common import model as serving
from common.auth import check_admin_key
from endpoints import server
from fastapi import Depends, HTTPException
import uvicorn
from loguru import logger

assert Path(exllamav3.__file__).is_relative_to(Path(os.environ.get('DS41_PACKAGE_ROOT','D:/tabbyAPI/venv')))
if os.environ.get('DS41_EXTENSION_DIR'):
    assert Path(ext.__file__).parent == Path(os.environ['DS41_EXTENSION_DIR'])
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,0,1'
assert 'RTX PRO 6000' in torch.cuda.get_device_name(0)
native_plan.enabled = False
state = dict(jobs=[], name='startup', profile=False, audit=False)
audit_saved=[]

def install_audit(c,enabled):
    for obj,name,original in reversed(audit_saved): setattr(obj,name,original)
    audit_saved.clear()
    state['oracles']=[];state['logits']=[];state['executions']={}
    if not enabled: return
    from exllamav3.modules import native_deepseek_v41
    from exllamav3.modules.hyperconnections import hc_flush
    def check(label,a,b):
        exact=torch.equal(a,b)
        state['oracles'].append(dict(label=label,exact=exact))
        assert exact,label
    class AuditPlan:
        def __init__(self,plan,block):self.plan,self.block=plan,block
        def __getattr__(self,name):return getattr(self.plan,name)
        def begin_carry(self,x,pre):
            self.reference=x.clone();self.params={'hc_pre':pre.view(*x.shape[:3]) if pre is not None else None}
            h,n=self.block.attn_hc,self.block.attn_norm
            self.post,self.comb,y,_=h.mix_norm(self.reference,self.params,n)
            expected=n.forward(y,self.params,out_dtype=torch.half)
            out=self.plan.begin_carry(x,pre)
            check(self.block.key+':begin',out,expected)
            check(self.block.key+':pre0',self.plan.carry_output,self.params['hc_pre'].view(-1,4))
            state['executions'][self.block.key]=state['executions'].get(self.block.key,0)+1
            return out
        def middle(self,x,y):
            self.block.attn_hc.apply_(self.reference,y,self.post,self.comb,self.params)
            hc_flush(self.params)
            h,n=self.block.mlp_hc,self.block.mlp_norm
            self.post,self.comb,z,_=h.mix_norm(self.reference,self.params,n)
            expected=n.forward(z,self.params,out_dtype=torch.half)
            out=self.plan.middle(x,y)
            check(self.block.key+':middle',out,expected)
            check(self.block.key+':residual0',x,self.reference)
            check(self.block.key+':pre1',self.plan.carry_output,self.params['hc_pre'].view(-1,4))
            return out
        def finish(self,x,y):
            self.block.mlp_hc.apply_(self.reference,y,self.post,self.comb,self.params)
            hc_flush(self.params)
            self.plan.finish(x,y)
            check(self.block.key+':finish',x,self.reference)
    for role,model in (('target',c.model),('draft',c.draft_model)):
        if model is None:continue
        original=model.forward;audit_saved.append((model,'forward',original))
        def forward(input_ids,params=None,*args,_fn=original,_role=role,**kwargs):
            result=_fn(input_ids,params,*args,**kwargs)
            if input_ids.shape[1]<=5:
                state['logits'].append(dict(role=_role,shape=list(input_ids.shape),ids=input_ids.tolist(),
                    sha256=hashlib.sha256(result.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()))
            return result
        model.forward=forward
        for block in model.modules:
            if not getattr(block,'_native_carry_pre',False):continue
            original=block._native_plan_get;audit_saved.append((block,'_native_plan_get',original))
            def select(b,x,params,_get=original):
                plan=_get(b,x,params)
                return AuditPlan(plan,b) if plan is not None else None
            block._native_plan_get=select
profiler = None
trace = None
trace_saved = []

def start_trace():
    global trace
    from exllamav3.modules.hyperconnections import HyperConnection
    from exllamav3.modules.dsv4 import DSV4Attention
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.row_table import RowTable
    from exllamav3.modules.ple import PLELayer
    from exllamav3.model.model import Model
    from exllamav3.modules.arch_specific.dspark import DSparkAttention
    trace=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
    trace.__enter__()
    for cls, name in ((HyperConnection,'mix_norm'),(HyperConnection,'apply_'),
                      (DSV4Attention,'forward'),(DSparkAttention,'forward'),
                      (BlockSparseMLP,'forward'),(RowTable,'lookup'),
                      (RowTable,'_gather_rows'),(PLELayer,'forward'),(Model,'forward')):
        original=getattr(cls,name);trace_saved.append((cls,name,original))
        def call(self,*args,_original=original,_label=cls.__name__+'.'+name,**kwargs):
            with torch.profiler.record_function('ds41.'+_label+':'+str(getattr(self,'key',getattr(self,'component','')))):
                return _original(self,*args,**kwargs)
        setattr(cls,name,call)

def stop_trace():
    global trace
    for cls,name,original in reversed(trace_saved): setattr(cls,name,original)
    trace_saved.clear()
    trace.__exit__(None,None,None)
    trace.export_chrome_trace(str(OUT/(state['name']+'-trace.json')))
    rows=[dict(key=e.key,count=e.count,cpu_total_us=e.cpu_time_total,cpu_self_us=e.self_cpu_time_total,
               device_total_us=e.device_time_total,device_self_us=e.self_device_time_total)
          for e in trace.key_averages()]
    (OUT/(state['name']+'-trace-summary.json')).write_text(json.dumps(rows,indent=2))
    trace=None;state['trace']=False

def redact(record):
    if 'Your API key is:' in record['message']:
        record['message'] = 'Existing authentication credentials loaded (redacted).'
logger.configure(patcher=redact)

def memory():
    return [dict(device=i, name=torch.cuda.get_device_name(i),
        allocated=torch.cuda.memory_allocated(i), reserved=torch.cuda.memory_reserved(i),
        peak_allocated=torch.cuda.max_memory_allocated(i),
        peak_reserved=torch.cuda.max_memory_reserved(i), free=torch.cuda.mem_get_info(i)[0])
        for i in range(torch.cuda.device_count())]

def status():
    c = serving.container
    g = c.generator.generator if c and c.generator else None
    mods = list(c.model.modules) if c and c.loaded else []
    ngrams = []
    def visit(m):
        if type(m).__name__ == 'NGramEmbedding':
            ngrams.append(dict(key=m.key, mode=m.mode, prefetch=m.prefetch_stats))
        for child in getattr(m, 'modules', []): visit(child)
    for m in mods: visit(m)
    plans=[plan for model in ((c.model,c.draft_model) if c and c.loaded else ()) if model
           for m in model.modules for _,plan in getattr(m,'_native_deepseek',{}).values()]
    return dict(version=md.version('exllamav3'), package=exllamav3.__file__, extension=ext.__file__,
        model_loaded=bool(c and c.loaded), ir=native_plan.enabled, memory=memory(), ngrams=ngrams,
        generator=dict(batch=g.max_batch_size, drafts=g.num_draft_tokens, dynamic=g.dynamic_draft,
                       cache_tokens=g.cache.max_num_tokens) if g else None,
        placement=[dict(key=m.key,device=str(m.device),attention=str(m.attn.device) if getattr(m,'attn',None) else None)
                   for m in mods], vision=bool(c and getattr(c,'vision_model',None)),
        active_tokens=[j.new_tokens for j in g.active_jobs] if g else [],
        native_plans=len(plans),graph_captures=sum(p.graph_captures for p in plans),
        graph_replays=sum(p.graph_replays for p in plans))

original_iterate = Generator.iterate
def iterate(self, *args, **kwargs):
    global profiler
    generated = max((j.new_tokens for j in self.active_jobs), default=0)
    if state.get('trace') and trace is None and generated >= 64: start_trace()
    if trace is not None and generated >= 96: stop_trace()
    if state['profile'] and profiler is None and generated >= 64:
        profiler = cProfile.Profile(); profiler.enable()
    results = original_iterate(self, *args, **kwargs)
    for r in results:
        if r.get('eos'):
            j = r['job']
            record = {k:r.get(k) for k in ('prompt_tokens','cached_tokens','new_tokens',
                'time_prefill','time_generate','accepted_draft_tokens','rejected_draft_tokens','eos_reason')}
            record['ids'] = j.sequences[0].sequence_ids.torch().tolist()
            record['memory'] = memory()
            state['jobs'].append(record)
            if profiler is not None:
                profiler.disable(); profiler.dump_stats(str(OUT / (state['name']+'.prof')))
                profiler = None; state['profile'] = False
            if trace is not None: stop_trace()
    return results
Generator.iterate = iterate

original_setup = server.setup_app
def setup(*args, **kwargs):
    app = original_setup(*args, **kwargs)
    @app.get('/__validation/status', dependencies=[Depends(check_admin_key)])
    async def get_status(): return status()
    @app.post('/__validation/control', dependencies=[Depends(check_admin_key)])
    async def control(body:dict):
        c = serving.container
        if not c or not c.loaded: raise HTTPException(409,'no model')
        if c.active_job_ids or c.generator.jobs: raise HTTPException(409,'requests active')
        # Recreate the generator to discard prefix-cache reuse between samples.
        saved_draft=(c.draft_model,c.draft_cache,c.draft_num_tokens)
        if not body.get('mtp'):
            c.draft_model=c.draft_cache=None
            c.draft_num_tokens=0
        try:
            await c.create_generator()
        finally:
            c.draft_model,c.draft_cache,c.draft_num_tokens=saved_draft
        g = c.generator.generator
        assert (g.draft_model is not None)==bool(body.get('mtp'))
        g.num_draft_tokens = 3 if body.get('mtp') else 0
        g.record_draft_stats = True
        native_plan.enabled = bool(body.get('ir',False))
        if getattr(ext,'native_block_plan_carry_abi',None)==1:
            from exllamav3.modules import native_deepseek_v41 as binding
            binding.graphs_enabled=bool(body.get('graphs',False))
            if binding.graphs_enabled:
                assert binding.graphs_available,'carry graphs unavailable'
            for model in (c.model,c.draft_model):
                if model:
                    for block in model.modules:
                        for _,plan in getattr(block,'_native_deepseek',{}).values():
                            plan.set_graphs(binding.graphs_enabled)
        install_audit(c,body.get('audit',False))
        state.update(jobs=[],name=body.get('name','run'),profile=body.get('profile',False),trace=body.get('trace',False))
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i); torch.cuda.reset_peak_memory_stats(i)
        return status()
    @app.get('/__validation/results', dependencies=[Depends(check_admin_key)])
    async def results(): return dict(state=state,status=status())
    @app.post('/__validation/shutdown', dependencies=[Depends(check_admin_key)])
    async def shutdown():
        instance.should_exit = True
        return {'shutdown':True}
    return app
server.setup_app = setup
original_init = uvicorn.Server.__init__
def init(self,*args,**kwargs):
    global instance
    instance=self; original_init(self,*args,**kwargs)
uvicorn.Server.__init__=init

if __name__ == '__main__':
    import main
    main.entrypoint()
