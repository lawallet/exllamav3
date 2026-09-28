"""Stock-only progressive CSA/indexer instrumentation; no production overrides."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BOOT=argparse.ArgumentParser(add_help=False)
BOOT.add_argument('--extension-dir',type=Path,default=ROOT/'native-results/block-ir/extension')
sys.path.insert(1,str(BOOT.parse_known_args()[0].extension_dir.resolve()))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules import dsv4, native_plan
from exllamav3.ext import exllamav3_ext as ext
from util.validate_deepseek_plan import direct, digest


def meta(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),
                ptr=t.data_ptr(),storage=t.untyped_storage().data_ptr(),offset=t.storage_offset(),
                bytes=t.untyped_storage().nbytes(),stream=torch.cuda.current_stream(t.device).cuda_stream if t.is_cuda else None)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__,parents=[BOOT])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--layer',type=int,default=36)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,obj): (args.output/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    native_plan.enabled=False
    cfg=Config.from_directory(r'D:\tabbyAPI\models\turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw')
    model=Model.from_config(cfg)
    cache=Cache(model,max_num_tokens=8192,max_batch_size=4,max_history=5)
    tokenizer=Tokenizer.from_config(cfg)
    model.load(use_per_device=[85,40],max_chunk_size=256,max_batch_size=4)
    block=next(b for b in model.modules if getattr(b,'layer_idx',None)==args.layer)
    attn=block.attn
    layer=cache.get_all_recurrent_layers()[(args.layer,0)]
    save('environment.json',dict(torch=torch.__version__,extension=ext.__file__,
        environment={k:v for k,v in os.environ.items() if k.startswith(('EXL3_','CUDA_'))},
        placement=[dict(key=m.key,device=str(m.device)) for m in model.modules]))
    lins=[attn.q_a,attn.wkv,attn.compressor.wkv,attn.compressor.wgate,attn.indexer.wkv,attn.indexer.wgate]
    save('weights.json',[dict(key=l.key,quant=l.quant_type,output=l.out_features,
        K=getattr(l.inner,'K',None),mcg=getattr(l.inner,'mcg',None),mul1=getattr(l.inner,'mul1',None),
        tensors={k:dict(**meta(t),sha256=digest(t.reshape(-1))) for k,t in vars(l.inner).items() if torch.is_tensor(t)}) for l in lins])
    events=[]
    samples={}
    active=False
    phase=''
    def record(name,t,**kw):
        sample=t.detach().cpu().clone()
        key=f'{phase}:{name}:{len(events)}'
        samples[key]=sample
        events.append(dict(name=name,phase=phase,meta=meta(t),sha256=digest(sample),
                           sample=sample.flatten()[:8].tolist(),**kw))
    original=attn.forward
    def attention(x,params,*a,**kw):
        nonlocal active
        states=params.get('recurrent_states',[])
        active=bool(states and states[0].position==300 and x.shape[1]==1)
        if active:
            s=states[0]
            record('attention-input',x,position=s.position,slot=s.slot,ring_index=s.position%layer.buf_rows,
                   current_device=torch.cuda.current_device())
            record('indexer-row-before',layer.idx_buf_kv[s.slot,s.position%layer.buf_rows])
        y=original(x,params,*a,**kw)
        if active:
            record('attention-output',y)
            record('indexer-row-after-attention',layer.idx_buf_kv[s.slot,s.position%layer.buf_rows])
        active=False
        return y
    mgemm=ext.exl3_mgemm
    def project(*a,**kw):
        fan=attn.x_fan
        relevant=active and fan is not None and a[1].data_ptr()==fan['trellis'].data_ptr()
        if relevant:
            record('projection-input',a[0])
            outs,ptrs,had=attn._fan_scratch[('x',1)]
            record('projection-pointer-table',ptrs)
            record('projection-widths',fan['n'])
            events.append(dict(phase=phase,name='projection-metadata',outputs=[meta(t) for t in outs],scratch=meta(had)))
        result=mgemm(*a,**kw)
        if relevant:
            events.append(dict(phase=phase,name='projection-dispatch',shape_index=result))
            for i,t in enumerate(outs): record(f'projection-output-{i}',t)
        return result
    compress=ext.dsv4_compress
    def write(*a,**kw):
        relevant=active and a[2].untyped_storage().data_ptr()==layer.idx_buf_kv.untyped_storage().data_ptr()
        if relevant:
            record('write-source-kv',a[0]); record('write-source-gate',a[1])
            record('write-destination-before',a[2][300%layer.buf_rows])
            record('write-position-bias',a[5])
            record('write-norm-weight',a[6])
            events.append(dict(phase=phase,name='write-metadata',position=a[11],rate=a[13],
                               kv=meta(a[2]),gate=meta(a[3]),overlap=meta(a[4])))
        result=compress(*a,**kw)
        if relevant:
            record('write-destination-after',a[2][300%layer.buf_rows])
            record('write-overlap-after',a[4])
        return result
    with ExitStack() as scope:
        scope.enter_context(patch.object(attn,'forward',attention))
        scope.enter_context(patch.object(ext,'exl3_mgemm',project))
        scope.enter_context(patch.object(ext,'dsv4_compress',write))
        for graphed in (True,False):
            dsv4.bc_dsa_enable=graphed
            for repeat in range(args.repeats):
                phase=f'{"graph" if graphed else "eager"}-{repeat}'
                result=direct(model,cache,tokenizer,1,1,2,300,False)
                save(f'{phase}.json',result)
                save('events.json',events)
                torch.save(samples,args.output/'samples.pt')
                print(f'Captured {phase}',flush=True)
    summary=[]
    for mode in ('graph','eager'):
        for repeat in range(1,args.repeats):
            left=[e for e in events if e['phase']==f'{mode}-0' and 'sha256' in e]
            right=[e for e in events if e['phase']==f'{mode}-{repeat}' and 'sha256' in e]
            summary.append(dict(mode=mode,repeat=repeat,comparisons=[dict(operation=a['name'],
                exact=a['sha256']==b['sha256']) for a,b in zip(left,right)]))
    save('summary.json',summary)
    model.unload()


if __name__=='__main__': main()
