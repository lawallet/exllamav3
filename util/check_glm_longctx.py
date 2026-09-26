"""64K closure gates against the unchanged concurrency wheel; no production patches."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['ordinary','mtp'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package-dir',type=Path,default=ROOT/'native-results/glm53-concurrency/wheel-stage')
    p.add_argument('--prompt-tokens',type=int,default=65216)
    p.add_argument('--tokens',type=int,default=256)
    p.add_argument('--forced',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,0'
    assert a.prompt_tokens+a.tokens+4<=65536
    sys.path.insert(0,str(a.package_dir.resolve()))
    import torch,exllamav3,exllamav3_ext
    from exllamav3 import Config,Model,Cache,Tokenizer
    from exllamav3.modules import TransformerBlock,MLAttention
    from util import check_glm_concurrency as h
    from util.check_glm_native import digest
    mtp=a.suite=='mtp'
    meta=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              package=exllamav3.__file__,extension=exllamav3_ext.__file__,
              extension_sha256=hashlib.sha256(Path(exllamav3_ext.__file__).read_bytes()).hexdigest(),
              base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              cuda=torch.version.cuda,torch=torch.__version__,python=sys.version,
              cache_tokens=65536,pages=256,history=3 if mtp else 0,slots=1,
              budgets=[90,44],gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,power.limit,driver_version','--format=csv'],text=True))
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    cfg=Config.from_directory(r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw')
    model=Model.from_config(cfg)
    draft=Model.from_config(cfg,component='mtp') if mtp else None
    cache=Cache(model,65536,max_batch_size=1,max_history=3 if mtp else 0)
    dc=Cache(draft,65536,max_batch_size=1) if mtp else None
    tokenizer=Tokenizer.from_config(cfg)
    try:
        if draft: draft.load(max_batch_size=1,max_chunk_size=256)
        model.load(use_per_device=[90,44],max_batch_size=1,max_chunk_size=256)
        if draft: draft.attach_to(model)
    except Exception as e:
        (a.output/'allocation-failure.json').write_text(json.dumps(dict(error=repr(e),memory=[torch.cuda.memory_summary(d) for d in (0,1)]),indent=2))
        raise
    placement={role:[dict(key=m.key,device=str(m.device)) for m in mod.modules] for role,mod in [('target',model)]+([('draft',draft)] if draft else [])}
    (a.output/'placement.json').write_text(json.dumps(placement,indent=2))
    for b in model.modules:
        if isinstance(b,TransformerBlock): assert str(b.device)==f'cuda:{0 if b.layer_idx<35 else 1}',placement
    assert str(model.modules[model.logit_layer_idx].device)=='cuda:1'
    if draft: assert all(str(m.device)=='cuda:0' for m in draft.modules)
    print('Loaded exact 35/10 split with 65536-token cache',flush=True)
    # Deterministic varied prose, retaining the model's chat prefix and assistant suffix.
    body='\n'.join(f'Record {i}: river level {i%97}; station {i%31}; note {(i*17)%101}. Explain observations using the recorded measurements.' for i in range(7000))
    encoded=tokenizer.encode(model.default_chat_prompt(body+'\nSummarize the records and explain how to compare changes over time.'),encode_special_tokens=True)
    ids=torch.cat((encoded[:,:a.prompt_tokens-2],encoded[:,-2:]),dim=1)
    assert ids.shape[1]==a.prompt_tokens
    (a.output/'prompt-ids.json').write_text(json.dumps(ids.tolist()))
    meta['prompt_sha256']=digest(ids)
    meta['cache_layers']={role:{str(k):dict(type=type(l).__name__,device=str(l.device),bytes=int(l.storage_size())) for k,l in c.layers.items()} for role,c in [('target',cache)]+([('draft',dc)] if dc else [])}
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    original_encode=tokenizer.encode
    protected=(ROOT/'doc/native_glm53_prompt.txt').read_text(encoding='utf-8')
    def encode(text,*args,**kwargs):
        return ids.clone() if text==protected else original_encode(text,*args,**kwargs)
    common=(model,draft,cache,dc,tokenizer)
    def run(native,correctness):
        sparse=[]; prefill_active=[False]; progress=[0]
        original_kv=h.kv
        # Prefill is stock in both modes. Hash its complete final prefix, then every
        # decode/rewind/repair state. Avoid hashing every intermediate growing prefix.
        def kv(c,j,valid):
            if valid<a.prompt_tokens-1: return dict(valid=valid,prefill_not_compared=True)
            return original_kv(c,j,valid)
        with ExitStack() as scope:
            scope.enter_context(patch.object(tokenizer,'encode',encode))
            scope.enter_context(patch.object(h,'kv',kv))
            scope.enter_context(patch.object(h,'digest',lambda t: 'stock-prefill-not-compared' if prefill_active[0] else digest(t)))
            for role,mod in [('target',model)]+([('draft',draft)] if draft else []):
                for method in ('forward','prefill'):
                    original=getattr(mod,method)
                    def prefill(input_ids,params=None,_f=original,_r=role):
                        past=int(params['cache_seqlens'][0])
                        long_prefill=past<a.prompt_tokens-1
                        previous=prefill_active[0]
                        prefill_active[0]=long_prefill
                        try: return _f(input_ids,params)
                        finally:
                            prefill_active[0]=previous
                            if _r=='target' and long_prefill and past+input_ids.shape[1]>=progress[0]+8192:
                                progress[0]=past+input_ids.shape[1]
                                print(f'prefill native={native}: {progress[0]} tokens',flush=True)
                    scope.enter_context(patch.object(mod,method,prefill))
                if correctness:
                    for b in mod.modules:
                        if not isinstance(b,TransformerBlock) or not isinstance(b.attn,MLAttention): continue
                        attention=b.attn; forward=attention.forward
                        def attention_forward(x,params,*args,_f=forward,_m=attention,_r=role,**kwargs):
                            y=_f(x,params,*args,**kwargs)
                            if not prefill_active[0] and params.get('dsa_topk_indices') is not None:
                                lengths=params['_mla_host_seqlens']; q=x.shape[1]
                                indices=params['dsa_topk_indices'].reshape(x.shape[0]*q,-1).detach().cpu()
                                rows=[]
                                for row,v in enumerate(indices):
                                    selected=v[v>=0]
                                    pos=int(lengths[row//q])+row%q+1
                                    assert not len(selected) or int(selected.max())<pos
                                    rows.append(dict(count=len(selected),indices=digest(selected),visible=pos,tail=pos%_m.index_kpool if _m.index_kpool else 0))
                                sparse.append(dict(role=_r,key=_m.key,shape=list(x.shape),rows=rows))
                            return y
                        scope.enter_context(patch.object(attention,'forward',attention_forward))
            events,report=h.run_jobs(*common,native=native,mtp=mtp,count=1,lengths=[a.tokens],correctness=correctness,force=a.forced,diagnostic=not correctness,deep_state_checks=mtp)
        name='native' if native else 'stock'
        (a.output/(name+'-events.json')).write_text(json.dumps(events))
        (a.output/(name+'-sparse.json')).write_text(json.dumps(sparse))
        (a.output/(name+'-summary.json')).write_text(json.dumps(report,indent=2))
        return events,sparse,report
    with torch.inference_mode():
        stock,ss,sr=run(False,True)
        native,ns,nr=run(True,True)
        assert len(stock)==len(native),(len(stock),len(native))
        for i,(x,y) in enumerate(zip(stock,native)):
            if x!=y:
                (a.output/'first-mismatch.json').write_text(json.dumps(dict(event=i,stock=x,native=y),indent=2))
                raise AssertionError(f'First event mismatch {i}: {x["kind"]}')
        if ss!=ns:
            first=next((i for i,(x,y) in enumerate(zip(ss,ns)) if x!=y),min(len(ss),len(ns)))
            (a.output/'first-sparse-mismatch.json').write_text(json.dumps(dict(event=first,stock=ss[first] if first<len(ss) else None,native=ns[first] if first<len(ns) else None),indent=2))
            raise AssertionError(f'Sparse selection mismatch at event {first}')
        assert ns and all(row['visible']>=a.prompt_tokens for event in ns for row in event['rows']), 'Long-context sparse selection not observed'
        expected_sparse={b.attn.key for b in model.modules if isinstance(b,TransformerBlock) and isinstance(b.attn,MLAttention) and b.attn.indexer_mode is not None}
        assert {event['key'] for event in ns if event['role']=='target'}==expected_sparse
        assert all(row['count']==2048+row['tail'] for event in ns for row in event['rows']), 'Unexpected sparse selection/tail width'
        transfers=lambda r:{k:v for k,v in r['calls'].items() if k.startswith('transfer/')}
        assert transfers(sr)==transfers(nr),'Transfer groups differ'
        for k in ('ids','rng','completion_sha256','proposed','accepted','rejected','rewinds'): assert sr['per_job']['0'][k]==nr['per_job']['0'][k],k
        key=f'target/b1/q{4 if mtp else 1}'
        assert nr['calls'][key+'/native']==nr['calls'][key+'/all']>0
        assert not any(k.startswith('draft/') and k.endswith('/native') for k in nr['calls'])
        if mtp: assert nr['per_job']['0']['accepted']>0 and nr['per_job']['0']['rejected']>0
        result=dict(exact=True,events=len(stock),sparse_events=len(ss),sparse_rows=sum(len(x['rows']) for x in ss),prompt_tokens=a.prompt_tokens,output_tokens=a.tokens,final_sequence_tokens=a.prompt_tokens+a.tokens,page_boundaries=[v for v in range(a.prompt_tokens, a.prompt_tokens+a.tokens) if v%256==0],calls=nr['calls'],transfers_exact=True,rollback_distribution=dict(Counter(nr['per_job']['0']['rewinds'])),per_job=nr['per_job'])
        (a.output/'parity.json').write_text(json.dumps(result,indent=2))
        print('PASS',a.output,flush=True)
        if a.smoke:
            smoke=[]
            for enabled in (False,True):
                with patch.object(tokenizer,'encode',encode):
                    _,r=h.run_jobs(*common,native=enabled,mtp=mtp,count=1,lengths=[a.tokens])
                assert r['per_job']['0']['ids']==nr['per_job']['0']['ids']
                smoke.append(r)
                print('SMOKE',enabled,r['per_job']['0']['decode_tps'],flush=True)
            (a.output/'smoke.json').write_text(json.dumps(smoke,indent=2))

if __name__=='__main__': main()
