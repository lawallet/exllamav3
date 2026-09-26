"""Validate dynamic batches before alternating throughput/latency benchmarks."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from util.check_glm_concurrency import check_jobs,run_jobs
from util.check_glm_native import check
from util.check_glm_mtp import check_mtp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extension-dir',type=Path)
    p.add_argument('--package-dir',type=Path)
    p.add_argument('--trace-only',action='store_true')
    p.add_argument('--suite',choices=['ordinary','mtp'],required=True)
    p.add_argument('--case',action='append',help='Run only named cases (for targeted follow-up validation)')
    p.add_argument('--regressions',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--skip-parity',action='store_true')
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--cache-tokens',type=int,default=5120)
    a=p.parse_args()
    if a.runs<3: p.error("At least three alternating pairs are required")
    if a.trace_only and a.benchmark: p.error("Trace-only cannot authorize a benchmark")
    if a.package_dir: sys.path.insert(0,str(a.package_dir.resolve()))
    elif a.extension_dir:
        sys.path.insert(1,str(a.extension_dir.resolve()))
    import torch,exllamav3,exllamav3_ext
    from exllamav3.version import __version__
    from exllamav3 import Config,Model,Cache,Tokenizer,Generator
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,0'
    if not a.trace_only: assert getattr(exllamav3_ext,'native_glm_max_batch',0)==4
    torch.cuda.set_device(0)
    cfg=Config.from_directory(a.model)
    assert cfg.arch_string=='Glm5NextForConditionalGeneration' and cfg.num_mtp_layers==1
    model=Model.from_config(cfg)
    draft=Model.from_config(cfg,component='mtp') if a.suite=='mtp' else None
    slots=3 if draft else 4
    cache=Cache(model,a.cache_tokens,max_batch_size=slots,max_history=3 if draft else 0)
    dc=Cache(draft,a.cache_tokens,max_batch_size=slots) if draft else None
    tokenizer=Tokenizer.from_config(cfg)
    if draft: draft.load(max_batch_size=slots,max_chunk_size=256)
    model.load(use_per_device=[90,44],max_batch_size=slots,max_chunk_size=256)
    if draft: draft.attach_to(model)
    a.output.mkdir(parents=True,exist_ok=True)
    meta=dict(package=exllamav3.__file__,extension=exllamav3_ext.__file__,
              version=__version__,installed_distribution=importlib.metadata.version('exllamav3'),
              torch=torch.__version__,cuda=torch.version.cuda,
              revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              environment={k:v for k,v in os.environ.items() if k.startswith(('EXL3_','CUDA_'))},
              mapping='2,0',cache_tokens=a.cache_tokens,cache_slots=slots,max_history=3 if draft else 0,
              sampler='greedy except explicit stochastic parity',seeds=[1234,1235,1236,1237,1238])
    meta.update(python=sys.version,dependencies={n:importlib.metadata.version(n) for n in ('triton-windows','flash-linear-attention')},gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,power.limit,driver_version','--format=csv'],text=True))
    (a.output/'model-config.json').write_text((Path(a.model)/'config.json').read_text(encoding='utf-8'),encoding='utf-8')
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (a.output/'placement.json').write_text(json.dumps({role:[dict(key=m.key,device=str(m.device)) for m in mod.modules] for role,mod in [('target',model)]+([('draft',draft)] if draft else [])},indent=2))
    import threading,time
    def telemetry():
        with (a.output/'telemetry.jsonl').open('w') as f:
            while True:
                r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,power.draw,memory.used,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True)
                f.write(json.dumps(dict(time=time.time(),returncode=r.returncode,csv=r.stdout,error=r.stderr))+'\n');f.flush();time.sleep(1)
    threading.Thread(target=telemetry,daemon=True).start()
    common=(model,draft,cache,dc,tokenizer)
    cases=[]
    if a.suite in ('ordinary','all'):
        for n in (2,3,4):
            cases.append((f'ordinary-{n}',dict(mtp=False,count=n,lengths=[24,40,56,72][:n])))
        cases.extend([
            ('ordinary-stagger',dict(mtp=False,count=4,lengths=[24,40,56,88,32],stagger=True)),
            ('ordinary-stochastic',dict(mtp=False,count=2,lengths=[32,48],stochastic=True)),
            ('ordinary-page-crossing',dict(mtp=False,count=2,lengths=[256,272])),
            ('ordinary-cancel',dict(mtp=False,count=4,lengths=[48,40,56,80,32],stagger=True,cancel=True)),
        ])
        if a.cache_tokens==1024:
            cases.append(('ordinary-page-reuse',dict(mtp=False,count=4,lengths=[24,40,56,88,32],
                          stagger=True,require_page_reuse=True)))
    if a.suite in ('mtp','all'):
        cases.extend([
            ('mtp-2',dict(mtp=True,count=2,lengths=[32,64])),
            ('mtp-3',dict(mtp=True,count=3,lengths=[24,40,64])),
            ('mtp-stagger-forced',dict(mtp=True,count=3,lengths=[24,40,72,32],stagger=True,force=True)),
            ('mtp-page-crossing',dict(mtp=True,count=2,lengths=[64,88],prefix_tokens=224)),
            ('mtp-cancel',dict(mtp=True,count=3,lengths=[48,40,56,24] if a.cache_tokens==768 else [48,64,96,40],stagger=True,cancel=True,force=True,require_page_reuse=a.cache_tokens==768)),
        ])
    if a.case:
        if set(a.case)-{name for name,_ in cases}:
            p.error('Unknown case or case outside selected suite')
        cases=[(name,kw) for name,kw in cases if name in a.case]
    for name,kw in cases:
        if a.trace_only:
            kw=dict(kw,lengths=[6,10,14,18,8][:len(kw['lengths'])])
            events,report=run_jobs(*common,native=False,correctness=True,**kw)
            (a.output/(name+'-stock-trace.json')).write_text(json.dumps(dict(events=events,report=report)))
        elif not a.skip_parity:
            check_jobs(*common,a.output/name,**kw)
        elif not (a.output/name/'parity.json').exists():
            raise ValueError(f'Missing exact parity result for {name}')
    if a.regressions and not draft:
        prompt=(ROOT/'doc/native_glm53_prompt.txt').read_text(encoding='utf-8')
        result=check(model,cache,tokenizer,prompt,steps=256)
        (a.output/'non-mtp-regression.json').write_text(json.dumps(result,indent=2))
    if a.benchmark:
        required=(['ordinary-2','ordinary-3','ordinary-4','ordinary-stagger','ordinary-stochastic','ordinary-page-crossing','ordinary-cancel'] if not draft else ['mtp-2','mtp-3','mtp-stagger-forced','mtp-page-crossing','mtp-cancel'])
        for name in required:
            proof=a.output/name/'parity.json'
            if not proof.exists() or not json.loads(proof.read_text())['exact']:
                raise ValueError(f'Benchmark requires exact parity: {name}')
        runs=[]
        for mtp,n,stagger in ([(False,n,False) for n in (1,2,3,4)]+[(False,4,True)] if not draft else [(True,n,False) for n in (1,2,3)]+[(True,3,True)]):
            lengths=([1000] if n==1 else [384,512,640,768][:n])
            if stagger:lengths.append(256)
            # Warm the actual cohort, including shape changes/page boundaries,
            # so a longer-context specialization is not first compiled in a timed run.
            warm={}
            for native in (False,True):
                _,warm[native]=run_jobs(*common,native=native,mtp=mtp,count=n,lengths=lengths,stagger=stagger,diagnostic=True)
                (a.output/f'warmup-{mtp}-{n}-{stagger}-{native}.json').write_text(json.dumps(warm[native],indent=2))
            assert {k:v for k,v in warm[False]['calls'].items() if k.startswith('transfer/')} == {k:v for k,v in warm[True]['calls'].items() if k.startswith('transfer/')}
            outputs=[]
            for repeat in range(a.runs):
                for native in ((False,True) if repeat%2==0 else (True,False)):
                    _,r=run_jobs(*common,native=native,mtp=mtp,count=n,lengths=lengths,stagger=stagger)
                    assert r['shapes']==warm[native]['shapes']
                    assert {j:(v['ids'],v['rng'],v['accepted'],v['rejected']) for j,v in r['per_job'].items()} == {j:(v['ids'],v['rng'],v['accepted'],v['rejected']) for j,v in warm[native]['per_job'].items()}
                    r['calls']=warm[native]['calls']
                    r['call_count_source']='separate full-cohort instrumented warmup, matching shapes and outputs'
                    r['repeat']=repeat
                    outputs.append({j:{k:v[k] for k in ('ids','rng','proposed','accepted','rejected','completion_sha256')}
                                    for j,v in r['per_job'].items()})
                    runs.append(r)
                    (a.output/'benchmark.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
                    print('BENCH',mtp,n,stagger,native,r['aggregate_tps'],flush=True)
            assert all(o==outputs[0] for o in outputs),'Benchmark per-job outputs/acceptance/RNG differ'
        (a.output/'benchmark-parity.json').write_text(json.dumps(dict(exact=True,runs=len(runs))),encoding='utf-8')


if __name__=='__main__':
    main()
