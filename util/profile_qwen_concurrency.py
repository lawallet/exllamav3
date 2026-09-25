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
from util.check_qwen_concurrency import check_jobs,run_jobs
from util.check_qwen_native import check
from util.check_qwen_mtp import check_mtp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=r'D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extension-dir',type=Path)
    p.add_argument('--installed',action='store_true')
    p.add_argument('--suite',choices=['ordinary','mtp','all'],default='all')
    p.add_argument('--case',action='append',help='Run only named cases (for targeted follow-up validation)')
    p.add_argument('--regressions',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--skip-parity',action='store_true')
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--cache-tokens',type=int,default=5120)
    a=p.parse_args()
    if a.installed:
        sys.path=[s for s in sys.path if Path(s or '.').resolve()!=ROOT]
    elif a.extension_dir:
        sys.path.insert(1,str(a.extension_dir.resolve()))
    import torch,exllamav3,exllamav3_ext
    from exllamav3.version import __version__
    from exllamav3 import Config,Model,Cache,Tokenizer,Generator
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2,0,1'
    assert getattr(exllamav3_ext,'native_qwen_max_batch',0)==4
    torch.cuda.set_device(0)
    cfg=Config.from_directory(a.model)
    assert cfg.arch_string=='Qwen4ExpForConditionalGeneration' and cfg.mtp_num_hidden_layers==1
    model,draft=Model.from_config(cfg),Model.from_config(cfg,component='mtp')
    cache=Cache(model,a.cache_tokens,max_batch_size=4,max_history=4)
    dc=Cache(draft,a.cache_tokens,max_batch_size=4)
    tokenizer=Tokenizer.from_config(cfg)
    draft.load(device=torch.device('cuda:0'))
    model.load(device=torch.device('cuda:0'))
    draft.attach_to(model)
    a.output.mkdir(parents=True,exist_ok=True)
    meta=dict(package=exllamav3.__file__,extension=exllamav3_ext.__file__,
              version=__version__,installed_distribution=importlib.metadata.version('exllamav3'),
              torch=torch.__version__,cuda=torch.version.cuda,
              revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              environment={k:v for k,v in os.environ.items() if k.startswith(('EXL3_','CUDA_'))},
              mapping='2,0,1',cache_tokens=a.cache_tokens,cache_slots=4,max_history=4,
              sampler='greedy except explicit stochastic parity',seeds=[1234,1235,1236,1237,1238])
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    common=(model,draft,cache,dc,tokenizer)
    cases=[]
    if a.suite in ('ordinary','all'):
        for n in (2,3,4):
            cases.append((f'ordinary-{n}',dict(mtp=False,count=n,lengths=[24,40,56,72][:n])))
        cases.extend([
            ('ordinary-stagger',dict(mtp=False,count=4,lengths=[24,40,56,88,32],stagger=True)),
            ('ordinary-stochastic',dict(mtp=False,count=2,lengths=[32,48],stochastic=True)),
            ('ordinary-page-crossing',dict(mtp=False,count=2,lengths=[256,272])),
        ])
        if a.cache_tokens==1024:
            cases.append(('ordinary-page-reuse',dict(mtp=False,count=4,lengths=[24,40,56,88,32],
                          stagger=True,require_page_reuse=True)))
    if a.suite in ('mtp','all'):
        cases.extend([
            ('mtp-2',dict(mtp=True,count=2,lengths=[32,64])),
            ('mtp-3',dict(mtp=True,count=3,lengths=[24,40,64])),
            ('mtp-stagger-forced',dict(mtp=True,count=3,lengths=[24,40,72,32],stagger=True,force=True)),
            ('mtp-page-crossing',dict(mtp=True,count=2,lengths=[256,280])),
        ])
    if a.case:
        if set(a.case)-{name for name,_ in cases}:
            p.error('Unknown case or case outside selected suite')
        cases=[(name,kw) for name,kw in cases if name in a.case]
    for name,kw in cases:
        if not a.skip_parity:
            check_jobs(*common,a.output/name,**kw)
        elif not (a.output/name/'parity.json').exists():
            raise ValueError(f'Missing exact parity result for {name}')
    if a.regressions:
        prompt=(ROOT/'doc/native_qwen_prompt.txt').read_text(encoding='utf-8')
        result=check(model,cache,tokenizer,prompt)
        (a.output/'non-mtp-regression.json').write_text(json.dumps(result,indent=2))
        def make_gen():
            return Generator(model,cache,tokenizer,max_batch_size=1,draft_model=draft,draft_cache=dc,
                             num_draft_tokens=4,dynamic_draft_tokens=False,record_draft_stats=True)
        check_mtp(model,draft,cache,dc,tokenizer,prompt,make_gen,a.output/'natural-regression',tokens=128)
        check_mtp(model,draft,cache,dc,tokenizer,prompt,make_gen,a.output/'forced-regression',tokens=64,force_rejections=True)
    if a.benchmark:
        required=('ordinary-2','ordinary-3','ordinary-4','ordinary-stagger','ordinary-stochastic',
                  'ordinary-page-crossing','mtp-2','mtp-3','mtp-stagger-forced','mtp-page-crossing')
        for name in required:
            proof=a.output/name/'parity.json'
            if not proof.exists() or not json.loads(proof.read_text())['exact']:
                raise ValueError(f'Benchmark requires exact parity: {name}')
        runs=[]
        for mtp,n,stagger in [(False,n,False) for n in (1,2,3,4)]+[(True,n,False) for n in (2,3)]+[(False,4,True),(True,3,True)]:
            lengths=[384,512,640,768][:n]
            if stagger:lengths.append(256)
            # Warm the actual cohort, including shape changes/page boundaries,
            # so a longer-context specialization is not first compiled in a timed run.
            for native in (False,True):
                run_jobs(*common,native=native,mtp=mtp,count=n,lengths=lengths,stagger=stagger)
            outputs=[]
            for repeat in range(a.runs):
                for native in ((False,True) if repeat%2==0 else (True,False)):
                    _,r=run_jobs(*common,native=native,mtp=mtp,count=n,lengths=lengths,stagger=stagger)
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
