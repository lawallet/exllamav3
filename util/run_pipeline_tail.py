"""Run corrected-order replication and separate profiling after cross-model gates."""
import argparse,json,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/generation-pipeline'
def main():
    p=argparse.ArgumentParser();p.add_argument('--package-dir',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);a=p.parse_args()
    out=BASE/'tail-validation';out.mkdir(exist_ok=False)
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        try:progress=json.loads((a.after/'progress.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):progress=[]
        assert not any(r['exit_code'] for r in progress),'cross-model gate failed'
        if len(progress)==7:break
        time.sleep(3)
    else:raise TimeoutError('cross-model campaign incomplete')
    commands=[('qwen-order-recheck',[sys.executable,'-I',str(ROOT/'util/benchmark_pipeline_models.py'),
        '--case','qwen','--output',str(BASE/'qwen-order-recheck'),'--package-dir',str(a.package_dir.resolve()),
        '--settings','0','1','--rounds','5','--tokens','256','--primary-only']),
        ('product-profile',[sys.executable,'-I',str(ROOT/'util/deepseek_v41_pipeline.py'),
        '--phase','candidate-profile','--package-dir',str(a.package_dir.resolve()),'--output',str(BASE/'product-profile-p3')])]
    results=[]
    for name,command in commands:
        print('START '+name,flush=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NATIVE_PLAN='1',EXL3_NATIVE_MOE_ASSIGN='1',
                 EXL3_OMP_BLOCKTIME_MS='0',EXL3_NGRAM_STREAM='1',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
        with (out/(name+'.log')).open('w') as log:
            r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        results.append(dict(name=name,command=command,exit_code=r.returncode));(out/'progress.json').write_text(json.dumps(results,indent=2))
        print('END '+name+' '+str(r.returncode),flush=True);r.check_returncode()
        if name=='qwen-order-recheck':
            medians=[]
            for enabled in (0,1):
                rows=[json.loads(p.read_text()) for p in (BASE/'qwen-order-recheck').glob(f'steady-*-cache{enabled}.json')]
                medians.append(statistics.median(x['jobs'][0]['new_tokens']/x['jobs'][0]['time_generate'] for x in rows))
            with (ROOT/'doc/deepseek_v41_ir_investigation.md').open('a',encoding='utf-8') as report:
                report.write(f'\nCorrected-order fresh Qwen replication completed: five alternating pairs, '
                    f'{medians[0]:.3f} -> {medians[1]:.3f} tok/s ({100*(medians[1]/medians[0]-1):+.2f}%), '
                    'exact full tokens/MTP. Evidence: `generation-pipeline/qwen-order-recheck`.\n')
    (out/'complete.json').write_text(json.dumps(dict(success=True),indent=2))
if __name__=='__main__':main()
