"""Sequential fresh-process validation; stops at the first failed cohort."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/attention-optimization'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cases',nargs='+',required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    package=a.package_dir.resolve();commands=[]
    for name in a.cases:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NATIVE_PLAN='1',EXL3_OMP_BLOCKTIME_MS='0',
            EXL3_STOP_MASK_CACHE='1',EXL3_ASYNC_DRAFT_SEED='1',EXL3_NATIVE_DSV41_PROJECTIONS='1',
            EXL3_NGRAM_STREAM='1',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
        if name.startswith('cross-'):
            case=name[6:];target=BASE/'cross-model'/case
            if case=='glm-tp':env['CUDA_VISIBLE_DEVICES']='2,0'
            command=[sys.executable,'-I',str(ROOT/'util/benchmark_projection_models.py'),'--case',case,
                '--package-dir',str(package),'--output',str(target),'--implemented','--tokens','64',
                '--warmup-rounds','0','--tail-only','--concurrency-tail','--no-telemetry']
        else:
            target=BASE/('native-'+name)
            phase='profile' if name.startswith('profile-') else name
            if name=='seeded-telemetry':phase='pairs'
            command=[sys.executable,'-I',str(ROOT/'util/deepseek_v41_attention.py'),'--phase',phase,
                '--implemented','--package-dir',str(package),'--output',str(target)]
            if name=='profile-candidate':command.append('--profile-candidate')
            if name=='telemetry':command.append('--telemetry')
            if name=='seeded-telemetry':command.extend(['--seed','841','--minimum','0','--telemetry','--pairs','5'])
        assert not target.exists(),target
        commands.append(dict(name=name,command=command,start_ns=time.time_ns(),output=str(target)))
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        commands[-1].update(end_ns=time.time_ns(),exit_code=result.returncode)
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        print('END '+name+' '+str(result.returncode),flush=True)
        if result.returncode:raise SystemExit(result.returncode)
    (out/'complete.json').write_text(json.dumps(dict(cases=a.cases,success=True),indent=2))
if __name__=='__main__':main()
