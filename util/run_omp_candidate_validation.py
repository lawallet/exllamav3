"""Continue the serialized release gates after the retained baseline queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-baseline',action='store_true')
    args=parser.parse_args()
    deadline=time.monotonic()+10800
    while not args.resume_baseline:
        index=BASE/'baseline-next-runs.json'
        runs=json.loads(index.read_text()) if index.exists() else []
        assert not any(r['code'] for r in runs), 'baseline queue failed; inspect retained logs'
        if any(r['name']=='baseline-next-qwen-tail' and r['code']==0 for r in runs):break
        if time.monotonic()>deadline:raise TimeoutError('baseline queue did not complete')
        time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='',
             EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',EXL3_NATIVE_QWEN_BLOCK='1',EXL3_NATIVE_MOE_ASSIGN='1')
    env.pop('EXL3_OMP_BLOCKTIME_MS',None)
    env.pop('EXL3_NGRAM_STREAM',None)
    records=[]
    def run(name,args,flags=env):
        path=BASE/(name+'.log');assert not path.exists(),path
        command=[sys.executable,'-I',*args]
        print('START',name,flush=True);start=time.time_ns()
        with path.open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env=flags,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        records.append(dict(name=name,command=command,code=result.returncode,start_ns=start,end_ns=time.time_ns()))
        (BASE/'candidate-validation-runs.json').write_text(json.dumps(records,indent=2))
        print('END',name,result.returncode,flush=True)
        result.check_returncode()
    if args.resume_baseline:
        run('baseline-devstral-tail',[str(ROOT/'util/benchmark_omp_models.py'),'--case','devstral',
            '--output',str(BASE/'baseline-devstral-tail'),'--tail-only','--concurrency-tail'])
        run('baseline-rest-matrix',[str(ROOT/'util/run_omp_policy_matrix.py'),'--label','baseline-rest',
            '--models','glm-tp','--complete-qwen'])
    for case in ('glm','nemotron'):
        name='baseline-masked-'+case
        run(name,[str(ROOT/'util/benchmark_omp_models.py'),'--case',case,'--output',str(BASE/name),
                  '--tail-only','--no-telemetry'])
    run('candidate-deepseek',[str(ROOT/'util/deepseek_v41_experiments.py'),'--phase','omp-implemented',
        '--package-dir',str(BASE/'candidate-stage'),'--runtime-policy','0','--output',str(BASE/'candidate-deepseek')])
    run('candidate-matrix',[str(ROOT/'util/run_omp_policy_matrix.py'),'--label','candidate',
        '--package-dir',str(BASE/'candidate-stage'),'--implemented','0','--cpu',
        '--models','qwen','glm','nemotron','deepseek4','gemma','devstral','glm-tp'])


if __name__=='__main__':main()
