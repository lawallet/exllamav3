"""Repeat two suspected regressions after the serialized staged-model gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def main():
    deadline=time.monotonic()+14400
    while True:
        path=BASE/'candidate-validation-runs.json'
        runs=json.loads(path.read_text()) if path.exists() else []
        assert not any(r['code'] for r in runs),'candidate validation failed'
        if any(r['name']=='candidate-matrix' for r in runs):break
        if time.monotonic()>deadline:raise TimeoutError('candidate gates did not complete')
        time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',
             EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',EXL3_NATIVE_QWEN_BLOCK='1',EXL3_NATIVE_MOE_ASSIGN='1')
    env.pop('EXL3_OMP_BLOCKTIME_MS',None);env.pop('EXL3_NGRAM_STREAM',None)
    results=[]
    for case in ('qwen','deepseek4'):
        name='baseline-concurrent-recheck-'+case;path=BASE/(name+'.log');assert not path.exists()
        command=[sys.executable,'-I',str(ROOT/'util/benchmark_omp_models.py'),'--case',case,
                 '--output',str(BASE/name),'--concurrent-only','--rounds','5','--settings','1','200','0']
        print('START',name,flush=True)
        with path.open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        results.append(dict(name=name,command=command,code=result.returncode))
        (BASE/'concurrency-rechecks.json').write_text(json.dumps(results,indent=2))
        print('END',name,result.returncode,flush=True);result.check_returncode()


if __name__=='__main__':main()
