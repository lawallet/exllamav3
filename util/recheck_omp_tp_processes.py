"""Matched-warmup A/B/A check of a TP between-process performance difference."""
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
    while not (BASE/'old-release-final-control-run.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('final baseline incomplete')
        time.sleep(5)
    assert json.loads((BASE/'old-release-final-control-run.json').read_text())['code']==0
    records=[]
    for label,implemented in (('candidate-tp-recheck-a',True),('baseline-tp-recheck',False),('candidate-tp-recheck-b',True)):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',
                 EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',EXL3_NATIVE_QWEN_BLOCK='1',EXL3_NATIVE_MOE_ASSIGN='1')
        env.pop('EXL3_OMP_BLOCKTIME_MS',None);env.pop('EXL3_NGRAM_STREAM',None)
        command=[sys.executable,'-I',str(ROOT/'util/benchmark_omp_models.py'),'--case','glm-tp',
            '--output',str(BASE/label),'--short-only','--warmup-rounds','4','--rounds','5','--settings','0']
        if implemented:
            env['EXL3_OMP_BLOCKTIME_MS']='0'
            command+=['--package-dir',str(BASE/'candidate-stage'),'--implemented']
        log=BASE/(label+'.log');assert not log.exists()
        print('START',label,flush=True)
        with log.open('w',encoding='utf-8') as output:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        records.append(dict(name=label,command=command,code=result.returncode))
        (BASE/'tp-process-rechecks.json').write_text(json.dumps(records,indent=2))
        print('END',label,result.returncode,flush=True);result.check_returncode()


if __name__=='__main__':main()
