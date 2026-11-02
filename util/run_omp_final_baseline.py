"""Limited old-wheel default control immediately before release packaging."""
import importlib.metadata
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
        index=BASE/'concurrency-rechecks.json'
        rows=json.loads(index.read_text()) if index.exists() else []
        assert not any(r['code'] for r in rows),'concurrency recheck failed'
        if len(rows)==2:break
        if time.monotonic()>deadline:raise TimeoutError('rechecks incomplete')
        time.sleep(5)
    assert importlib.metadata.version('exllamav3')=='1.6.6+nativeplan.v2.tp.moe1.ds41.1'
    log=BASE/'old-release-final-control.log';assert not log.exists()
    env=dict(os.environ);env.pop('EXL3_OMP_BLOCKTIME_MS',None)
    command=[sys.executable,'-I',str(ROOT/'util/deepseek_v41_experiments.py'),'--phase','omp-implemented',
             '--skip-long','--pairs','3','--output',str(BASE/'old-release-final-control')]
    print('START old-release-final-control',flush=True)
    with log.open('w',encoding='utf-8') as output:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    (BASE/'old-release-final-control-run.json').write_text(json.dumps(dict(command=command,code=result.returncode),indent=2))
    print('END old-release-final-control',result.returncode,flush=True);result.check_returncode()


if __name__=='__main__':main()
