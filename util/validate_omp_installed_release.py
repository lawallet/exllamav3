"""Final fresh-process installed-wheel throughput and unset compatibility gates."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def main():
    proof=json.loads((BASE/'package/proof.json').read_text())
    assert importlib.metadata.version('exllamav3')==proof['version']
    records=[]
    for name,options in (
        ('installed-deepseek',['--runtime-policy','0','--monitor-implemented']),
        ('installed-deepseek-unset',['--skip-long','--pairs','2']),
    ):
        command=[sys.executable,'-I',str(ROOT/'util/deepseek_v41_experiments.py'),'--phase','omp-implemented',
                 '--output',str(BASE/name),*options]
        env=dict(os.environ);env.pop('EXL3_OMP_BLOCKTIME_MS',None)
        log=BASE/(name+'.log');assert not log.exists()
        print('START',name,flush=True)
        with log.open('w',encoding='utf-8') as output:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        records.append(dict(name=name,command=command,code=result.returncode))
        (BASE/'installed-validation-runs.json').write_text(json.dumps(records,indent=2))
        print('END',name,result.returncode,flush=True);result.check_returncode()


if __name__=='__main__':main()
