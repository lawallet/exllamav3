"""Serialize GPU cohorts; preserve each run and keep model processes isolated."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-sweep',action='store_true')
    p.add_argument('--repeats',type=int,default=0)
    p.add_argument('--models',nargs='*',default=['qwen','glm','nemotron','deepseek4','gemma','devstral','glm-tp'])
    p.add_argument('--label',default='baseline')
    p.add_argument('--package-dir')
    p.add_argument('--implemented',type=int)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--complete-qwen',action='store_true')
    a=p.parse_args()
    if a.wait_sweep:
        server=json.loads((BASE/'deepseek-sweep/server.json').read_text())
        import psutil
        deadline=time.monotonic()+7200
        while psutil.pid_exists(server['pid']):
            if time.monotonic()>deadline:raise TimeoutError('sweep server still alive')
            time.sleep(2)
        assert (BASE/'deepseek-sweep/final-status.json').exists(),'sweep failed'
    runs=[]
    def run(name,command,env):
        log=BASE/(name+'.log')
        assert not log.exists(),log
        start=time.time_ns();print('START',name,flush=True)
        with log.open('w',encoding='utf-8') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            code=child.wait()
        runs.append(dict(name=name,command=command,pid=child.pid,code=code,start_ns=start,end_ns=time.time_ns()))
        (BASE/(a.label+'-runs.json')).write_text(json.dumps(runs,indent=2))
        print('END',name,code,flush=True)
        if code:raise SystemExit(code)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='')
    env.pop('EXL3_OMP_BLOCKTIME_MS',None)
    for i in range(a.repeats):
        name=f'{a.label}-deepseek-repeat{i}'
        run(name,[sys.executable,'-I',str(ROOT/'util/deepseek_v41_experiments.py'),'--phase','omp-repeat','--output',str(BASE/name)],env)
    if a.cpu:
        name=a.label+'-cpu'
        command=[sys.executable,'-I',str(ROOT/'util/benchmark_omp_cpu.py'),'--output',str(BASE/name)]
        flags=dict(env)
        if a.package_dir:command+=['--package-dir',a.package_dir]
        if a.implemented is not None:
            flags['EXL3_OMP_BLOCKTIME_MS']=str(a.implemented)
            command+=['--implemented',str(a.implemented)]
        run(name,command,flags)
    for case in a.models:
        name=a.label+'-'+case
        flags=dict(env,EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',EXL3_NATIVE_QWEN_BLOCK='1',EXL3_NATIVE_MOE_ASSIGN='1')
        flags.pop('EXL3_NGRAM_STREAM',None)
        if case=='glm-tp':flags['CUDA_VISIBLE_DEVICES']='2,0'
        command=[sys.executable,'-I',str(ROOT/'util/benchmark_omp_models.py'),'--case',case,'--output',str(BASE/name)]
        if a.package_dir:command+=['--package-dir',a.package_dir]
        if a.implemented is not None:
            flags['EXL3_OMP_BLOCKTIME_MS']=str(a.implemented)
            command+=['--implemented','--settings',str(a.implemented)]
        run(name,command,flags)
    if a.complete_qwen:
        name=a.label+'-qwen-tail'
        flags=dict(env,EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',EXL3_NATIVE_QWEN_BLOCK='1',EXL3_NATIVE_MOE_ASSIGN='1')
        run(name,[sys.executable,'-I',str(ROOT/'util/benchmark_omp_models.py'),'--case','qwen','--output',str(BASE/name),'--tail-only'],flags)


if __name__=='__main__':main()
