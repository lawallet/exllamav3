"""Single-GPU parity and timing control for the two-GPU power-state confound."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'native-results/laguna/performance-control'
out.mkdir(exist_ok=False)
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert k.SetProcessAffinityMask(k.GetCurrentProcess(),0xff)
runs=[]
for name,args in [('short',['--steps','4']),('performance',['--phase','benchmark','--context','512','--steps','256','--runs','5'])]:
    cmd=[sys.executable,str(ROOT/'util/validate_laguna.py'),'--output',str(out/name),'--budgets','90',*args]
    start=time.time()
    with (out/(name+'.log')).open('w') as log:
        r=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NATIVE_PLAN='1'),stdout=log,stderr=subprocess.STDOUT)
    runs.append(dict(name=name,command=cmd,returncode=r.returncode,seconds=time.time()-start,cpu_affinity='0xff'))
    (out/'runs.json').write_text(json.dumps(runs,indent=2))
    print(name,r.returncode,flush=True)
    if r.returncode: raise SystemExit(r.returncode)
