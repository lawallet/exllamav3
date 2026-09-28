"""Longer alternating timing pairs with read-only GPU telemetry after instability."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import ctypes
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'native-results/laguna/performance-repeat'
out.mkdir(exist_ok=False)
cmd=[sys.executable,str(ROOT/'util/validate_laguna.py'),'--output',str(out/'results'),
     '--phase','benchmark','--context','512','--steps','512','--runs','10']
started=time.time()
# Limit only this isolated benchmark process (and its child) to one fixed CPU
# subset. Do not alter other processes, GPU clocks, power policy or product code.
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),0xff), ctypes.get_last_error()
with (out/'gpu.csv').open('w') as telemetry, (out/'run.log').open('w') as log:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                              '--format=csv','-l','1'],stdout=telemetry,stderr=subprocess.STDOUT)
    try:
        with subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0,1',EXL3_NATIVE_PLAN='1'),
                              stdout=log,stderr=subprocess.STDOUT) as run:
            mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            assert kernel.GetProcessAffinityMask(int(run._handle),ctypes.byref(mask),ctypes.byref(system_mask))
            assert mask.value==0xff, mask.value
            run.wait()
    finally:
        monitor.terminate(); monitor.wait()
(out/'run.json').write_text(json.dumps(dict(command=cmd,returncode=run.returncode,seconds=time.time()-started,
    cpu_affinity_mask='0xff',reason='Initial unrestricted timings were unstable; fixed subset shared by both backends'),indent=2))
raise SystemExit(run.returncode)
