"""Scope-matched Gemma checks for additive v2 residual instructions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--extension-dir',type=Path,default=ROOT/'native-results/devstral2/extension')
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
jobs=[
    ('sites-201','2,0,1','check_gemma4_v2_sites.py',[]),
    ('sites-20','2,0','check_gemma4_v2_sites.py',[]),
    ('short-201','2,0,1','validate_gemma4_v2.py',['--steps','8']),
    ('reuse-201','2,0,1','validate_gemma4_v2.py',['--phase','generator','--case','5','--context','240','--steps','48','--cache-tokens','1536']),
    ('short-20','2,0','validate_gemma4_v2.py',['--steps','8','--reference-dir',str(a.output/'short-201')]),
]
runs=[]
for name,visibility,script,args in jobs:
    command=[sys.executable,str(ROOT/'util'/script),'--extension-dir',str(a.extension_dir.resolve()),'--output',str(a.output/name),*args]
    print('START',name,flush=True);started=time.monotonic()
    with (a.output/(name+'.log')).open('w') as log:
        r=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=visibility,EXL3_NATIVE_PLAN='1'),stdout=log,stderr=subprocess.STDOUT)
    runs.append(dict(name=name,command=command,visibility=visibility,returncode=r.returncode,seconds=time.monotonic()-started))
    (a.output/'runs.json').write_text(json.dumps(runs,indent=2))
    print('END',name,r.returncode,flush=True)
    if r.returncode: raise SystemExit(r.returncode)
