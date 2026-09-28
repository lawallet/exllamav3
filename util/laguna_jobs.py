"""Sequential, fail-fast Laguna acceptance and wheel smoke driver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--start-at', default='inspection')
p.add_argument('--package', type=Path)
a = p.parse_args()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
jobs = [
    ('inspection','2,0,1','inspect_laguna_ir.py',[]),
    ('guards','2,0,1',None,[]),
    ('sites','2,0,1','check_laguna_sites.py',[]),
    ('short-201','2,0,1','validate_laguna.py',['--reload']),
    ('generator','2,0,1','validate_laguna.py',['--phase','generator','--context','480','--steps','48']),
    ('reuse','2,0,1','validate_laguna.py',['--phase','generator','--context','240','--steps','48','--case','4','--case','5','--cache-tokens','1536']),
    ('short-20','2,0','validate_laguna.py',['--reload','--reference-dir',str(a.output/'short-201')]),
    ('remapped','0,2','validate_laguna.py',['--budgets','16','70','--steps','16']),
    ('long','2,0,1','validate_laguna.py',['--phase','long','--context','8192','--steps','16']),
    ('performance','2,0,1','validate_laguna.py',['--phase','benchmark','--context','512','--steps','256','--runs','5']),
    ('gemma-smoke','2,0,1','validate_gemma4_v2.py',['--steps','4']),
    ('deepseek-smoke','2,0,1','validate_deepseek_plan.py',['--steps','4']),
]
if a.package:
    jobs = [
        ('sites','2,0,1','check_laguna_sites.py',[]),
        ('short-201','2,0,1','validate_laguna.py',['--steps','8','--reload']),
        ('short-20','2,0','validate_laguna.py',['--steps','8','--reference-dir',str(a.output/'short-201')]),
    ]
    a.start_at = 'sites'
records = []
started = False
for name, visibility, script, args in jobs:
    if name == a.start_at: started = True
    if not started: continue
    if script is None:
        cmd = [sys.executable,'-m','unittest','discover','-s','tests','-p','test_native*.py']
    elif a.package:
        cmd = [sys.executable,str(ROOT/'util/verify_laguna_wheel.py'),'--package',str(a.package.resolve()),
               '--script',script,'--','--output',str(a.output/name),*args]
    else:
        dest = a.output/(name+'.json') if name=='inspection' else a.output/name
        cmd = [sys.executable,str(ROOT/'util'/script),'--output',str(dest),
               '--extension-dir',str(ROOT/'native-results/devstral2/extension'),*args]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visibility, EXL3_NATIVE_PLAN='1',
               PYTHONPATH=str(ROOT/'native-results/devstral2/extension')+os.pathsep+str(ROOT))
    if script is None: env['EXL3_NATIVE_PLAN'] = '0'  # Unit fixtures toggle/mock backends themselves.
    print('START',name,flush=True)
    start = time.monotonic()
    with (a.output/(name+'.log')).open('w') as log:
        r = subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(name=name,command=cmd,visibility=visibility,returncode=r.returncode,seconds=time.monotonic()-start))
    (a.output/'runs.json').write_text(json.dumps(records,indent=2))
    print('END',name,r.returncode,flush=True)
    if r.returncode: raise SystemExit(r.returncode)
(a.output/'passed.json').write_text(json.dumps(dict(all_selected_gates_passed=True)))
