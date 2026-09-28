"""Fail-fast cross-architecture validation of an extracted candidate wheel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--package', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--only', nargs='+', help='Resume selected gates; does not certify omitted gates')
a = p.parse_args()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
jobs = [
    ('devstral-sites', '2,0,1', 'check_devstral2_sites.py', []),
    ('devstral-short', '2,0,1', 'validate_devstral2.py', ['--steps','8','--reload']),
    ('devstral-topology', '2,0', 'validate_devstral2.py', ['--steps','8','--reference-dir',str(a.output/'devstral-short')]),
    ('gemma-sites', '2,0,1', 'check_gemma4_v2_sites.py', []),
    ('gemma-short', '2,0,1', 'validate_gemma4_v2.py', ['--steps','8']),
    ('deepseek-short', '2,0,1', 'validate_deepseek_plan.py', ['--steps','8']),
    ('qwen-short', '2,0,1', 'profile_qwen_decode.py', ['--model',r'D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw',
        '--prompt-file','doc/native_qwen_prompt.txt','--device','cuda:0','--ab','--compare-plan','--tokens','128','--runs','1']),
    ('glm-short', '2,0,1', 'profile_glm_decode.py', ['--model',r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw',
        '--split','90','44','--ab','--smoke','--compare-plan','--tokens','256','--runs','1']),
]
records = []
for name, visibility, script, args in jobs:
    if a.only and name not in a.only: continue
    command = [sys.executable, str(ROOT/'util/verify_devstral2_wheel.py'),
               '--package',str(a.package.resolve()),'--script',script,'--','--output',str(a.output/name),*args]
    print('START',name,flush=True)
    started = time.monotonic()
    with (a.output/(name+'.log')).open('w') as log:
        r = subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=visibility,EXL3_NATIVE_PLAN='1'),
                           stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(name=name,visibility=visibility,command=command,returncode=r.returncode,seconds=time.monotonic()-started))
    (a.output/'runs.json').write_text(json.dumps(records,indent=2))
    print('END',name,r.returncode,flush=True)
    if r.returncode: raise SystemExit(r.returncode)
(a.output/'passed.json').write_text(json.dumps(dict(selected_package_gates_passed=True, all_package_gates_passed=len(records)==len(jobs))))
