"""Ordered acceptance campaign; stop at the first failed correctness gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--start-at', default='sites-201')
a = p.parse_args()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
jobs = [
    ('sites-201', '2,0,1', 'check_gemma4_v2_sites.py', []),
    ('sites-20', '2,0', 'check_gemma4_v2_sites.py', []),
    ('short-201', '2,0,1', 'validate_gemma4_v2.py', ['--reload']),
    ('generator-201', '2,0,1', 'validate_gemma4_v2.py', ['--phase', 'generator', '--context', '240', '--steps', '48']),
    ('reuse-201', '2,0,1', 'validate_gemma4_v2.py', ['--phase', 'generator', '--context', '240', '--steps', '48', '--case', '4', '--case', '5', '--cache-tokens', '1536']),
    ('short-20', '2,0', 'validate_gemma4_v2.py', ['--reload', '--reference-dir', str(a.output/'short-201')]),
    ('remapped-02', '0,2', 'validate_gemma4_v2.py', ['--budgets', '16', '50', '--steps', '16']),
    ('long-8192', '2,0,1', 'validate_gemma4_v2.py', ['--phase', 'long', '--context', '8192', '--steps', '16']),
    ('performance', '2,0,1', 'validate_gemma4_v2.py', ['--phase', 'benchmark', '--context', '512', '--steps', '256', '--runs', '5']),
]
records = []
assert a.start_at in {job[0] for job in jobs}
started = False
for name, visibility, script, args in jobs:
    if name == a.start_at: started = True
    if not started: continue
    cmd = [sys.executable, str(ROOT/'util'/script), '--output', str(a.output/name), *args]
    print('START', name, flush=True)
    start = time.monotonic()
    with (a.output/(name+'.log')).open('w') as log:
        r = subprocess.run(cmd, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=visibility, EXL3_NATIVE_PLAN='1'),
            stdout=log, stderr=subprocess.STDOUT)
    records.append(dict(name=name, command=cmd, visibility=visibility,
                        returncode=r.returncode, seconds=time.monotonic()-start))
    (a.output/'runs.json').write_text(json.dumps(records, indent=2))
    print('END', name, r.returncode, flush=True)
    if r.returncode: raise SystemExit(r.returncode)
