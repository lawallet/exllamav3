"""Run established v1 GPU/guard gates against the additive v2 extension."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--only', nargs='+')
p.add_argument('--extension-dir', type=Path, default=ROOT/'native-results/block-ir-v2/extension-r2')
args = p.parse_args()
OUT = args.output.resolve()
EXT = args.extension_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
qwen = r'D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw'
glm = r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw'
jobs = [
    ('guards', ['-c', "import sys,unittest; sys.path.insert(0,r'%s'); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests',pattern='test_native*.py')); sys.exit(not r.wasSuccessful())" % EXT]),
    ('glm-sites', ['util/check_native_plan_primitives.py']),
    ('deepseek-sites', ['util/check_deepseek_plan_sites.py']),
    ('qwen-short', ['util/profile_qwen_decode.py', '--model', qwen, '--prompt-file', 'doc/native_qwen_prompt.txt', '--device', 'cuda:0', '--ab', '--compare-plan', '--tokens', '128', '--runs', '1']),
    ('glm-short', ['util/profile_glm_decode.py', '--model', glm, '--split', '90', '44', '--ab', '--smoke', '--compare-plan', '--tokens', '128', '--runs', '1']),
    ('deepseek-short', ['util/validate_deepseek_plan.py', '--steps', '16', '--reload']),
    ('qwen-mtp', ['util/profile_qwen_mtp.py', '--model', qwen, '--parity-only', '--parity-tokens', '64']),
    ('glm-mtp', ['util/profile_glm_mtp.py', '--model', glm, '--parity-only', '--parity-tokens', '64', '--forced-tokens', '64']),
    ('deepseek-mtp', ['util/validate_deepseek_plan.py', '--phase', 'generator', '--mtp', '--case', '0', '--case', '8', '--steps', '32']),
]
records = []
for name, command in jobs:
    if args.only and name not in args.only: continue
    if name != 'guards':
        suffix = '.json' if name.endswith('sites') else ''
        command += ['--extension-dir', str(EXT), '--output', str(OUT/(name+suffix))]
    start = time.monotonic()
    print('START', name, flush=True)
    with (OUT/(name+'.log')).open('w') as log:
        r = subprocess.run([sys.executable, *command], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='2,0' if name=='glm-mtp' else '2,0,1',
                     EXL3_NATIVE_PLAN='0' if name=='guards' else '1'),
            stdout=log, stderr=subprocess.STDOUT)
    records.append(dict(name=name, command=command, returncode=r.returncode, seconds=time.monotonic()-start))
    (OUT/'runs.json').write_text(json.dumps(records, indent=2))
    print('END', name, r.returncode, flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)
