"""Run representative parity checks with imports pinned to the extracted wheel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
assert Path(sys.executable).resolve() == Path(r'D:\tabbyAPI\venv\Scripts\python.exe').resolve(), 'Use the existing TabbyAPI interpreter'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--package', type=Path, required=True, help='Packaging output directory')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
package = a.package.resolve()
extracted = package/'extracted'
proof = json.loads((package/'package-proof.json').read_text())
binary, = extracted.glob('exllamav3_ext*.pyd')
with binary.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == proof['extension_sha256']
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
jobs = [
    ('qwen-short', 'profile_qwen_decode.py', ['--model', r'D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw',
        '--prompt-file', str(ROOT/'doc/native_qwen_prompt.txt'), '--device', 'cuda:0',
        '--ab', '--compare-plan', '--tokens', '128', '--runs', '1']),
    ('gemma-short', 'validate_gemma4_v2.py', ['--steps', '8']),
]
runs = []
for name, script, args in jobs:
    command = [sys.executable, str(ROOT/'util/verify_v154_wheel.py'), '--package', str(extracted),
               '--script', script, '--', '--output', str(out/name), *args]
    print('START', name, flush=True)
    start = time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='2,0,1', EXL3_NATIVE_PLAN='1'))
    runs.append(dict(name=name, command=command, returncode=result.returncode,
                     seconds=time.monotonic()-start))
    (out/'runs.json').write_text(json.dumps(runs, indent=2))
    print('END', name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
