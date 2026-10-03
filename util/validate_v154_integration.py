"""Sequential integration gates; never install into the pinned runtime."""
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
p.add_argument('--extension-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--only', nargs='+')
a = p.parse_args()
ext = a.extension_dir.resolve()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
binary, = ext.glob('exllamav3_ext*.pyd')
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def sources():
    return {f.relative_to(ROOT).as_posix(): sha(f) for f in (ROOT/'exllamav3').rglob('*')
            if f.is_file() and f.suffix in ('.py', '.cpp', '.h', '.cu', '.cuh')}
before = sources()
proof = dict(extension=str(binary), extension_sha256=sha(binary), source_hashes=before)
(out/'inputs.json').write_text(json.dumps(proof, indent=2))
qwen = r'D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw'
glm = r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw'
jobs = []
def unit(name, patterns):
    code = ("import sys, unittest; sys.path.insert(0, %r); "
            "suite=unittest.TestSuite(unittest.defaultTestLoader.discover('tests',pattern=p) for p in %r); "
            "r=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not r.wasSuccessful())") % (str(ext), patterns)
    jobs.append((name, ['-c', code], '2,0,1'))
def script(name, path, args=(), mapping='2,0,1', json_output=False):
    jobs.append((name, [path, '--extension-dir', str(ext), '--output',
                       str(out/(name+('.json' if json_output else ''))), *args], mapping))
unit('guards', ['test_native*.py', 'test_decode_dispatch_profile.py'])
unit('shared', ['test_recurrent_host_pool.py', 'test_generator_checkpoint_intervals.py',
                'test_job_params.py', 'test_gr_mix_tiled.py'])
for device in ('cuda:0', 'cuda:1'):
    script('mgemm-'+device[-1], 'tests/test_mgemm_variable_width.py',
           ['--device', device], json_output=True)
script('glm-sites', 'util/check_native_plan_primitives.py', json_output=True)
script('deepseek-sites', 'util/check_deepseek_plan_sites.py', json_output=True)
script('gemma-sites', 'util/check_gemma4_v2_sites.py')
script('qwen-short', 'util/profile_qwen_decode.py', ['--model', qwen, '--prompt-file',
       'doc/native_qwen_prompt.txt', '--device', 'cuda:0', '--ab', '--compare-plan', '--tokens', '128', '--runs', '1'])
script('deepseek-short', 'util/validate_deepseek_plan.py', ['--steps', '16', '--reload'])
script('gemma-short', 'util/validate_gemma4_v2.py', ['--steps', '8', '--reload'])
script('glm-mtp', 'util/profile_glm_mtp.py', ['--model', glm, '--parity-only',
       '--parity-tokens', '64', '--forced-tokens', '64'])
script('qwen-mtp', 'util/profile_qwen_mtp.py', ['--model', qwen, '--parity-only', '--parity-tokens', '64'])
script('qwen-concurrency', 'util/profile_qwen_concurrency.py', ['--suite', 'ordinary', '--compare-plan'])
script('deepseek-mtp', 'util/validate_deepseek_plan.py', ['--phase', 'generator', '--mtp',
       '--case', '0', '--case', '8', '--steps', '32'])
script('gemma-reuse', 'util/validate_gemma4_v2.py', ['--phase', 'generator', '--case', '5',
       '--context', '240', '--steps', '48', '--cache-tokens', '1536'])
script('devstral-short', 'util/validate_devstral2.py', ['--steps', '8'])
script('laguna-short', 'util/validate_laguna.py', ['--steps', '8'])
if a.only:
    assert set(a.only) <= {name for name, _, _ in jobs}, 'Unknown gate'
runs = []
for name, command, mapping in jobs:
    if a.only and name not in a.only:
        continue
    print('START', name, flush=True)
    started = time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result = subprocess.run([sys.executable, *command], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=mapping, EXL3_NATIVE_PLAN='0' if name=='guards' else '1'),
            stdout=log, stderr=subprocess.STDOUT)
    runs.append(dict(name=name, command=command, mapping=mapping,
                     returncode=result.returncode, seconds=time.monotonic()-started))
    (out/'runs.json').write_text(json.dumps(runs, indent=2))
    print('END', name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
assert sources() == before, 'Product source changed during validation'
assert sha(binary) == proof['extension_sha256'], 'Extension changed during validation'
(out/'passed.json').write_text(json.dumps(dict(all_selected_gates_passed=True,
    gates=[r['name'] for r in runs], **proof), indent=2))
