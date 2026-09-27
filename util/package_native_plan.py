"""Package the measured binary unchanged; refuse incomplete validation gates.

Uses setuptools' per-process DIST_EXTRA_CONFIG to select an isolated build tree.
Only build_py runs: bdist_wheel --skip-build retains the tested extension bytes.
No install, dependency resolution, compiler invocation or protected wheel edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--validation', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--extension-dir', type=Path, required=True)
a = p.parse_args()
v = a.validation.resolve()
def read(path): return json.loads((v/path).read_text())
for arch in ('qwen', 'glm'):
    assert read(f'short-final/{arch}-20/parity.json')['exact_bespoke_plan']
    assert read(f'remaining/reuse/{arch}-reuse/ordinary-page-reuse/parity.json')['exact']
for phase in ('mtp', 'concurrency'):
    assert read(f'remaining/{phase}/bespoke-plan-parity.json')['exact']
topology = read('remaining/topology/topology-comparison.json')
assert set(topology) == {'qwen', 'glm'}
for result in topology.values():
    assert all(result[k] for k in ('exact_state', 'exact_completions', 'same_call_counts',
        'same_eligibility', 'same_logical_transfers', 'same_bespoke_plan_boundary_counts', 'zero_decode_fallbacks'))
    assert result['unused_device_peak_allocated'] == 0
for case in ('qwen-qsa-64k', 'qwen-qsa-32k-mtp', 'glm-64k-ordinary', 'glm-64k-mtp'):
    assert read(f'remaining/long/{case}/parity.json')['exact']
primitive = read('primitive-parity.json')
assert len(primitive) == 7 and all(r['exact'] and r['invalid_boundary_transactional'] and r['batch_query_distinct'] for r in primitive)
performance = read('performance-summary.json')
assert performance
for result in performance:
    if 'topology' in result['source']: continue
    assert min(map(len, result['samples'].values())) >= 3 and result['relative_change'] >= -.03, result
assert 'nativeplan' in (ROOT/'exllamav3/version.py').read_text(), 'Assign the new version only after gates pass'

out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
stage = out/'stage'
stage.mkdir()
extensions = list(a.extension_dir.glob('exllamav3_ext*.pyd'))
assert len(extensions) == 1
extension = extensions[0]
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
tested_sha = sha(extension)
shutil.copy2(extension, stage/extension.name)
config = out/'build.ini'
config.write_text('[build]\nbuild_lib = '+stage.as_posix()+'\n')
env = dict(os.environ, DIST_EXTRA_CONFIG=str(config), CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8')
env.pop('EXLLAMA_NOCOMPILE', None)
with (out/'build.log').open('w') as log:
    subprocess.run([sys.executable, 'setup.py', 'build_py', 'bdist_wheel', '--skip-build',
        '--bdist-dir', str(out/'bdist'), '--dist-dir', str(out/'wheels')], cwd=ROOT,
        env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
wheels = list((out/'wheels').glob('*.whl'))
assert len(wheels) == 1
wheel = wheels[0]
sources = {}
with zipfile.ZipFile(wheel) as z:
    with z.open(extension.name) as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == tested_sha, 'Packaging changed the measured binary'
    for source in (ROOT/'exllamav3').rglob('*.py'):
        name = source.relative_to(ROOT).as_posix()
        sources[name] = sha(source)
        assert hashlib.sha256(z.read(name)).hexdigest() == sources[name], name
    metadata = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
    assert 'Root-Is-Purelib: false' in metadata and 'Tag: cp313-cp313-win_amd64' in metadata, metadata
proof = dict(wheel=wheel.name, wheel_sha256=sha(wheel), extension_sha256=tested_sha,
             python_sources=sources, dependencies_unchanged=True, compiled_during_packaging=False)
(out/'package-proof.json').write_text(json.dumps(proof, indent=2))
print(json.dumps({k: value for k, value in proof.items() if k != 'python_sources'}, indent=2))
