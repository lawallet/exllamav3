"""Gate and archive the 1.5.4 integration using the established release workflow."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'guards', 'shared', 'mgemm-0', 'mgemm-1', 'glm-sites', 'deepseek-sites',
            'gemma-sites', 'qwen-short', 'deepseek-short', 'gemma-short', 'glm-mtp',
            'qwen-mtp', 'qwen-concurrency', 'deepseek-mtp', 'gemma-reuse',
            'devstral-short', 'laguna-short'}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('phase', choices=['gates', 'finalize'])
p.add_argument('--validation', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--package', type=Path)
p.add_argument('--wheel-validation', type=Path)
p.add_argument('--tag', default='native-plan-v1.5.4-integration-v1')
a = p.parse_args()
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
commit = git('rev-parse', 'HEAD')
assert not git('diff', 'HEAD', '--', 'exllamav3', 'doc', 'tests', 'util')
assert git('branch', '--show-current') == 'native-plan-v1.5.4-integration'
original = 'd70a1de81f7fb72d45d95ffd87d76dd0c60cf2c0'
for ref in ('native-laguna-ir-v2', 'backup-native-laguna-pre-v1.5.4-2026-10-03'):
    assert git('rev-parse', ref) == original
subprocess.run(['git', 'merge-base', '--is-ancestor', 'v1.5.4', 'HEAD'], cwd=ROOT, check=True)
passed = set()
proof = None
for directory in a.validation:
    candidate = read(directory/'inputs.json')
    if proof is None:
        proof = candidate
    assert candidate == proof, 'Validation inputs differ between campaigns'
    for record in read(directory/'runs.json'):
        if record['returncode'] == 0:
            passed.add(record['name'])
assert REQUIRED <= passed, 'Missing required gates: '+str(REQUIRED-passed)
for name, expected in proof['source_hashes'].items():
    assert sha(ROOT/name) == expected, 'Source changed: '+name
assert sha(Path(proof['extension'])) == proof['extension_sha256']
if a.phase == 'gates':
    assert not a.output.exists()
    a.output.write_text(json.dumps(dict(all_required_checkout_gates_passed=True,
        commit=commit, branch=git('branch', '--show-current'), gates=sorted(passed),
        validation=[str(v.resolve()) for v in a.validation], **proof), indent=2))
else:
    assert a.package and a.wheel_validation
    package = read(a.package/'package-proof.json')
    assert package['commit'] == commit == git('rev-parse', a.tag+'^{commit}')
    assert package['extension_sha256'] == proof['extension_sha256']
    wheel_runs = read(a.wheel_validation/'runs.json')
    assert {r['name'] for r in wheel_runs} == {'qwen-short', 'gemma-short'}
    assert all(r['returncode'] == 0 for r in wheel_runs)
    for r in wheel_runs:
        assert 'PASS all product imports came from extracted wheel' in (
            a.wheel_validation/(r['name']+'.log')).read_text()
    wheel = Path(package['wheel'])
    assert sha(wheel) == package['wheel_sha256']
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(wheel, out/wheel.name)
    shutil.copy2(ROOT/'doc/native_plan_v154_integration.md', out/'README.md')
    shutil.copy2(a.package/'checkout-gates.json', out/'checkout-gates.json')
    shutil.copy2(a.wheel_validation/'runs.json', out/'extracted-wheel-runs.json')
    package['status'] = 'validated'
    (out/'package-proof.json').write_text(json.dumps(package, indent=2))
    metadata = dict(commit=commit, tag=a.tag, original_commit=original,
        branch=git('branch', '--show-current'), upstream=git('rev-parse', 'v1.5.4^{commit}'),
        wheel=wheel.name, wheel_sha256=sha(wheel), extension_sha256=proof['extension_sha256'],
        version='1.5.4+nativeplan.v2.integration1', installed=False,
        validation=[str(v.resolve()) for v in a.validation], wheel_validation=str(a.wheel_validation.resolve()))
    (out/'checkpoint.json').write_text(json.dumps(metadata, indent=2))
    subprocess.run(['git', 'archive', '--format=zip', '--output='+str(out/'source.zip'), commit], cwd=ROOT, check=True)
    subprocess.run(['git', 'bundle', 'create', str(out/'repository.bundle'), '--all'], cwd=ROOT, check=True)
    subprocess.run(['git', 'bundle', 'verify', str(out/'repository.bundle')], cwd=ROOT, check=True)
    (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in sorted(out.iterdir()) if f.is_file()))
    print(json.dumps(metadata, indent=2))
