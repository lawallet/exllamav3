"""Stage a candidate wheel from a clean checkpoint and the exact validated binary.

Extracted-wheel gates must pass before the candidate is described as releasable.
Never installs into a production environment.
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
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--gates', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gates = json.loads(a.gates.read_text())
    assert gates['all_required_checkout_gates_passed'] is True
    assert not git('diff', 'HEAD', '--', 'exllamav3', 'doc', 'tests', 'util')
    commit = git('rev-parse', 'HEAD')
    assert gates['commit'] == commit
    for name, expected in gates['source_hashes'].items():
        assert sha(ROOT/name) == expected, 'Source changed after validation: '+name
    extension, = a.extension_dir.resolve().glob('exllamav3_ext*.pyd')
    assert sha(extension) == gates['extension_sha256']
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = out/'stage'; stage.mkdir()
    shutil.copy2(extension, stage/extension.name)
    config = out/'build.ini'
    config.write_text('[build]\nbuild_lib = '+stage.as_posix()+'\n')
    epoch = git('show', '-s', '--format=%ct', commit)
    env = dict(os.environ, DIST_EXTRA_CONFIG=str(config), SOURCE_DATE_EPOCH=epoch,
               CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8')
    env.pop('EXLLAMA_NOCOMPILE', None)
    with (out/'build.log').open('w') as log:
        subprocess.run([sys.executable, 'setup.py', 'build_py', 'bdist_wheel', '--skip-build',
                        '--bdist-dir', str(out/'bdist'), '--dist-dir', str(out/'wheels')],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    wheel, = (out/'wheels').glob('*.whl')
    sources = {}
    with zipfile.ZipFile(wheel) as z:
        assert hashlib.sha256(z.read(extension.name)).hexdigest() == sha(extension)
        for source in (ROOT/'exllamav3').rglob('*.py'):
            name = source.relative_to(ROOT).as_posix()
            sources[name] = sha(source)
            assert sources[name] == gates['source_hashes'][name], 'Source changed after validation: '+name
            assert hashlib.sha256(z.read(name)).hexdigest() == sources[name], name
            committed = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
            assert source.read_bytes().replace(b'\r\n', b'\n') == committed.replace(b'\r\n', b'\n'), name
        metadata = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
        assert 'Tag: cp313-cp313-win_amd64' in metadata and 'Root-Is-Purelib: false' in metadata
        package_metadata = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        assert 'Version: 1.5.1+nativeplan.v2.laguna1' in package_metadata
        z.extractall(out/'extracted')
    proof = dict(commit=commit, branch=git('branch','--show-current'), wheel=str(wheel),
        wheel_sha256=sha(wheel), extension_sha256=sha(extension), python_sources=sources,
        compiled_during_packaging=False, source_date_epoch=epoch,
        installed=False, status='candidate_pending_extracted_validation')
    (out/'package-proof.json').write_text(json.dumps(proof, indent=2))
    shutil.copy2(a.gates, out/'checkout-gates.json')
    print(json.dumps({k:v for k,v in proof.items() if k!='python_sources'}, indent=2))
if __name__ == '__main__': main()
