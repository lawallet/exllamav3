"""Package validated DeepSeek sources and the exact measured extension, without installing."""
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
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    v = a.validation.resolve()
    gates = ('resume-short', 'resume-generator-v2', 'resume-mtp', 'resume-mtp-cancel',
             'resume-topology-short', 'resume-topology-mtp', 'resume-remapped',
             'resume-long', 'resume-long-mtp', 'resume-benchmark', 'resume-benchmark-mtp')
    for gate in gates:
        assert json.loads((v/gate/'passed.json').read_text()), gate
    guard_bytes = (v/'guards-resumed.log').read_bytes()
    guard_log = guard_bytes.decode('utf-16' if guard_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
    assert 'Ran 113 tests' in guard_log
    assert 'OK' in guard_log
    extension, = (v/'extension').glob('exllamav3_ext*.pyd')
    expected = '40ca8e4f2910e72655dc42422af94d72dd629f314746c2d565840c9787c72473'
    assert sha(extension) == expected, 'Extension differs from the validated build'
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = out/'stage'
    stage.mkdir()
    shutil.copy2(extension, stage/extension.name)
    config = out/'build.ini'
    config.write_text('[build]\nbuild_lib = '+stage.as_posix()+'\n')
    env = dict(os.environ, DIST_EXTRA_CONFIG=str(config),
               CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8')
    env.pop('EXLLAMA_NOCOMPILE', None)
    with (out/'build.log').open('w') as log:
        subprocess.run([sys.executable, 'setup.py', 'build_py', 'bdist_wheel', '--skip-build',
                        '--bdist-dir', str(out/'bdist'), '--dist-dir', str(out/'wheels')],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    wheel, = (out/'wheels').glob('*.whl')
    sources = {}
    with zipfile.ZipFile(wheel) as z:
        assert hashlib.sha256(z.read(extension.name)).hexdigest() == expected
        for source in (ROOT/'exllamav3').rglob('*.py'):
            name = source.relative_to(ROOT).as_posix()
            sources[name] = sha(source)
            assert hashlib.sha256(z.read(name)).hexdigest() == sources[name], name
        metadata = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
        assert 'Root-Is-Purelib: false' in metadata
        assert 'Tag: cp313-cp313-win_amd64' in metadata
        z.extractall(out/'extracted')
    proof = dict(wheel=str(wheel), wheel_sha256=sha(wheel), extension_sha256=expected,
                 python_sources=sources, gates=gates, compiled_during_packaging=False,
                 installed=False)
    (out/'package-proof.json').write_text(json.dumps(proof, indent=2))
    (out/'SHA256SUMS.txt').write_text(f'{sha(wheel)}  {wheel.name}\n')
    print(json.dumps({k:v for k,v in proof.items() if k != 'python_sources'}, indent=2))


if __name__ == '__main__':
    main()
