"""Package the Python-only WDDM fix with the unchanged, previously tested native extension."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'native-results/wddm-autosplit/package'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    previous = json.loads((ROOT/'native-results/deepseek-v41/attention-optimization/package/proof.json').read_text())
    old = Path(previous['wheel'])
    assert sha(old) == previous['wheel_sha256']
    OUT.mkdir(parents=True, exist_ok=False)
    stage = OUT/'stage'
    stage.mkdir()
    with zipfile.ZipFile(old) as z:
        binary, = [n for n in z.namelist() if n.endswith('.pyd')]
        for p in (ROOT/'exllamav3/exllamav3_ext').rglob('*'):
            if p.is_file() and p.suffix in ('.cu','.cpp','.h','.cuh'):
                assert z.read(p.relative_to(ROOT).as_posix()) == p.read_bytes(), p
        (stage/binary).write_bytes(z.read(binary))
    installed = subprocess.check_output([sys.executable, '-I', '-c',
        'import importlib.metadata as m,json; d=m.distribution("exllamav3"); '
        'print(json.dumps(dict(version=d.version,location=str(d.locate_file("")))))'], text=True)
    (OUT/'rollback.json').write_text(json.dumps(dict(installed=json.loads(installed),
        wheel=str(old),sha256=sha(old)),indent=2))
    ini=OUT/'build.ini'
    ini.write_text('[build]\nbuild_lib = '+stage.as_posix()+'\n')
    env=dict(os.environ,DIST_EXTRA_CONFIG=str(ini),CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
             TORCH_CUDA_ARCH_LIST='8.9;12.0')
    for key in ('PYTHONPATH','EXLLAMA_NOCOMPILE'):
        env.pop(key,None)
    with (OUT/'build.log').open('w') as log:
        subprocess.run([sys.executable,'setup.py','build_py','bdist_wheel','--skip-build',
                        '--bdist-dir',str(OUT/'bdist'),'--dist-dir',str(OUT/'wheels')],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    wheel, = (OUT/'wheels').glob('*.whl')
    hashes={}
    with zipfile.ZipFile(wheel) as z:
        assert z.testzip() is None
        assert z.read(binary)==(stage/binary).read_bytes()
        for p in (ROOT/'exllamav3').rglob('*'):
            if p.is_file() and p.suffix in ('.py','.cu','.cpp','.h','.cuh'):
                name=p.relative_to(ROOT).as_posix()
                assert z.read(name)==p.read_bytes(), name
                hashes[name]=sha(p)
    proof=dict(wheel=str(wheel),wheel_sha256=sha(wheel),extension_sha256=sha(stage/binary),
               base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               native_binary_reused=True,payload_hashes=hashes)
    (OUT/'proof.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k!='payload_hashes'},indent=2))


if __name__ == '__main__':
    main()
