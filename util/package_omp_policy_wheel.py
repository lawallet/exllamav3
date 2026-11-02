"""Package committed Python policy with the unchanged validated SM89/SM120 binary."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def main():
    assert git('branch','--show-current')=='feature/native-ir-deepseek-v41'
    assert not git('diff','HEAD','--name-only','--','exllamav3'),'commit product source before packaging'
    old=json.loads((ROOT/'native-results/deepseek-v41/package/installed-proof.json').read_text())
    assert sha(old['wheel'])==old['wheel_sha256']
    binary=Path('D:/tabbyAPI/venv/Lib/site-packages/exllamav3_ext.cp313-win_amd64.pyd')
    assert sha(binary)==old['extension_sha256']
    for name,value in old['payload_hashes'].items():
        if name.startswith('exllamav3/exllamav3_ext/'):
            assert sha(ROOT/name)==value,('native source changed',name)
    out=BASE/'package';out.mkdir(parents=True,exist_ok=False)
    stage=out/'stage';stage.mkdir();shutil.copy2(binary,stage/binary.name)
    ini=out/'build.ini';ini.write_text('[build]\nbuild_lib = '+stage.as_posix()+'\n')
    env=dict(os.environ,DIST_EXTRA_CONFIG=str(ini),SOURCE_DATE_EPOCH=git('show','-s','--format=%ct','HEAD'),
             CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',TORCH_CUDA_ARCH_LIST='8.9;12.0')
    for key in ('PYTHONPATH','EXLLAMA_NOCOMPILE','EXL3_OMP_BLOCKTIME_MS'):env.pop(key,None)
    with (out/'build.log').open('w') as log:
        subprocess.run([sys.executable,'setup.py','build_py','bdist_wheel','--skip-build',
            '--bdist-dir',str(out/'bdist'),'--dist-dir',str(out/'wheels')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    wheel,=(out/'wheels').glob('*.whl');hashes={}
    with zipfile.ZipFile(wheel) as z:
        assert z.testzip() is None
        assert z.read(binary.name)==binary.read_bytes()
        for name in git('ls-files','exllamav3').splitlines():
            path=ROOT/name
            if path.suffix not in ('.py','.cu','.cpp','.h','.cuh'):continue
            assert z.read(name)==path.read_bytes(),name
            hashes[name]=sha(path)
        wheel_meta=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
        assert 'Tag: cp313-cp313-win_amd64' in wheel_meta and 'Root-Is-Purelib: false' in wheel_meta
        metadata=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        version=next(s[9:] for s in metadata.splitlines() if s.startswith('Version: '))
        assert version=='1.6.6+nativeplan.v2.tp.moe1.ds41.1.omp1'
        z.extractall(out/'extracted')
    proof=dict(wheel=str(wheel),version=version,wheel_sha256=sha(wheel),extension_sha256=sha(binary),
        source_commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),compiled_during_packaging=False,
        architectures=['sm_89','sm_120'],binary_provenance='native-results/deepseek-v41/package/installed-proof.json',
        rollback_wheel=old['wheel'],rollback_sha256=old['wheel_sha256'],payload_hashes=hashes)
    (out/'proof.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k!='payload_hashes'},indent=2))


if __name__=='__main__':main()
