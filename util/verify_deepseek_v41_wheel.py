"""Verify wheel payload against the checkout and the tested extension; save provenance."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def sha(data): return hashlib.sha256(data).hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('wheel',type=Path)
    p.add_argument('--installed',action='store_true')
    a=p.parse_args();wheel=a.wheel.resolve()
    assert git('branch','--show-current')=='feature/native-ir-deepseek-v41'
    assert not git('diff','HEAD','--name-only','--','exllamav3'), 'uncommitted product sources'
    binary=ROOT/'native-results/deepseek-v41/package/build/lib.win-amd64-cpython-313/exllamav3_ext.cp313-win_amd64.pyd'
    hashes={}
    with zipfile.ZipFile(wheel) as z:
        assert z.testzip() is None
        assert z.read(binary.name)==binary.read_bytes(),'extension differs from tested artifact'
        for name in git('ls-files','exllamav3').splitlines():
            path=ROOT/name
            if path.suffix not in ('.py','.cpp','.cu','.h','.cuh'):continue
            assert z.read(name)==path.read_bytes(),name
            hashes[name]=sha(path.read_bytes())
        metadata=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
        assert 'Tag: cp313-cp313-win_amd64' in metadata
        assert 'Root-Is-Purelib: false' in metadata
        package_metadata=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        version=next(line.removeprefix('Version: ') for line in package_metadata.splitlines() if line.startswith('Version: '))
        if a.installed:
            site=Path('D:/tabbyAPI/venv/Lib/site-packages')
            assert importlib.metadata.version('exllamav3')==version
            for name in [*hashes,binary.name]:
                assert (site/name).read_bytes()==z.read(name),('installed payload mismatch',name)
    proof=dict(wheel=str(wheel),version=version,wheel_sha256=sha(wheel.read_bytes()),
        extension_sha256=sha(binary.read_bytes()),source_commit=git('rev-parse','HEAD'),
        branch=git('branch','--show-current'),architectures=['sm_89','sm_120'],
        build_log='native-results/deepseek-v41/package/build.log',
        product_sources_committed=True,installed_payload_verified=a.installed,payload_hashes=hashes)
    (wheel.parent.parent/('installed-proof.json' if a.installed else 'proof.json')).write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k!='payload_hashes'},indent=2))

if __name__=='__main__':main()
