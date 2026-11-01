"""Verify and extract the isolated wheel without modifying an installed runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-version', default='1.6.5+nativeplan.v2.tp1')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    with zipfile.ZipFile(a.wheel) as z:
        assert z.testzip() is None
        for f in (ROOT/'exllamav3').rglob('*'):
            if not f.is_file() or f.suffix not in ('.py', '.cpp', '.cu', '.cuh', '.h'):
                continue
            name = f.relative_to(ROOT).as_posix()
            payload = z.read(name)
            assert payload == f.read_bytes(), f'Stale wheel source: {name}'
            hashes[name] = hashlib.sha256(payload).hexdigest()
        binaries = [n for n in z.namelist() if n.endswith('.pyd')]
        assert len(binaries) == 1 and binaries[0] == 'exllamav3_ext.cp313-win_amd64.pyd', binaries
        hashes[binaries[0]] = hashlib.sha256(z.read(binaries[0])).hexdigest()
        z.extractall(out)
    code = """
import json, sys, torch
from pathlib import Path
import exllamav3
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.version import __version__
root = Path.cwd()
assert Path(exllamav3.__file__).is_relative_to(root), exllamav3.__file__
assert Path(ext.__file__).is_relative_to(root), ext.__file__
assert __version__ == sys.argv[1], __version__
for name in ('BC_BlockPlan','BC_BlockPlanV2','BC_GlmBlock','BC_QwenBlock'):
    assert hasattr(ext, name), name
print(json.dumps(dict(version=__version__, python=sys.version, torch=torch.__version__,
                     cuda=torch.version.cuda, package=exllamav3.__file__, extension=ext.__file__)))
"""
    result = subprocess.run([sys.executable, '-c', code, a.expected_version], cwd=out, text=True, capture_output=True)
    (out.parent/(out.name+'-import.log')).write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    proof = dict(wheel=str(a.wheel.resolve()), sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest(),
                 bytes=a.wheel.stat().st_size, payload_hashes=hashes, runtime=json.loads(result.stdout))
    (out.parent/(out.name+'-proof.json')).write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k != 'payload_hashes'},indent=2))


if __name__ == '__main__':
    main()
