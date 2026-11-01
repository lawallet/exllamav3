"""Verify installed package/extension bytes against a wheel, using Python -I."""
import argparse
import email
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert sys.flags.isolated, 'Invoke with python -I to exclude checkout imports'
    import torch
    import exllamav3
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.version import __version__
    installed = Path(exllamav3.__file__).resolve().parent.parent
    assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
    assert Path(ext.__file__).parent == installed, ext.__file__
    hashes = {}
    with zipfile.ZipFile(a.wheel) as wheel:
        assert wheel.testzip() is None
        metadata, = [n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        version = email.message_from_bytes(wheel.read(metadata))['Version']
        assert version == __version__ == importlib.metadata.version('exllamav3')
        for name in wheel.namelist():
            if name.endswith('/') or not (name.startswith('exllamav3/') or name.endswith('.pyd')):
                continue
            payload = wheel.read(name)
            assert (installed/name).read_bytes() == payload, f'Installed payload differs: {name}'
            hashes[name] = hashlib.sha256(payload).hexdigest()
    for name in ('BC_BlockPlan','BC_BlockPlanV2','BC_GlmBlock','BC_QwenBlock','native_moe_assign'):
        assert hasattr(ext, name), name
    assert ext.native_moe_assignment_abi == 1
    report = dict(version=version, wheel=str(a.wheel.resolve()),
                  wheel_sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest(),
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  package=exllamav3.__file__, extension=ext.__file__, payload_hashes=hashes)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'payload_hashes'}, indent=2))


if __name__ == '__main__':
    main()
