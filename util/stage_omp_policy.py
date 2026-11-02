"""Stage candidate Python plus the preserved validated extension, without installing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=json.loads((ROOT/'native-results/deepseek-v41/package/installed-proof.json').read_text())
    extension=Path('D:/tabbyAPI/venv/Lib/site-packages/exllamav3_ext.cp313-win_amd64.pyd')
    assert hashlib.sha256(extension.read_bytes()).hexdigest()==old['extension_sha256']
    shutil.copytree(ROOT/'exllamav3',out/'exllamav3',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(extension,out/extension.name)
    hashes={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (out/'exllamav3').rglob('*') if p.is_file()}
    (out/'source-manifest.json').write_text(json.dumps(dict(extension_sha256=old['extension_sha256'],sources=hashes),indent=2))
    print(out)


if __name__=='__main__':main()
