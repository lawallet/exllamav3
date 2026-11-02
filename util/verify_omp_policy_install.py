"""Verify installed wheel payload and unchanged configuration/dependencies/rollback wheel."""
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    before=json.loads((BASE/'environment-before.json').read_text())
    proof=json.loads((BASE/'package/proof.json').read_text())
    site=Path('D:/tabbyAPI/venv/Lib/site-packages')
    assert md.version('exllamav3')==proof['version']
    for name,value in proof['payload_hashes'].items():assert sha(site/name)==value,name
    assert sha(site/'exllamav3_ext.cp313-win_amd64.pyd')==proof['extension_sha256']==before['extension_sha256']
    assert sha(proof['wheel'])==proof['wheel_sha256']
    assert sha(before['rollback_wheel'])==before['wheel_sha256']
    packages={d.metadata['Name']:d.version for d in md.distributions()}
    assert {k:v for k,v in packages.items() if k.lower()!='exllamav3'}=={k:v for k,v in before['packages'].items() if k.lower()!='exllamav3'}
    assert all(sha(Path('D:/tabbyAPI')/n)==v for n,v in before['config_hashes'].items())
    assert sha(ROOT/'convert.py')==before['convert_sha256']
    result=dict(version=proof['version'],payload_files=len(proof['payload_hashes']),payload_exact=True,
        wheel_sha256=proof['wheel_sha256'],extension_sha256=proof['extension_sha256'],rollback_preserved=True,
        other_package_versions_unchanged=True,persistent_configuration_unchanged=True,convert_edit_preserved=True)
    (BASE/'package/installed-proof.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
