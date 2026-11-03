"""Snapshot or verify only the authorized TabbyAPI installation and rollback wheel."""
import argparse,hashlib,importlib.metadata as md,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/attention-optimization'
SITE=Path('D:/tabbyAPI/venv/Lib/site-packages')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
    packages={d.metadata['Name']:d.version for d in md.distributions()}
    if a.snapshot:
        old=json.loads((ROOT/'native-results/deepseek-v41/generation-pipeline/package/proof.json').read_text())
        original=json.loads((ROOT/'native-results/deepseek-v41/generation-pipeline/environment-before.json').read_text())
        assert md.version('exllamav3')==old['version']
        for name,value in old['payload_hashes'].items():assert sha(SITE/name)==value,name
        assert sha(SITE/'exllamav3_ext.cp313-win_amd64.pyd')==old['extension_sha256']
        assert sha(old['wheel'])==old['wheel_sha256']
        result=dict(packages=packages,rollback_wheel=old['wheel'],rollback_sha256=old['wheel_sha256'],
            extension_sha256=old['extension_sha256'],convert_sha256=sha(ROOT/'convert.py'),
            config_hashes={n:sha(Path('D:/tabbyAPI')/n) for n in original['config_hashes']})
        target=a.output or BASE/'environment-before.json'
    else:
        before=json.loads((BASE/'environment-before.json').read_text());proof=json.loads((BASE/'package/proof.json').read_text())
        assert md.version('exllamav3')==proof['version']
        for name,value in proof['payload_hashes'].items():assert sha(SITE/name)==value,name
        assert sha(SITE/'exllamav3_ext.cp313-win_amd64.pyd')==proof['extension_sha256']
        assert sha(proof['wheel'])==proof['wheel_sha256']
        assert sha(before['rollback_wheel'])==before['rollback_sha256']
        assert {k:v for k,v in packages.items() if k.lower()!='exllamav3'}=={k:v for k,v in before['packages'].items() if k.lower()!='exllamav3'}
        assert all(sha(Path('D:/tabbyAPI')/n)==v for n,v in before['config_hashes'].items())
        assert sha(ROOT/'convert.py')==before['convert_sha256']
        result=dict(version=proof['version'],payload_files=len(proof['payload_hashes']),payload_exact=True,
            wheel_sha256=proof['wheel_sha256'],extension_sha256=proof['extension_sha256'],rollback_preserved=True,
            other_package_versions_unchanged=True,persistent_configuration_unchanged=True,convert_edit_preserved=True)
        target=a.output or BASE/'package/installed-proof.json'
    assert not target.exists();target.write_text(json.dumps(result,indent=2));print(str(target))
if __name__=='__main__':main()
