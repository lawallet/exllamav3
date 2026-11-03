"""Read-only provenance snapshot for the pipeline campaign and rollback wheel."""
import hashlib,importlib.metadata as md,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/generation-pipeline'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    path=BASE/'environment-before.json';assert not path.exists()
    proof=json.loads((ROOT/'native-results/deepseek-v41/omp-policy/package/proof.json').read_text())
    previous=json.loads((ROOT/'native-results/deepseek-v41/omp-policy/environment-before.json').read_text())
    site=Path('D:/tabbyAPI/venv/Lib/site-packages')
    assert md.version('exllamav3')==proof['version']
    assert sha(proof['wheel'])==proof['wheel_sha256']
    for name,value in proof['payload_hashes'].items():assert sha(site/name)==value,name
    extension=site/'exllamav3_ext.cp313-win_amd64.pyd'
    assert sha(extension)==proof['extension_sha256']
    result=dict(installed_version=md.version('exllamav3'),rollback_wheel=proof['wheel'],wheel_sha256=proof['wheel_sha256'],
        extension=str(extension),extension_sha256=sha(extension),payload_files=len(proof['payload_hashes']),payload_exact=True,
        config_hashes={n:sha(Path('D:/tabbyAPI')/n) for n in previous['config_hashes']},
        convert_sha256=sha(ROOT/'convert.py'),packages={d.metadata['Name']:d.version for d in md.distributions()},
        baseline_commit='68483a4b',branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip())
    path.write_text(json.dumps(result,indent=2));print(path)
if __name__=='__main__':main()
