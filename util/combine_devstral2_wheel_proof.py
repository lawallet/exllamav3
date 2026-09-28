"""Carry passed gates only across byte-identical wheel payloads, never code changes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
p=argparse.ArgumentParser()
for arg in ('previous-package','package','previous-validation','resumed-validation','output'):
    p.add_argument('--'+arg,type=Path,required=True)
a=p.parse_args()
def read(path): return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
old=read(a.previous_package/'package-proof.json'); new=read(a.package/'package-proof.json')
assert old['python_sources']==new['python_sources']
assert old['extension_sha256']==new['extension_sha256']
for proof in (old,new): assert sha(Path(proof['wheel']))==proof['wheel_sha256']
with zipfile.ZipFile(old['wheel']) as left, zipfile.ZipFile(new['wheel']) as right:
    names=set(left.namelist());assert names==set(right.namelist())
    for name in names:
        with left.open(name) as x,right.open(name) as y:
            assert hashlib.file_digest(x,'sha256').digest()==hashlib.file_digest(y,'sha256').digest(),name
assert read(a.resumed_validation/'passed.json')['selected_package_gates_passed']
passed={r['name']:dict(r,carried_from=str(a.previous_validation.resolve()))
        for r in read(a.previous_validation/'runs.json') if r['returncode']==0}
for r in read(a.resumed_validation/'runs.json'):
    assert r['returncode']==0
    passed[r['name']]=r
required=['devstral-sites','devstral-short','devstral-topology','gemma-sites','gemma-short','deepseek-short','qwen-short','glm-short']
assert set(passed)==set(required)
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'runs.json').write_text(json.dumps([passed[n] for n in required],indent=2))
(a.output/'payload-equivalence.json').write_text(json.dumps(dict(
    previous_wheel_sha256=old['wheel_sha256'],final_wheel_sha256=new['wheel_sha256'],
    all_member_bytes_equal=True,member_count=len(names),native_extension_sha256=new['extension_sha256'],
    previous_commit=old['commit'],final_commit=new['commit'],
    correction='Validation-only Qwen/GLM cache capacity; no product/binary changes',
    preserved_failure=str((a.previous_validation/'qwen-short.log').resolve())),indent=2))
(a.output/'passed.json').write_text(json.dumps(dict(all_package_gates_passed=True)))
print('PASS all eight package gates, with proven byte-identical payload equivalence')
