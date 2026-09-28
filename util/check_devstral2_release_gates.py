"""Verify retained evidence and source identity before candidate packaging."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--validation',type=Path,default=ROOT/'native-results/devstral2')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();v=a.validation.resolve()
def read(name): return json.loads((v/name).read_text(encoding='utf-8-sig'))
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
required={
    'campaign':{'generator-201','reuse-201','short-20','remapped-02','long-8192','performance'},
    'v1-regressions':{'guards','glm-sites','deepseek-sites','qwen-short','glm-short','deepseek-short','qwen-mtp'},
    'gemma-smokes':{'sites-201','sites-20','short-201','reuse-201','short-20'},
}
for directory,names in required.items():
    runs=read(directory+'/runs.json')
    assert {r['name'] for r in runs}==names and all(r['returncode']==0 for r in runs),directory
assert read('glm-mtp-completion.json')['returncode']==0
generic=read('generic.json')
assert len(generic)==4 and all(r['exact'] and r['rejection_safe'] for r in generic)
sites=read('sites-201/sites.json')['records']
assert len(sites)==528 and all(r['exact'] and r['rejected_safely'] for r in sites)
for b,q in ((1,1),(2,1),(3,1),(4,1),(1,5),(2,6),(1,7)):
    stock=read(f'short-201/b{b}-q{q}-0.json'); plan=read(f'short-201/b{b}-q{q}-1.json')
    assert stock['records']==plan['records'] and stock['boundaries']==plan['boundaries']
    assert plan['counts'].get('begin',0)==(2816 if q<=6 else 0)
assert read('short-201/reload.json')['exact']
assert read('campaign/short-20/topology.json')['exact']
assert read('campaign/short-20/reload.json')['exact']
for i in (4,5): assert read(f'campaign/reuse-201/case{i}-1.json')['reuse']['pages']>0
assert all(r['median_change']>=-.03 for r in read('campaign/performance/performance.json'))
sources=read('validated-source.json')
for name,expected in sources.items(): assert sha(ROOT/name)==expected,name
extension,=(v/'extension').glob('exllamav3_ext*.pyd')
assert sha(extension)=='b89a859f85765405f77e517f6faed02456cd23a46e48226258ba27c5dd1d6fd3'
assert not subprocess.check_output(['git','diff','HEAD','--','exllamav3','doc','tests','util'],cwd=ROOT)
result=dict(all_required_checkout_gates_passed=True,
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
    extension_sha256=sha(extension),source_hashes=sources,
    scope='Full Devstral; unaffected v1 ordinary smokes plus already-completed Qwen/GLM MTP; focused Gemma v2 sites/short/reuse/topology',
    skipped='Queued DeepSeek MTP and broad Gemma campaign cancelled at user request',
    evidence=str(v),campaigns={k:sorted(names) for k,names in required.items()})
assert not a.output.exists()
a.output.write_text(json.dumps(result,indent=2))
print('PASS checkout release gates and exact source hashes')
