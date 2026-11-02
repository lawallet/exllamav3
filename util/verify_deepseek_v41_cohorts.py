"""Strict same-mode token/logit/repeatability checks on completed HTTP cohorts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
a=p.parse_args();groups=defaultdict(list);total_oracles=0
for path in sorted(a.directory.glob('c*-r*-mtp*-ir*.json')):
    if 'trace' in path.name or 'profile' in path.name:continue
    d=json.loads(path.read_text());s=d['result']['state']
    key=path.stem.split('-r')[0]+'-mtp'+path.stem.split('-mtp')[1][0]
    groups[key].append((path.name,d,s))
report={}
for key,rows in groups.items():
    _,_,reference=rows[0]
    assert len(rows)>=2,(key,'not enough comparisons')
    for name,d,s in rows:
        assert s['jobs'][0]['ids']==reference['jobs'][0]['ids'],(name,'token mismatch')
        if s.get('logits'):
            assert s['logits']==reference['logits'],(name,'logit/forward mismatch')
        for field in ('accepted_draft_tokens','rejected_draft_tokens'):
            assert s['jobs'][0][field]==reference['jobs'][0][field],(name,field)
        if '-ir1' in name and '-audit' in name:
            assert s['oracles'] and s['executions'],(name,'no native execution')
            assert all(o['exact'] for o in s['oracles']),(name,'inexact boundary')
            assert len(s['executions'])==(43 if key.endswith('1') else 40),(name,'missing block')
        total_oracles+=len(s.get('oracles',[]))
    report[key]=dict(runs=len(rows),tokens=reference['jobs'][0]['new_tokens'],exact=True,
                     logits_checked=bool(reference.get('logits')))
assert report,'no cohorts'
result=dict(groups=report,total_exact_oracles=total_oracles)
(a.directory/'parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
