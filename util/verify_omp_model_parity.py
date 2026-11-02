"""Compare saved full token/MTP outputs across baseline, staged and installed code."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate-prefix',default='candidate')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    groups={};counts={};failures=[]
    folders=[f for f in BASE.iterdir() if f.is_dir() and
             (f.name.startswith(('baseline','deepseek-sweep','old-release-final-control','candidate')) or f.name.startswith(a.candidate_prefix))]
    for folder in sorted(folders):
        meta=folder/'metadata.json'
        if meta.exists():
            case=json.loads(meta.read_text())['case']
        elif 'deepseek' in folder.name or folder.name=='old-release-final-control':case='deepseek41'
        else:continue
        for path in sorted(folder.glob('*.json')):
            try:data=json.loads(path.read_text(encoding='utf-8-sig'))
            except (UnicodeError,json.JSONDecodeError):continue
            if not isinstance(data,dict) or 'request_start_ns' not in data:continue
            jobs=data.get('jobs') or data.get('result',{}).get('state',{}).get('jobs')
            if not jobs:continue
            # Native and stock must agree. Ordinary and MTP may differ in accepted
            # counts and therefore have separate signatures, even when tokens agree.
            mtp=data.get('mtp',True)
            key=(case,data.get('context','short'),len(jobs),jobs[0]['new_tokens'],mtp,data.get('eos_mask',False))
            signature=json.dumps([(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens']) for j in jobs])
            digest=hashlib.sha256(signature.encode()).hexdigest()
            if key in groups and groups[key]['sha256']!=digest:
                failures.append(dict(file=str(path),reference=groups[key]['file']))
            else:groups.setdefault(key,dict(sha256=digest,file=str(path)))
            counts[folder.name]=counts.get(folder.name,0)+1
    result=dict(exact=not failures,cohorts=len(groups),requests_by_folder=counts,failures=failures,
                references={str(k):v for k,v in groups.items()})
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='references'},indent=2))
    assert any(k.startswith(a.candidate_prefix) for k in counts),'candidate outputs missing'
    assert not failures,'see saved mismatches'


if __name__=='__main__':main()
