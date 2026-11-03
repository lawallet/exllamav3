"""Cross-model exact-parity audit, including retained pre-pipeline evidence."""
import argparse,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/generation-pipeline'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    refs={};counts={};groups=defaultdict(list);failures=[]
    folders=list((ROOT/'native-results/deepseek-v41/omp-policy').glob('candidate-*'))
    folders+=list((BASE/'cross-model-v3').glob('*'))+[BASE/'qwen-order-recheck']
    folders+=list((BASE/'tp-controls').glob('*'))+list((BASE/'mode-controls').glob('*'))
    for folder in folders:
        if not folder.is_dir() or not (folder/'metadata.json').exists():continue
        if folder.is_relative_to(BASE) and not (folder/'complete.json').exists():continue
        meta=json.loads((folder/'metadata.json').read_text());case=meta['case']
        for path in sorted(folder.glob('*.json')):
            if 'telemetry' in path.name:continue
            d=json.loads(path.read_text())
            if 'request_start_ns' not in d or not d.get('jobs'):continue
            jobs=d['jobs']
            key=(case,d['context'],len(jobs),jobs[0]['new_tokens'],d['mtp'],d.get('eos_mask',False),d.get('oracle'))
            signature=json.dumps([(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens']) for j in jobs])
            digest=hashlib.sha256(signature.encode()).hexdigest()
            if key in refs and refs[key]['sha256']!=digest:failures.append(dict(file=str(path),reference=refs[key]['file']))
            else:refs.setdefault(key,dict(file=str(path),sha256=digest))
            counts[str(folder.relative_to(ROOT))]=counts.get(str(folder.relative_to(ROOT)),0)+1
            if folder.is_relative_to(BASE):
                assert d['exact'] and all(s['milliseconds']==0 for s in d['runtime']),path
                if d['name'].startswith('steady'):
                    groups[str((str(folder.relative_to(BASE)),d['candidate']))].append(jobs[0]['new_tokens']/jobs[0]['time_generate'])
    summary=dict(exact=not failures,cohorts=len(refs),requests_by_folder=counts,failures=failures,
        primary={k:dict(n=len(v),median=statistics.median(v),minimum=min(v),maximum=max(v),stdev=statistics.stdev(v),variance=statistics.variance(v)) for k,v in groups.items()})
    a.output.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));assert not failures
if __name__=='__main__':main()
