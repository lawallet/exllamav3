"""Summarize saved HTTP backend timings and verify full-token/MTP parity."""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quiet',action='store_true')
    a=p.parse_args();groups=defaultdict(list);signatures={};checks=[]
    for folder in a.folders:
        for path in sorted(folder.glob('*.json')):
            data=json.loads(path.read_text())
            if 'result' not in data or 'response' not in data:continue
            jobs=data['result']['state']['jobs']
            assert len(jobs)==1,path
            j=jobs[0]
            assert j['cached_tokens']==0,(path,j['cached_tokens'])
            key=(data['context'],data.get('tokens',512),data.get('mtp',True),data.get('minimum',512),data.get('ir',True))
            sig=(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens'])
            if key in signatures:assert sig==signatures[key],str(path)+' token/MTP mismatch'
            else:signatures[key]=sig
            checks.append(str(path))
            if not data['name'].startswith('steady'):continue
            group=(str(folder),*key,data.get('serial_readback',False),data.get('candidate',False),data.get('async_seed',False))
            groups[str(group)].append(j['new_tokens']/j['time_generate'])
    summary=dict(parity_requests=len(checks),cohorts=len(signatures),files=checks,groups={})
    for key,values in groups.items():
        summary['groups'][key]=dict(n=len(values),median=statistics.median(values),minimum=min(values),maximum=max(values),
            stdev=statistics.stdev(values) if len(values)>1 else None,variance=statistics.variance(values) if len(values)>1 else None,values=values)
    a.output.write_text(json.dumps(summary,indent=2))
    if a.quiet:
        print(json.dumps(dict(parity_requests=len(checks),cohorts=len(signatures),groups=len(groups))))
    else:
        print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))
if __name__=='__main__':main()
