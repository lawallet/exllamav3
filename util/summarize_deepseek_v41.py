"""Summarize saved timing, repeatability and GPU trace evidence (no inference)."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result={'timings':{},'traces':{}}
    groups=defaultdict(list)
    for path in a.directory.glob('c*-r*-mtp*-ir*.json'):
        if any(s in path.stem for s in ('profile','trace','audit')):continue
        d=json.loads(path.read_text());jobs=d['result']['state']['jobs']
        if not jobs:continue
        j=jobs[0];key=path.stem.split('-r')[0]+'-'+path.stem.split('-mtp')[1]
        groups[key].append(dict(file=path.name,tps=j['new_tokens']/j['time_generate'],
            prefill=j['time_prefill'],prompt=j['prompt_tokens'],tokens=j['new_tokens'],
            ids_sha256=hashlib.sha256(str(j['ids']).encode()).hexdigest(),
            accepted=j['accepted_draft_tokens'],rejected=j['rejected_draft_tokens'],memory=j['memory']))
    for k,rows in groups.items():
        t=[r['tps'] for r in rows]
        result['timings'][k]=dict(median=statistics.median(t),minimum=min(t),maximum=max(t),
            stdev=statistics.stdev(t) if len(t)>1 else 0,identical_ids=len({r['ids_sha256'] for r in rows})==1,runs=rows)
    for path in a.directory.glob('*-trace-trace.json'):
        events=json.loads(path.read_text())['traceEvents']
        categories=defaultdict(float);devices=defaultdict(list);counts=Counter();host=defaultdict(float)
        for e in events:
            cat,name=e.get('cat'),e.get('name','');dur=e.get('dur',0)
            if cat=='user_annotation':host[name]+=dur
            if cat not in ('kernel','gpu_memcpy'):continue
            device=e['args'].get('device');devices[device].append((e['ts'],e['ts']+dur))
            if cat=='gpu_memcpy':kind='copy:'+name
            elif 'hc_' in name:kind='hyperconnection'
            elif 'moe_' in name:kind='expert'
            elif 'gemm' in name or 'gemv' in name:kind='gemm/gemv'
            elif 'dsa_' in name or 'dsv4_' in name or 'attn' in name:kind='attention'
            elif 'norm' in name:kind='normalization'
            elif 'rope' in name:kind='rope'
            elif 'routing' in name:kind='routing'
            else:kind='other'
            categories[kind]+=dur;counts[kind]+=1
        active={}
        for device,intervals in devices.items():
            lo=hi=total=0
            for start,end in sorted(intervals):
                if start>hi:total+=hi-lo;lo,hi=start,end
                else:hi=max(hi,end)
            active[device]=(total+hi-lo)/1000
        result['traces'][path.name]=dict(kernel_and_copy_ms={k:v/1000 for k,v in categories.items()},
            counts=dict(counts),device_active_ms=active,
            target_forwards=sum(e.get('cat')=='user_annotation' and e.get('name')=='ds41.Model.forward:text' for e in events),
            draft_forwards=sum(e.get('cat')=='user_annotation' and e.get('name')=='ds41.Model.forward:mtp' for e in events),
            host_annotation_ms={k:v/1000 for k,v in host.items()})
    (a.directory/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{key:({kk:vv for kk,vv in val.items() if kk not in ('runs','host_annotation_ms')})
                         for key,val in section.items()} for k,section in result.items()},indent=2))

if __name__=='__main__':main()
