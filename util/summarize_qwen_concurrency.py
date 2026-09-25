"""Summarize aggregate throughput and per-job latency with external GPU telemetry."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics as st


def summarize(directory,telemetry=None):
    runs=json.loads((directory/'benchmark.json').read_text(encoding='utf-8'))
    samples=[]
    if telemetry:
        with telemetry.open() as f:
            for r in csv.DictReader(f,skipinitialspace=True):
                try:
                    if int(r['index'])!=2:continue
                    assert float(r['power.limit [W]'].split()[0])==300
                    samples.append((datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f').timestamp(),
                                    float(r['utilization.gpu [%]'].split()[0]),float(r['power.draw [W]'].split()[0])))
                except (ValueError,KeyError,TypeError):
                    continue
    result={}
    for mtp,n,stagger in sorted({(r['mtp'],r['jobs'],r['stagger']) for r in runs}):
        key=f'{"mtp" if mtp else "ordinary"}-{n}'+('-stagger' if stagger else '')
        group=result[key]={}
        for native in (False,True):
            rows=[r for r in runs if (r['mtp'],r['jobs'],r['stagger'],r['native'])==(mtp,n,stagger,native)]
            ts=[s for s in samples if any(r['start_epoch']<=s[0]<=r['end_epoch'] for r in rows)]
            entry=dict(aggregate_tps=[r['aggregate_tps'] for r in rows],
                mean_aggregate_tps=st.mean(r['aggregate_tps'] for r in rows),
                mean_gpu_utilization=st.mean(s[1] for s in ts) if ts else None,
                mean_gpu_power_watts=st.mean(s[2] for s in ts) if ts else None,
                gpu_samples=len(ts),native_calls=[r['calls']['native'] for r in rows],
                fallback_calls=[r['calls']['fallback'] for r in rows],
                shapes=sorted({k for r in rows for k in r['shapes']}),per_job={},phases={})
            for shape in entry['shapes']:
                ps=[r['phases'][shape] for r in rows if shape in r['phases']]
                seconds=sum(v['seconds'] for v in ps)
                tokens=sum(v['tokens'] for v in ps)
                entry['phases'][shape]=dict(tokens=tokens,seconds=seconds,effective_tps=tokens/seconds)
            for j in rows[0]['per_job']:
                js=[r['per_job'][j] for r in rows]
                entry['per_job'][j]={k:st.mean(v[k] for v in js) for k in
                    ('tokens','first_token_seconds','completion_seconds','effective_tps','decode_tps',
                     'max_delivery_gap_seconds','join_delivery_gap_seconds','proposed','accepted','rejected','acceptance')}
            group['native' if native else 'stock']=entry
        group['improvement_percent']=100*(group['native']['mean_aggregate_tps']/group['stock']['mean_aggregate_tps']-1)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--telemetry',type=Path)
    a=p.parse_args()
    print(json.dumps(summarize(a.directory,a.telemetry),indent=2))
