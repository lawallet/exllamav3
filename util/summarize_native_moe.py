"""Summarize paired native MoE benchmarks and CUPTI traces (not additive timings)."""
import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics as st
from summarize_nemotron_moe import merge, overlap, length


def quant(xs,q):
    xs=sorted(xs)
    return xs[int((len(xs)-1)*q)] if xs else None


def traces(path):
    data=json.loads(path.read_text())
    es=[e for e in data['traceEvents'] if e.get('ph')=='X' and 'dur' in e]
    es.sort(key=lambda e:e['ts'])
    timestamps=[e['ts'] for e in es]
    def interval(e):return e['ts'],e['ts']+e['dur']
    annotations=[e for e in es if e.get('cat')=='user_annotation' and e['name'].startswith('IRMOE:')]
    targets=[interval(e) for e in annotations if e['name']=='IRMOE:target']
    windows=[interval(e) for e in annotations if e['name']=='IRMOE:round']
    def within(e,ranges):return any(a<=e['ts']<b for a,b in ranges)
    gpu=[e for e in es if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    busy=merge(interval(e) for e in gpu)
    api=[e for e in es if e.get('cat') in ('cuda_runtime','cuda_driver')]
    # Label target expert preparation as the region from argsort/IR stage entry
    # to the grouped-expert call; this includes Python metadata preparation.
    prep=[]
    for m in annotations:
        if m['name']!='IRMOE:moe' or not within(m,targets):continue
        children=es[bisect.bisect_left(timestamps,m['ts']):bisect.bisect_left(timestamps,m['ts']+m['dur'])]
        begin=next((e for e in children if e['name'] in ('aten::argsort','IRMOE:stage')),None)
        expert=next((e for e in children if e['name']=='IRMOE:experts'),None)
        if begin and expert:prep.append((begin['ts'],expert['ts']))
    labels={}
    for label in ('experts','gather','assignment','shared','latent_in','latent_out'):
        labels[label]=[interval(e) for e in annotations if e['name']=='IRMOE:'+label and within(e,targets)]
    labels['preparation']=prep
    labels['target']=targets
    labels['round']=windows
    runtime_corr={e.get('args',{}).get('correlation'):e for e in api}
    result={}
    for label,ranges in labels.items():
        def in_ranges(t):return any(a<=t<b for a,b in ranges)
        activities=[e for e in gpu if (r:=runtime_corr.get(e.get('args',{}).get('correlation'))) and in_ranges(r['ts'])]
        calls=[e for e in api if in_ranges(e['ts'])]
        sync=[e for e in calls if 'Synchronize' in e['name'] or e['name'] in ('cudaMemcpy','cuMemcpyDtoH_v2')]
        span=length(ranges); used=overlap(ranges,busy)
        result[label]=dict(host_inclusive_ms=span/1000,host_no_device_activity_ms=(span-used)/1000,
            gpu_ms=sum(e['dur'] for e in activities)/1000,gpu_calls=len(activities),
            host_sync_ms=sum(e['dur'] for e in sync)/1000, sync_calls=len(sync),
            sync_names=dict(Counter(e['name'] for e in sync)),
            kernel_counts=dict(Counter(e['name'] for e in activities)),
            api_counts=dict(Counter(e['name'] for e in calls)))
    kernel_groups=defaultdict(lambda:[0,0.])
    for e in gpu:
        kernel_groups[e['name']][0]+=1;kernel_groups[e['name']][1]+=e['dur']/1000
    record=dict(rounds=len(windows), stages=result,
                kernels=dict(sorted(kernel_groups.items(),key=lambda kv:-kv[1][1])))
    path.with_name(path.name.replace('.trace.json','.analysis.json')).write_text(json.dumps(record,indent=2))
    print(path.stem,'rounds',len(windows),'target',result['target']['host_inclusive_ms']/len(windows),
          'prep',result['preparation']['host_inclusive_ms']/len(windows),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    groups=defaultdict(list)
    hosts=[]
    for path in a.directory.glob('c*.json'):
        if '.trace.' in path.name or '.analysis.' in path.name:continue
        r=json.loads(path.read_text())
        if r.get('host_spans'):
            spans=defaultdict(lambda:[0,0.])
            for label,begin,end in r['host_spans']:
                spans[label][0]+=1;spans[label][1]+=(end-begin)/1e6
            hosts.append(dict(name=r['name'],rounds=len(r['rounds']),per_round={k:dict(
                calls=v[0]/len(r['rounds']),inclusive_ms=v[1]/len(r['rounds'])) for k,v in spans.items()}))
            continue
        if r['repeat']<0 or r['audit'] or r['trace']:continue
        groups[(r['context'],r['batch'],r['ir'])].append(r)
    results=[]
    for (context,batch,ir),rs in sorted(groups.items()):
        rates=[r['tokens']/r['seconds'] for r in rs]
        bursts=[x['seconds']*1000 for r in rs for x in r['rounds']]
        delivery=[]
        for r in rs:
            for i in range(batch):
                prev=0
                for x in r['rounds']:
                    if x['after'][i]>x['before'][i]:
                        delivery.append((x['at']-prev)*1000);prev=x['at']
        first=rs[0]
        results.append(dict(context=context,batch=batch,ir=ir,repeats=len(rs),rates=rates,
            tps_median=st.median(rates),tps_min=min(rates),tps_max=max(rates),
            round_ms=[r['seconds']/len(r['rounds'])*1000 for r in rs],
            burst_p50_ms=quant(bursts,.5),burst_p95_ms=quant(bursts,.95),
            delivery_p50_ms=quant(delivery,.5),delivery_p95_ms=quant(delivery,.95),
            prompt_tokens=first['prompt_tokens'],begin=first['begin'],end=first['end'],
            tokens=first['tokens'],accepted=first['accepted'],rejected=first['rejected'],
            tuning_hashes=sorted({r['tune_sha256'] for r in rs}),
            streams_repeat_exact=len({tuple(r['completion_hashes']) for r in rs})==1,
            allocated=[r['allocated'] for r in rs],peak_allocated=[r['peak_allocated'] for r in rs]))
    (a.directory/'measurements.json').write_text(json.dumps(results,indent=2))
    if hosts:(a.directory/'host-measurements.json').write_text(json.dumps(hosts,indent=2))
    for path in a.directory.glob('*.trace.json'):traces(path)
    for r in results:print(r['context'],r['batch'],r['ir'],r['rates'],flush=True)


if __name__=='__main__':main()
