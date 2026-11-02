"""Align separate telemetry to saved request windows; correlations, not causation."""
import argparse,datetime,json,statistics
from collections import defaultdict
from pathlib import Path
def summary(v):
    return dict(n=len(v),median=statistics.median(v),mean=statistics.mean(v),minimum=min(v),maximum=max(v)) if v else {}
def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--telemetry',default='telemetry');a=p.parse_args()
    host=[json.loads(s) for s in (a.folder/a.telemetry/'host.jsonl').read_text().splitlines()]
    metadata=json.loads((a.folder/a.telemetry/'metadata.json').read_text())
    tree_pids=set(metadata.get('process_tree_pids',[metadata['pid']]))
    resolution=a.folder/a.telemetry/'process-resolution.json'
    actual_pid=json.loads(resolution.read_text())['inference_pid'] if resolution.exists() else None
    gpu=[]
    for line in (a.folder/a.telemetry/'gpu.jsonl').read_text().splitlines():
        r=json.loads(line);values=[v.strip() for v in r['csv'].split(',')]
        try:
            stamp=datetime.datetime.strptime(values[0],'%Y/%m/%d %H:%M:%S.%f').timestamp()*1e9
            row=dict(time_ns=stamp,index=int(values[1]))
            names=['sm_mhz','memory_mhz','power_w','gpu_util','memory_util','pstate','temperature_c','used_mib','free_mib','pcie_gen','pcie_width','clock_reasons']
            for name,value in zip(names,values[2:]):
                try:value=float(value.split()[0])
                except ValueError:pass
                row[name]=value
            gpu.append(row)
        except ValueError:pass
    result={}
    for path in a.folder.glob('*.json'):
        try:data=json.loads(path.read_text(encoding='utf-8-sig'))
        except json.JSONDecodeError:continue  # A live request may still be writing its artifact.
        if not isinstance(data,dict) or 'request_start_ns' not in data:continue
        jobs=data.get('jobs') or data.get('result',{}).get('state',{}).get('jobs')
        if not jobs:continue
        job=jobs[0];entry={}
        for phase,lo,hi in [('request',data['request_start_ns'],data['response_end_ns']),
                            ('prefill',data['request_start_ns'],data['request_start_ns']+job['time_prefill']*1e9),
                            ('decode',data['response_end_ns']-job['time_generate']*1e9,data['response_end_ns'])]:
            h=[r for r in host if lo<=r['time_ns']<=hi];g=[r for r in gpu if lo<=r['time_ns']<=hi]
            if len(h)<2:continue
            span=(h[-1]['time_ns']-h[0]['time_ns'])/1e9
            cpu=[sum(r['process']['cpu_times'][k] for k in ('user','system')) for r in (h[0],h[-1])]
            if actual_pid is not None:cpu=[next(sum(q['cpu']) for q in r['other_cpu'] if q['pid']==actual_pid) for r in (h[0],h[-1])]
            tree_cpu=[sum(sum(q['cpu']) for q in r['other_cpu'] if q['pid'] in tree_pids) for r in (h[0],h[-1])]
            before={t['id']:t['user_time']+t['system_time'] for t in h[0]['process']['threads']}
            threads=sorted([dict(id=t['id'],cpu_s=t['user_time']+t['system_time']-before.get(t['id'],0)) for t in h[-1]['process']['threads']],key=lambda r:-r['cpu_s'])
            processes={r['pid']:sum(r['cpu']) for r in h[0]['other_cpu']}
            others=sorted([dict(pid=r['pid'],name=r['name'],cpu_s=sum(r['cpu'])-processes.get(r['pid'],sum(r['cpu']))) for r in h[-1]['other_cpu'] if r['pid']!=0],key=lambda r:-r['cpu_s'])
            counters=defaultdict(list);perf=[]
            for r in h:
                for counter,values in r['pdh'].items():
                    if 'error' in values:continue
                    if '_Total' in values:counters[counter].append(values['_Total'])
                    elif len(values)==1:counters[counter].append(next(iter(values.values())))
                    if 'GPU Adapter Memory' in counter:counters[counter+':sum'].append(sum(values.values()))
                busy=r['pdh'].get(r'\Processor Information(*)\% Processor Time',{})
                performance=r['pdh'].get(r'\Processor Information(*)\% Processor Performance',{})
                for core,util in busy.items():
                    if '_Total' not in core and util>=30 and core in performance:perf.append(performance[core])
            entry[phase]=dict(samples=len(h),span_s=span,server_cpu_core_equivalents=(cpu[1]-cpu[0])/span,
                process_tree_core_equivalents=(tree_cpu[1]-tree_cpu[0])/span,
                top_threads=threads[:12],top_cpu_processes=others[:12],
                process_io_delta={k:h[-1]['process']['io'][k]-h[0]['process']['io'][k] for k in h[0]['process']['io']},
                process_memory={k:summary([r['process']['memory'][k] for r in h]) for k in h[0]['process']['memory']},
                pdh={k:summary(v) for k,v in counters.items()},busy_core_performance_percent=summary(perf),
                pcie_samples=[r['pcie'] for r in h],
                gpu={str(i):{k:summary([r[k] for r in g if r['index']==i and isinstance(r[k],(int,float))])
                    if all(isinstance(r[k],(int,float)) for r in g if r['index']==i) else sorted(set(str(r[k]) for r in g if r['index']==i))
                    for k in g[0] if k not in ('time_ns','index')} for i in (0,2)} if g else {})
            if actual_pid is not None:
                for k in ('top_threads','process_io_delta','process_memory'):entry[phase][k]={'unavailable':'initial monitor sampled launcher; see process-resolution.json'}
        if entry:result[path.name]=dict(tps=job['new_tokens']/job['time_generate'],prefill_s=job['time_prefill'],phases=entry)
    (a.folder/(a.telemetry+'-summary.json')).write_text(json.dumps(result,indent=2))
    for name,r in result.items():
        print(name,round(r['tps'],3),round(r['prefill_s'],3))
        for phase,d in r['phases'].items():
            if phase=='request':continue
            print(phase,'CPU',round(d['server_cpu_core_equivalents'],2),'perf',d['busy_core_performance_percent'],
                'page_reads',d['pdh'].get(r'\Memory\Page Reads/sec'),'disk',d['pdh'].get(r'\PhysicalDisk(*)\Disk Read Bytes/sec'))
            print('GPU',[(i,{k:v for k,v in g.items() if k in ('sm_mhz','memory_mhz','gpu_util','used_mib','free_mib')}) for i,g in d['gpu'].items()])
if __name__=='__main__':main()
