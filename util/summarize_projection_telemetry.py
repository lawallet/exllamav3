"""Decode-window correlations only; telemetry runs are not primary benchmarks."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path

def stats(v):
    return dict(n=len(v),median=statistics.median(v),minimum=min(v),maximum=max(v)) if v else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args()
    host=[json.loads(s) for s in (a.folder/'telemetry/host.jsonl').read_text().splitlines()]
    gpu=[json.loads(s) for s in (a.folder/'telemetry/gpu.jsonl').read_text().splitlines()]
    result={}
    for path in sorted([*a.folder.glob('monitored-*.json'),*a.folder.glob('steady-*.json')]):
        d=json.loads(path.read_text());j,=d['result']['state']['jobs']
        end=d['response_end_ns']-500_000_000
        start=d['response_end_ns']-int(j['time_generate']*1e9)+500_000_000
        h=[r for r in host if start<=r['time_ns']<=end]
        cores=[];active=[]
        for l,r in zip(h,h[1:]):
            dt=(r['time_ns']-l['time_ns'])/1e9
            cpu=lambda row:sum(row['process']['cpu_times'][k] for k in ('user','system'))
            cores.append((cpu(r)-cpu(l))/dt)
            old={t['id']:t['user_time']+t['system_time'] for t in l['process']['threads']}
            active.append(sum(t['user_time']+t['system_time']-old.get(t['id'],t['user_time']+t['system_time'])>0.001
                              for t in r['process']['threads']))
        devices=defaultdict(lambda:defaultdict(list))
        for row in gpu:
            if not start<=row['time_ns']<=end:continue
            cells=next(csv.reader([row['csv']]))
            if len(cells)!=14 or not cells[1].strip().isdigit():continue
            dev=devices[cells[1].strip()]
            for i,name in ((2,'core_mhz'),(3,'memory_mhz'),(4,'power_w'),(5,'gpu_percent'),
                           (6,'memory_percent'),(8,'temperature_c'),(9,'used_mib'),(10,'free_mib'),
                           (11,'pcie_generation'),(12,'pcie_width')):
                try:dev[name].append(float(cells[i].strip().split()[0]))
                except ValueError:pass
            dev['pstate'].append(cells[7].strip());dev['clock_reasons'].append(cells[13].strip())
        result[path.stem]=dict(context=d['context'],candidate=d['candidate'],
            decode_tps=j['new_tokens']/j['time_generate'],core_equivalents=stats(cores),
            active_threads_per_sample=stats(active),threads=stats([r['process']['thread_count'] for r in h]),
            rss=stats([r['process']['memory']['rss'] for r in h]),
            gpu={k:{n:(sorted(set(v)) if n in ('pstate','clock_reasons') else stats(v)) for n,v in dev.items()}
                 for k,dev in devices.items()},
            allocation_after=d['extra_after']['memory'],dxgi_after=d['extra_after']['dxgi'],
            candidate_after=d['attention_after']['candidate'])
    (a.folder/'telemetry-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{n:v for n,v in r.items() if n not in ('allocation_after','dxgi_after','candidate_after')} for k,r in result.items()},indent=2))
if __name__=='__main__':main()
