"""Aggregate and per-job latency plus both-GPU telemetry, retaining every run."""
import json
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from util.summarize_qwen_concurrency import summarize

def main():
    p=Path(sys.argv[1]); result=summarize(p)
    runs=json.loads((p/'benchmark.json').read_text()); samples=[]
    for line in (p/'telemetry.jsonl').read_text().splitlines():
        item=json.loads(line)
        for row in item['csv'].splitlines():
            cells=[v.strip() for v in row.split(',')]
            if len(cells)!=6 or cells[0] not in ('0','2'): continue
            try: vals=list(map(float,cells[2:]))
            except ValueError: continue
            samples.append((item['time'],cells[0],vals))
    for run in runs:
        run['telemetry']={}
        for gpu in ('2','0'):
            rows=[v for t,g,v in samples if g==gpu and run['start_epoch']<=t<=run['end_epoch']]
            run['telemetry'][gpu]={key:dict(mean=st.mean(v[i] for v in rows),maximum=max(v[i] for v in rows),samples=len(rows)) for i,key in enumerate(('utilization','power','vram_mib','power_limit'))} if rows else {}
    for name,group in result.items():
        for native in (False,True):
            mode=group['native' if native else 'stock']
            rows=[r for r in runs if ('mtp' if r['mtp'] else 'ordinary')+'-'+str(r['jobs'])+('-stagger' if r['stagger'] else '')==name and r['native']==native]
            mode['per_run_telemetry']=[r['telemetry'] for r in rows]
            mode['transfer_counts']=[{k:v for k,v in r['calls'].items() if k.startswith('transfer/')} for r in rows]
    (p/'summary.json').write_text(json.dumps(result,indent=2))
    (p/'benchmark-telemetry.json').write_text(json.dumps(runs,indent=2))
    print(json.dumps({k:dict(stock=v['stock']['mean_aggregate_tps'],native=v['native']['mean_aggregate_tps'],improvement=v['improvement_percent']) for k,v in result.items()},indent=2))

if __name__=='__main__': main()
