"""Summarize matched workloads separately; never pool profiler or startup runs."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/generation-pipeline'

def stats(v):
    return dict(n=len(v),median=statistics.median(v),minimum=min(v),maximum=max(v),
        stdev=statistics.stdev(v) if len(v)>1 else None,
        variance=statistics.variance(v) if len(v)>1 else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={}
    folders=list((BASE/'cross-model-v3').glob('*'))+[BASE/'qwen-order-recheck']+list((BASE/'tp-controls').glob('*'))
    folders+=list((BASE/'mode-controls').glob('*'))
    for folder in folders:
        if not folder.is_dir() or not (folder/'complete.json').exists():continue
        groups=defaultdict(lambda:defaultdict(list))
        for path in folder.glob('*.json'):
            d=json.loads(path.read_text())
            if 'request_start_ns' not in d or not d.get('jobs'):continue
            name=d['name']
            if 'warm' in name:continue
            kind=next((k for k in ('steady','long-r','concurrent-r','masked-r','masked512','unmasked','stock','ordinary','monitored') if name.startswith(k)),None)
            if kind is None:continue
            j=d['jobs'][0];g=groups[(kind,d['candidate'])]
            g['decode_first_job'].append(j['new_tokens']/j['time_generate'])
            g['prefill_first_job_s'].append(j['time_prefill'])
            g['request_s'].append(d['wall_s'])
            if len(d['jobs'])>1:
                g['aggregate_request_tps'].append(sum(x['new_tokens'] for x in d['jobs'])/d['wall_s'])
        rows={str(k):{n:stats(v) for n,v in g.items()} for k,g in groups.items()}
        result[folder.relative_to(BASE).as_posix()]=rows
    a.output.write_text(json.dumps(result,indent=2))
    for folder,groups in result.items():
        print(folder)
        for kind in ('steady','long-r','concurrent-r','ordinary','stock','unmasked','masked512','monitored'):
            pair=[groups.get(str((kind,b))) for b in (False,True)]
            if not all(pair):continue
            metric='aggregate_request_tps' if kind=='concurrent-r' else 'decode_first_job'
            rates=[v[metric]['median'] for v in pair]
            prefill=[v['prefill_first_job_s']['median'] for v in pair]
            print(kind,metric,[round(v,3) for v in rates],f'{100*(rates[1]/rates[0]-1):+.2f}%',
                'prefill',[round(v,4) for v in prefill])
if __name__=='__main__':main()
