"""Separate CUDA activity, API envelopes and nested host scopes; never sum them."""
import argparse,json,statistics
from collections import Counter,defaultdict
from pathlib import Path

def union_us(intervals):
    total=0;end=None
    for left,right in sorted(intervals):
        if end is None or left>end:total+=right-left;end=right
        elif right>end:total+=right-end;end=right
    return total

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folders',nargs='+',type=Path);a=ap.parse_args()
    for folder in a.folders:
        path=folder/'trace-short-trace.json'
        events=json.loads(path.read_text())['traceEvents']
        kernels=defaultdict(list);apis=defaultdict(list);scopes=[];mem=Counter()
        for e in events:
            if e.get('ph')!='X' or 'dur' not in e:continue
            cat=e.get('cat','');name=e.get('name','')
            if cat=='kernel':kernels[str(e.get('args',{}).get('device',e.get('pid')))].append(e)
            elif cat in ('cuda_runtime','cuda_driver'):apis[(cat,name)].append(e['dur'])
            elif cat in ('gpu_memcpy','gpu_memset'):mem[name]+=1
            if name.startswith('ds41.projections:'):scopes.append(e['dur'])
        result=dict(note='GPU sums/active unions and host/API envelopes are separate overlapping quantities, not removable wall time.',
            gpu={k:dict(kernels=len(v),summed_kernel_ms=sum(e['dur'] for e in v)/1000,
                        active_union_ms=union_us([(e['ts'],e['ts']+e['dur']) for e in v])/1000,
                        span_ms=(max(e['ts']+e['dur'] for e in v)-min(e['ts'] for e in v))/1000,
                        kernel_names=dict(Counter(e['name'] for e in v))) for k,v in kernels.items()},
            api={str(k):dict(count=len(v),sum_ms=sum(v)/1000,median_us=statistics.median(v),max_us=max(v)) for k,v in apis.items()},
            memory_activity=dict(mem),projection_host_scopes=dict(count=len(scopes),sum_ms=sum(scopes)/1000,
                median_us=statistics.median(scopes) if scopes else None))
        (folder/'trace-analysis.json').write_text(json.dumps(result,indent=2))
        brief={**result,'gpu':{k:{n:x for n,x in v.items() if n!='kernel_names'} for k,v in result['gpu'].items()}}
        print(folder.name,json.dumps(brief,indent=2))
if __name__=='__main__':main()
