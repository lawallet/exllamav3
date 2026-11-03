"""Keep nested host envelopes distinct from sampled CUDA event intervals."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args();result={}
    for path in a.folder.glob('*-attention.json'):
        d=json.loads(path.read_text());groups=defaultdict(lambda:dict(calls=0,wall_ms=0,exclusive_ms=0))
        for name,r in d['rows'].items():
            if not name.startswith(('Attention.','Projection.','ProjectionPlan.')):continue
            g=groups[name.split('|')[0]];g['calls']+=r['count'];g['wall_ms']+=r['wall_ns']/1e6;g['exclusive_ms']+=r['exclusive_ns']/1e6
        events=defaultdict(list)
        for e in d['events']:events[str((e['key'][0],e['device'],e['key'][2]))].append(e['gpu_ms'])
        result[path.name]=dict(host=dict(groups),sampled_intervals={k:dict(n=len(v),median_ms=statistics.median(v),minimum_ms=min(v),maximum_ms=max(v)) for k,v in events.items()})
    (a.folder/'attention-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
