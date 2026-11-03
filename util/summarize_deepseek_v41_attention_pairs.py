"""Summarize only within-process, matched, uninstrumented projection pairs."""
import argparse,json,statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args()
    groups={};refs={}
    for p in sorted(a.folder.glob('steady-*.json')):
        d=json.loads(p.read_text());j,=d['result']['state']['jobs'];context=d['context']
        sig=(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens'])
        assert context not in refs or refs[context]==sig;refs[context]=sig
        g=groups.setdefault(context,{False:[],True:[], 'pairs':{}})
        row=dict(name=d['name'],tps=j['new_tokens']/j['time_generate'],prefill=j['time_prefill'],
                 accepted=j['accepted_draft_tokens'],rejected=j['rejected_draft_tokens'])
        g[d['candidate']].append(row)
        g['pairs'].setdefault(d['name'].rsplit('-candidate',1)[0],{})[d['candidate']]=row['tps']
    result={}
    for context,g in groups.items():
        out={'exact_tokens_and_mtp':True}
        for flag in (False,True):
            v=[r['tps'] for r in g[flag]]
            out['candidate' if flag else 'baseline']=dict(n=len(v),median=statistics.median(v),minimum=min(v),maximum=max(v),
                stdev=statistics.stdev(v) if len(v)>1 else 0,rows=g[flag])
        gains=[(p[True]/p[False]-1)*100 for p in g['pairs'].values() if len(p)==2]
        out['paired_percent']=dict(values=gains,median=statistics.median(gains),minimum=min(gains),maximum=max(gains))
        result[context]=out
    (a.folder/'pair-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
