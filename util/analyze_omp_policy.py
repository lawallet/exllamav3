"""Offline medians/variance/parity for the OpenMP policy release investigation."""
import json
from pathlib import Path
import statistics
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/omp-policy'


def stats(xs):
    return dict(n=len(xs),median=statistics.median(xs),minimum=min(xs),maximum=max(xs),
                stdev=statistics.stdev(xs) if len(xs)>1 else None,
                variance=statistics.variance(xs) if len(xs)>1 else None)


def main():
    result={}
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():continue
        groups=defaultdict(list);references={};rows=[]
        for path in sorted(folder.glob('*.json')):
            try:data=json.loads(path.read_text(encoding='utf-8-sig'))
            except (UnicodeError,json.JSONDecodeError):continue
            if not isinstance(data,dict) or 'request_start_ns' not in data:continue
            jobs=data.get('jobs') or data.get('result',{}).get('state',{}).get('jobs')
            if not jobs:continue
            block=data.get('blocktime',data.get('extra_before',{}).get('blocktime',{}).get('milliseconds'))
            context=data.get('context','short');native=data.get('native',data.get('ir',True));mtp=data.get('mtp',True)
            signature=[(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens']) for j in jobs]
            key=(context,len(jobs),jobs[0]['new_tokens'],native,mtp,data.get('eos_mask',False))
            exact=key not in references or signature==references[key]
            references.setdefault(key,signature)
            name=path.stem
            phase=('warmup' if 'warmup' in name else 'monitored' if name.startswith('monitored') else
                   'steady' if name.startswith('steady') else 'long' if context!='short' else
                   'concurrent' if name.startswith('concurrent-r') else 'masked' if name.startswith('masked-r') else 'control')
            row=dict(file=path.name,blocktime=block,phase=phase,context=context,native=native,mtp=mtp,
                tokens=sum(j['new_tokens'] for j in jobs),jobs=len(jobs),exact=exact,
                tps=jobs[0]['new_tokens']/jobs[0]['time_generate'],prefill_s=jobs[0]['time_prefill'],
                cpu_core_equivalents=data.get('cpu_core_equivalents'),aggregate_output_tps=data.get('aggregate_output_tps'),
                accepted=[j['accepted_draft_tokens'] for j in jobs],rejected=[j['rejected_draft_tokens'] for j in jobs])
            rows.append(row);groups[f'{phase}/{context}/b{len(jobs)}/native{int(native)}/mtp{int(mtp)}/ms{block}'].append(row)
        if not rows:continue
        summary={key:{field:stats([r[field] for r in group if r[field] is not None]) for field in
                       ('tps','prefill_s','cpu_core_equivalents','aggregate_output_tps') if any(r[field] is not None for r in group)}
                 for key,group in groups.items()}
        result[folder.name]=dict(all_exact=all(r['exact'] for r in rows),requests=len(rows),groups=summary,rows=rows)
        print(folder.name,'exact',result[folder.name]['all_exact'],'requests',len(rows))
        for key,s in summary.items():
            if key.startswith(('steady','long','concurrent')):
                print(key,json.dumps(s))
    (BASE/'summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
