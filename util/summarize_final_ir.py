"""Summarize profile_final_ir artifacts; never treat instrumented runs as timings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pstats
import statistics


def summarize(root):
    groups = defaultdict(list)
    path = root / 'results.json'
    for row in json.loads(path.read_text()) if path.exists() else []:
        assert not row['instrumented']
        groups[(row['mode'], row['length'])].append(row)
    timings = []
    for (mode, length), rows in groups.items():
        rates = [r['decode_tps'] for r in rows]
        timings.append(dict(mode=mode, length=length, samples=len(rows),
            decode_median=statistics.median(rates), decode_min=min(rates), decode_max=max(rates),
            decode_stdev=statistics.stdev(rates) if len(rates)>1 else None,
            wall_median=statistics.median(r['wall_tps'] for r in rows),
            completions_identical=len({tuple(r['completion_hashes']) for r in rows}) == 1,
            acceptance=[dict(accepted=sum(s['accepted_draft_tokens'] or 0 for s in r['stats']),
                             rejected=sum(s['rejected_draft_tokens'] or 0 for s in r['stats'])) for r in rows],
            execution_deltas=[r['deltas'] for r in rows]))
    pairs = []
    for baseline, candidate in (('stock', 'native'), ('native', 'graphs')):
        for length in sorted({length for mode,length in groups}):
            reference = {r['pair']:r for r in groups.get((baseline,length), [])}
            changed = {r['pair']:r for r in groups.get((candidate,length), [])}
            matched = sorted(reference.keys() & changed.keys())
            if matched:
                changes = [100*(changed[i]['decode_tps']/reference[i]['decode_tps']-1) for i in matched]
                pairs.append(dict(baseline=baseline, candidate=candidate, length=length,
                    paired_percent_changes=changes, median_paired_percent=statistics.median(changes),
                    completions_match=[reference[i]['completion_hashes']==changed[i]['completion_hashes'] for i in matched],
                    acceptance_match=[[(s['accepted_draft_tokens'],s['rejected_draft_tokens']) for s in reference[i]['stats']] ==
                                      [(s['accepted_draft_tokens'],s['rejected_draft_tokens']) for s in changed[i]['stats']]
                                      for i in matched]))
    profiles = {}
    for path in root.glob('*.prof'):
        stats = pstats.Stats(str(path))
        entries = [dict(file=k[0], line=k[1], function=k[2], calls=v[1], self_seconds=v[2],
                        cumulative_seconds=v[3]) for k,v in stats.stats.items()]
        profiles[path.name] = dict(total_seconds=stats.total_tt,
            top_self=sorted(entries, key=lambda e:e['self_seconds'], reverse=True)[:35],
            selected=[e for e in entries if ('native_tp.py' in e['file'] or
                e['function'] in ('iterate_draftmodel_mtp_gen','iterate_gen','forward_tp',
                                  'all_reduce','broadcast','sample','_sample','tp_drain_acks') or
                'BC_BlockPlan' in e['function'] or 'pg_all_reduce' in e['function'])])
    host = {}
    for path in root.glob('profile-*.json'):
        data = json.loads(path.read_text())
        ranks = []
        for rank in data.get('rank_host_timings', []):
            stages = defaultdict(lambda: [0,0])
            for name, (count,ns) in rank['totals'].items():
                key = name if name.startswith('collective_launch:') else name.rsplit(':',1)[-1]
                stages[key][0] += count
                stages[key][1] += ns
            ranks.append(dict(rank=rank['rank'], device=rank['device'],
                stages={k:dict(calls=v[0], total_ms=v[1]/1e6, mean_us=v[1]/v[0]/1e3)
                        for k,v in stages.items()}))
        host[path.name] = dict(observed_start_tokens=data.get('observed_start_tokens'),
            target_shapes=data.get('target_shapes'), draft_shapes=data.get('draft_shapes'),
            ranks=ranks, parent={k:dict(calls=v[0],total_ms=v[1]/1e6,mean_us=v[1]/v[0]/1e3)
                                for k,v in data.get('parent_host_timings',{}).items()})
    return dict(completed=(root/'completed.json').exists(), timings=timings,
                pairs=pairs, profiles=profiles, host_timings=host)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps({str(p):summarize(p) for p in args.directories}, indent=2))


if __name__ == '__main__':
    main()
