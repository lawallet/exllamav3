"""Report measured warmed medians; keep historical and fresh baselines distinct."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'native-results/v1.6.6-integration'


def read(path):
    return json.loads(path.read_text()) if path.exists() else []


def stats(values):
    return dict(median=median(values), minimum=min(values), maximum=max(values), samples=values)


def acceptance(row):
    records = row['stats']
    if isinstance(records, dict):
        records = [records]
    return [(r['accepted_draft_tokens'], r['rejected_draft_tokens']) for r in records]


comparisons = []
def compare(label, old, new, kind, **extra):
    if not old or not new:
        return
    before, after = stats(old), stats(new)
    comparisons.append(dict(workload=label, baseline_kind=kind, before=before, after=after,
                            change_percent=100*(after['median']/before['median']-1), **extra))


old_glm = read(OUT/'baseline-priority/glm-bench/results.json')
new_glm = read(OUT/'runtime-followup/glm-bench/results.json')
for length in (128, 1024):
    for mode in ('native', 'graphs'):
        a = [r for r in old_glm if r['length']==length and r['mode']==mode]
        b = [r for r in new_glm if r['length']==length and r['mode']==mode]
        compare(f'GLM TP2 MTP {length} tokens {mode}', [r['decode_tps'] for r in a],
                [r['decode_tps'] for r in b], 'fresh previous wheel',
                completion_hashes_equal=bool(a and b and a[0]['completion_hashes']==b[0]['completion_hashes']),
                acceptance_equal=bool(a and b and acceptance(a[0])==acceptance(b[0])))

for enabled in (False, True):
    a = read(OUT/'baseline-priority/qwen-two/generator-timings.json')
    b = read(OUT/'runtime-priority/qwen-two/generator-timings.json')
    compare(f'Qwen TP2 MTP native={enabled}',
            [r['tokens_per_second'] for r in a if r['native']==enabled],
            [r['tokens_per_second'] for r in b if r['native']==enabled], 'fresh previous wheel')

for enabled in (False, True):
    a = read(OUT/'qwen-repeat-baseline/qwen-repeat/generator-timings.json')
    b = read(OUT/'qwen-repeat-candidate/qwen-repeat/generator-timings.json')
    compare(f'Qwen TP2 MTP repeat native={enabled}',
            [r['tokens_per_second'] for r in a if r['native']==enabled],
            [r['tokens_per_second'] for r in b if r['native']==enabled],
            'fresh five-pair repeat under each wheel',
            completion_equal=bool(a and b and a[0]['completion']==b[0]['completion']))

moe = {}
moe_rounds = {}
for version, directory in [('before','baseline-priority'), ('after','runtime-followup')]:
    groups = defaultdict(list)
    round_groups = defaultdict(list)
    for row in read(OUT/directory/'nemotron-bench/summary.json'):
        if row['repeat'] >= 0 and not row['audit'] and not row['trace'] and not row['host']:
            groups[row['context'],row['batch'],row['ir']].append(row['tokens']/row['seconds'])
            round_groups[row['context'],row['batch'],row['ir']].append(len(row['rounds'])/row['seconds'])
    moe[version] = groups
    moe_rounds[version] = round_groups
for key in sorted(moe['after']):
    context,batch,ir = key
    compare(f'Nemotron context={context} batch={batch} native={ir}', moe['before'][key],
            moe['after'][key], 'fresh previous wheel')
    if ir and batch > 1:
        compare(f'Nemotron rounds/s context={context} batch={batch}', moe_rounds['before'][key],
                moe_rounds['after'][key], 'fresh previous wheel; fixed 48 MTP rounds',
                metric='rounds_per_second')

for enabled in (False, True):
    a = read(OUT/'baseline-priority/nemotron-control/summary.json')
    b = read(OUT/'runtime-final/nemotron-control/summary.json')
    compare(f'Nemotron 16K B1 long warmup native={enabled}',
            [r['tokens']/r['seconds'] for r in a if r['repeat'] >= 0 and r['ir']==enabled],
            [r['tokens']/r['seconds'] for r in b if r['repeat'] >= 0 and r['ir']==enabled],
            'fresh previous wheel; 128 warmup + 256 measured rounds')

for enabled in (False, True):
    a = [r for r in read(OUT/'nemotron-repeat-baseline/nemotron-batch-control/summary.json')
         if r['repeat'] >= 0 and r['ir']==enabled]
    b = [r for r in read(OUT/'nemotron-repeat-candidate/nemotron-batch-control/summary.json')
         if r['repeat'] >= 0 and r['ir']==enabled]
    compare(f'Nemotron short B4 extended window native={enabled}',
            [r['tokens']/r['seconds'] for r in a], [r['tokens']/r['seconds'] for r in b],
            'fresh previous wheel; 64 warmup + 128 measured rounds')
    compare(f'Nemotron short B4 extended rounds/s native={enabled}',
            [len(r['rounds'])/r['seconds'] for r in a], [len(r['rounds'])/r['seconds'] for r in b],
            'fresh previous wheel; 64 warmup + 128 measured rounds', metric='rounds_per_second')

historical = ROOT/'native-results/v1.6.5-integration/models-r1'
for directory in ('runtime-priority', 'runtime-followup', 'runtime-remaining', 'runtime-final'):
    for path in (OUT/directory).glob('*/timings.json'):
        if not (path.parent/'passed.json').exists():
            continue
        name = path.parent.name
        if name.startswith('glm'):
            continue  # Its older matrix used eager native IR; use matched graph measurements above.
        a, b = read(historical/name/'timings.json'), read(path)
        for enabled in (False, True):
            compare(f'{name} fixed decode native={enabled}',
                    [r['tokens_per_second'] for r in a if r['native']==enabled],
                    [r['tokens_per_second'] for r in b if r['native']==enabled], 'historical v1.6.5 matrix')

report = dict(comparisons=comparisons, native_moe_within_release={})
for version, groups in moe.items():
    rows = []
    for context,batch,ir in sorted(groups):
        if ir and groups[context,batch,False]:
            stock, native = stats(groups[context,batch,False]), stats(groups[context,batch,True])
            rows.append(dict(context=context,batch=batch,stock=stock,native=native,
                             gain_percent=100*(native['median']/stock['median']-1)))
    report['native_moe_within_release'][version] = rows
(OUT/'performance-summary.json').write_text(json.dumps(report, indent=2))
for row in comparisons:
    print(f"{row['workload']}: {row['before']['median']:.2f} -> {row['after']['median']:.2f} "
          f"({row['change_percent']:+.1f}%; {row['baseline_kind']})")
