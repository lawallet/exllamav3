"""Summarize retained direct-backend measurements without discarding samples."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
summary = []
for path in sorted(a.directory.rglob('bespoke-plan-benchmark.json')):
    records = json.loads(path.read_text())
    groups = defaultdict(list)
    for r in records:
        groups[(r.get('mtp'), r.get('jobs', 1))].append(r)
    telemetry_path = path.parent/'telemetry.jsonl'
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()] if telemetry_path.exists() else []
    for (mtp, jobs), group in groups.items():
        metric = 'aggregate_tps' if 'aggregate_tps' in group[0] else 'decode_tokens_per_second'
        samples = {name: [r[metric] for r in group if r['backend'] == name] for name in ('bespoke', 'plan')}
        if not all(samples.values()): continue
        rates = {k: median(v) for k, v in samples.items()}
        observational = defaultdict(lambda: defaultdict(list))
        for r in group:
            if 'start_epoch' not in r: continue
            for sample in telemetry:
                if not r['start_epoch'] <= sample['time'] <= r['end_epoch']: continue
                for row in csv.reader(io.StringIO(sample['csv']), skipinitialspace=True):
                    key = f'{r["backend"]}/physical-gpu-{row[0]}'
                    observational[key]['utilization_percent'].append(float(row[2]))
                    observational[key]['power_watts'].append(float(row[3]))
        summary.append(dict(source=str(path.relative_to(a.directory)), mtp=mtp, jobs=jobs,
            metric=metric, samples=samples, medians=rates, relative_change=rates['plan']/rates['bespoke']-1,
            telemetry_medians={k: {m: median(v) for m, v in values.items()} for k, values in observational.items()}))
(a.directory/'performance-summary.json').write_text(json.dumps(summary, indent=2))
lines = ['| Case | Metric | Bespoke median | Plan median | Change |', '| --- | --- | ---: | ---: | ---: |']
for s in summary:
    lines.append(f'| {s["source"]} (B{s["jobs"]}) | {s["metric"]} | {s["medians"]["bespoke"]:.2f} | {s["medians"]["plan"]:.2f} | {100*s["relative_change"]:+.2f}% |')
(a.directory/'performance-summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
