"""Render measured tables from retained JSON; no timing or model execution."""
import json
from pathlib import Path
import statistics
from collections import defaultdict

BASE=Path(__file__).resolve().parents[1]/'native-results/deepseek-v41/omp-policy'


def main():
    summary=json.loads((BASE/'summary.json').read_text())
    lines=['# OpenMP policy measured tables','',
           'Decode figures exclude prefill. Concurrent aggregate rates include the complete request wall time.',
           'The CPU column uses whole-request main-process CPU time; TP children are excluded there. Separate telemetry summaries include process-tree CPU consumption.',
           'SD is sample standard deviation; variance is SD squared. Warmup and monitored requests are excluded here.','']
    for phase in ('steady','long','masked','concurrent'):
        lines += ['## '+phase,'',
                  '| Process | Context | Native / MTP | ms | n | Decode median (range), tok/s | SD | Variance | Prefill median, s | CPU cores | Aggregate request tok/s |',
                  '|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
        for folder,entry in summary.items():
            for key,group in sorted(entry['groups'].items()):
                if not key.startswith(phase+'/'):continue
                parts=key.split('/');s=group['tps'];prefill=group['prefill_s']['median']
                cpu=group.get('cpu_core_equivalents',{}).get('median')
                aggregate=group.get('aggregate_output_tps',{}).get('median')
                sd=f"{s['stdev']:.3f}" if s['stdev'] is not None else '—'
                variance=f"{s['variance']:.4f}" if s['variance'] is not None else '—'
                cpu_text=f'{cpu:.3f}' if cpu is not None else 'separate telemetry'
                aggregate_text=f'{aggregate:.3f}' if aggregate is not None else '—'
                lines.append(f"| {folder} | {parts[1]}/{parts[2]} | {parts[3]}/{parts[4]} | {parts[5][2:]} | {s['n']} | "
                    f"{s['median']:.3f} ({s['minimum']:.3f}–{s['maximum']:.3f}) | {sd} | {variance} | "
                    f"{prefill:.4f} | {cpu_text} | {aggregate_text} |")
        lines.append('')
    lines+=['## CPU-only controls','', '| Process | Workload | Submitters | ms | n | Wall median (range), s | SD | Exact |',
            '|---|---|---:|---:|---:|---|---:|---|']
    for folder in sorted(BASE.glob('*-cpu')):
        path=folder/'results.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        groups=defaultdict(list)
        for row in data:groups[(row['kind'],row['submitters'],row['blocktime'])].append(row)
        for (kind,submitters,ms),rows in sorted(groups.items()):
            values=[r['wall_s'] for r in rows]
            lines.append(f"| {folder.name} | {kind} | {submitters} | {ms} | {len(values)} | "
                f"{statistics.median(values):.5f} ({min(values):.5f}–{max(values):.5f}) | "
                f"{statistics.stdev(values) if len(values)>1 else 0:.5f} | {all(r['exact'] for r in rows)} |")
    (BASE/'measured-tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
