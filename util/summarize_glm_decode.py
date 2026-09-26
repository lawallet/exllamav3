"""Summarize all retained GLM runs and their timestamp-matched telemetry."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    runs = json.loads((args.directory / 'benchmark.json').read_text())
    samples = [json.loads(line) for line in (args.directory / 'telemetry.jsonl').read_text().splitlines()]
    modes = {}
    for native in sorted({r.get('native', False) for r in runs}):
        selected = [r for r in runs if r.get('native', False) == native]
        rates = [r['decode_tokens_per_second'] for r in selected]
        modes[str(native)] = dict(rates=rates, mean=statistics.mean(rates), median=statistics.median(rates))
    per_run = []
    for r in runs:
        devices = {}
        for sample in samples:
            if not r['start_epoch'] <= sample['time'] <= r['end_epoch'] or sample['returncode']:
                continue
            for line in sample['csv'].splitlines():
                idx, uuid, util, power, memory, limit = [v.strip() for v in line.split(',')]
                values = devices.setdefault(idx, dict(uuid=uuid, utilization=[], power=[], memory=[], limits=set()))
                for k, v in (('utilization', util), ('power', power), ('memory', memory)):
                    try:
                        values[k].append(float(v))
                    except ValueError:
                        pass
                values['limits'].add(limit)
        telemetry = {}
        for idx, d in devices.items():
            telemetry[idx] = dict(uuid=d['uuid'], power_limits=sorted(d['limits']),
                **{k: dict(mean=statistics.mean(d[k]), maximum=max(d[k]), samples=len(d[k]))
                   for k in ('utilization', 'power', 'memory') if d[k]})
        per_run.append(dict(**r, telemetry=telemetry))
    summary = dict(modes=modes, runs=per_run,
                   completion_hashes=sorted({r['completion_sha256'] for r in runs}))
    if 'True' in modes and 'False' in modes:
        summary['improvement_percent'] = 100 * (modes['True']['mean']/modes['False']['mean'] - 1)
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
