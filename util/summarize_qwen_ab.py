"""Join uninstrumented A/B generations with external nvidia-smi samples."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--telemetry', type=Path)
    parser.add_argument('--physical-gpu', type=int, default=2)
    args = parser.parse_args()
    runs = json.loads((args.directory / 'benchmark.json').read_text())
    samples = []
    if args.telemetry:
        with args.telemetry.open() as file:
            for row in csv.DictReader(file, skipinitialspace=True):
                try:
                    if int(row['index']) != args.physical_gpu:
                        continue
                    samples.append((datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp(),
                                    float(row['utilization.gpu [%]'].split()[0]),
                                    float(row['power.draw [W]'].split()[0])))
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue  # A live telemetry file may end with a partial row.
    summary = {}
    for enabled in (False, True):
        group = [r for r in runs if r['native'] == enabled]
        if not group:
            continue
        matching = [s for s in samples if any(r['start_epoch'] <= s[0] <= r['end_epoch'] for r in group)]
        summary['native' if enabled else 'stock'] = dict(
            runs=len(group), tokens_per_second=[r['decode_tokens_per_second'] for r in group],
            mean_tokens_per_second=statistics.mean(r['decode_tokens_per_second'] for r in group),
            mean_total_seconds=statistics.mean(r['total_seconds'] for r in group),
            mean_prefill_seconds=statistics.mean(r['time_prefill'] for r in group),
            telemetry_samples=len(matching),
            mean_gpu_util_percent=statistics.mean(s[1] for s in matching) if matching else None,
            mean_gpu_power_watts=statistics.mean(s[2] for s in matching) if matching else None)
        if 'proposed_draft_tokens' in group[0]:
            metrics = summary['native' if enabled else 'stock']
            for field in ('proposed_draft_tokens', 'accepted_draft_tokens', 'rejected_draft_tokens', 'acceptance_rate'):
                metrics[field] = [r[field] for r in group]
            metrics['native_calls'] = [r['calls']['native'] for r in group]
            metrics['fallback_calls'] = [r['calls']['fallback'] for r in group]
            if 'effective_output_tokens_per_second' in group[0]:
                metrics['mean_effective_output_tokens_per_second'] = statistics.mean(
                    r['effective_output_tokens_per_second'] for r in group)
    if 'stock' in summary and 'native' in summary:
        summary['improvement_percent'] = 100 * (summary['native']['mean_tokens_per_second'] /
                                                summary['stock']['mean_tokens_per_second'] - 1)
    summary['all_completion_hashes_match'] = len({r['completion_sha256'] for r in runs}) == 1
    summary['prompt_tokens'] = sorted({r['prompt_tokens'] for r in runs})
    summary['generated_tokens'] = sorted({r['new_tokens'] for r in runs})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
