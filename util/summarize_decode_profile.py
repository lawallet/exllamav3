"""Summarize host counts and union of device activities without double counting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def union_us(intervals):
    total = 0
    end = float('-inf')
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    report = json.loads((a.directory / 'dispatch.json').read_text())
    tokens = report['decode_forward_calls']
    groups = defaultdict(lambda: [0, 0., 0.])
    for row in report['calls']:
        group = row['name'].split(':')[0] + '.' + row['name'].rsplit('.', 1)[-1]
        v = groups[group]
        v[0] += row['count']
        v[1] += row['inclusive_ms']
        v[2] += row['exclusive_ms']
    result = {'decode_forward_calls': tokens, 'host_per_decode': {
        k: dict(calls=v[0] / tokens, inclusive_ms=v[1] / tokens, exclusive_ms=v[2] / tokens)
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1][2])}}
    trace_path = a.directory / 'trace.json'
    if trace_path.exists():
        events = json.loads(trace_path.read_text())['traceEvents']
        windows = [(e['ts'], e['ts'] + e['dur']) for e in events
                   if e.get('cat') == 'user_annotation' and e.get('name', '').endswith(':model.forward')]
        if windows:
            start, stop = min(x[0] for x in windows), max(x[1] for x in windows)
            devices = defaultdict(list)
            for e in events:
                if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset') or 'dur' not in e:
                    continue
                lo, hi = max(start, e['ts']), min(stop, e['ts'] + e['dur'])
                if hi > lo:
                    devices[str(e.get('args', {}).get('device', e.get('pid')))].append((lo, hi))
            result['trace'] = dict(window_ms=(stop - start) / 1000,
                device_activity_ms={k: union_us(v) / 1000 for k, v in devices.items()},
                caveat='Instrumented trace window includes inter-token gaps; missing device activity is not all Python overhead.')
        runtimes = defaultdict(lambda: [0, 0.])
        for e in events:
            if e.get('cat') == 'cuda_runtime':
                runtimes[e['name']][0] += 1
                runtimes[e['name']][1] += e.get('dur', 0) / 1000
        result['cuda_runtime'] = dict(sorted(runtimes.items(), key=lambda kv: -kv[1][1])[:20])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
