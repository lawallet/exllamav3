"""Summarize retained DeepSeek validation artifacts without loading a model."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize(directory):
    result = {'directory': str(directory), 'passed': (directory / 'passed.json').exists(), 'cases': []}
    for path in sorted(directory.glob('case*-1.json')):
        r = json.loads(path.read_text())
        positions = {}
        for event in r['states']:
            positions.setdefault(event['job'], []).append(event['state']['0:position'][0])
        result['cases'].append(dict(
            name=path.stem, counts=r['counts'], reuse=r['reuse'], cancelled=r['cancelled'],
            accepted=sum(j['accepted'] for j in r['jobs']),
            rejected=sum(j['rejected'] for j in r['jobs']),
            positive_rewinds=sum(n > 0 for n in r['rewinds']),
            page_crossings={j:max(ps)//256-min(ps)//256 for j,ps in positions.items()},
            position_ranges={j:[min(ps),max(ps)] for j,ps in positions.items()}))
    bench = directory / 'benchmark.json'
    if bench.exists():
        samples = json.loads(bench.read_text())
        result['benchmark'] = []
        for batch in sorted({s['batch'] for s in samples}):
            group = [s for s in samples if s['batch'] == batch]
            rates = {p:[s['tokens_per_second'] for s in group if s['plan'] == p] for p in (False,True)}
            paired = []
            for repeat in sorted({s['repeat'] for s in group}):
                pair = {s['plan']:s['tokens_per_second'] for s in group if s['repeat'] == repeat}
                paired.append(100*(pair[True]/pair[False]-1))
            result['benchmark'].append(dict(batch=batch, stock=rates[False], plan=rates[True],
                stock_median=median(rates[False]), plan_median=median(rates[True]),
                paired_percent=paired, median_paired_percent=median(paired)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps([summarize(d) for d in a.directories], indent=2))
    print(f'Summarized {len(a.directories)} runs in {a.output}')


if __name__ == '__main__':
    main()
