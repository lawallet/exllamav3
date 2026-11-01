"""Consolidate actual per-case outcomes from the integration suite and reruns."""
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]/'native-results/v1.6.6-integration'
cases = {}
for name in ('suite-progress.jsonl', 'resume-progress.jsonl', 'slow-progress.jsonl'):
    path = ROOT/name
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if 'collection' in row:
            for node in row['collection']:
                cases.setdefault(node, dict(result='not_run', evidence=name))
        if 'nodeid' not in row or row.get('subtest'):
            continue
        if row['outcome'] in ('failed', 'skipped') or row['when'] == 'call':
            cases[row['nodeid']] = dict(result=row['outcome'], evidence=name)

def xml_key(node):
    filename, *parts = node.split('::')
    classname = filename.removesuffix('.py').replace('/', '.')
    if len(parts) > 1:
        classname += '.' + '.'.join(parts[:-1])
    return classname, parts[-1]

lookup = {xml_key(node): node for node in cases}
path = ROOT/'native-tp-isolated/suite.xml'
if path.exists():
    for case in ET.parse(path).iter('testcase'):
        node = lookup.get((case.get('classname'), case.get('name')))
        if node:
            result = ('failed' if case.find('failure') is not None or case.find('error') is not None
                      else 'skipped' if case.find('skipped') is not None else 'passed')
            cases[node] = dict(result=result, evidence=str(path.relative_to(ROOT)))
report = dict(total=len(cases), counts=dict(Counter(r['result'] for r in cases.values())),
              failed=[n for n,r in cases.items() if r['result']=='failed'],
              not_run=[n for n,r in cases.items() if r['result']=='not_run'], cases=cases)
(ROOT/'coverage-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, indent=2))
