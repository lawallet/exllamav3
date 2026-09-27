"""Sequential plan gates using the existing architecture validation harnesses.

Each subprocess has its own output/log and stops the phase on failure. Never
installs an extension or writes into a protected release directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--phase', choices=['short', 'mtp', 'concurrency', 'topology', 'long', 'reuse', 'remaining'], required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--extension-dir', type=Path, default=ROOT/'native-results/block-ir/extension')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
if a.phase == 'remaining':
    for phase in ('mtp', 'concurrency', 'reuse', 'topology', 'long'):
        result = subprocess.run([sys.executable, __file__, '--phase', phase,
            '--output', str(a.output/phase), '--extension-dir', str(a.extension_dir)], cwd=ROOT)
        if result.returncode: raise SystemExit(result.returncode)
    raise SystemExit(0)
qwen = r'D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw'
glm = r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw'
jobs = []
def add(name, script, args, mapping='2,0', plan=True):
    jobs.append((name, script, args, mapping, plan))

if a.phase in ('short', 'topology'):
    mappings = ('2,0',) if a.phase == 'short' else ('2,0', '2,0,1')
    for mapping in mappings:
        suffix = mapping.replace(',', '')
        perf = ['--compare-plan', '--tokens', '1000', '--runs', '3'] if a.phase == 'short' else ['--compare-plan', '--tokens', '128', '--runs', '1']
        add('qwen-'+suffix, 'profile_qwen_decode.py', ['--model', qwen, '--prompt-file', 'doc/native_qwen_prompt.txt', '--device', 'cuda:0', '--ab']+perf, mapping)
        add('glm-'+suffix, 'profile_glm_decode.py', ['--model', glm, '--split', '90', '44', '--ab']+perf+(['--smoke'] if a.phase == 'topology' else []), mapping)
elif a.phase == 'mtp':
    for plan in (False, True):
        suffix = 'plan' if plan else 'bespoke'
        mode = ['--compare-plan'] if plan else ['--parity-only']
        add('qwen-'+suffix, 'profile_qwen_mtp.py', ['--model', qwen]+mode, '2,0,1', plan)
        add('glm-'+suffix, 'profile_glm_mtp.py', ['--model', glm]+mode, '2,0', plan)
elif a.phase == 'concurrency':
    for plan in (False, True):
        for arch in ('qwen', 'glm'):
            for suite in ('ordinary', 'mtp'):
                add(f'{arch}-{suite}-'+('plan' if plan else 'bespoke'), f'profile_{arch}_concurrency.py',
                    ['--suite', suite]+(['--compare-plan'] if plan else []), '2,0,1' if arch == 'qwen' else '2,0', plan)
elif a.phase == 'reuse':
    for arch in ('qwen', 'glm'):
        add(arch+'-reuse', f'profile_{arch}_concurrency.py',
            ['--suite', 'ordinary', '--cache-tokens', '1024', '--case', 'ordinary-page-reuse'],
            '2,0,1' if arch == 'qwen' else '2,0')
else:
    add('qwen-qsa-64k', 'check_qwen_plan_longctx.py', [], '2,0,1')
    add('qwen-qsa-32k-mtp', 'check_qwen_plan_longctx.py', ['--suite', 'mtp', '--prompt-tokens', '32768'], '2,0,1')
    for suite in ('ordinary', 'mtp'):
        add('glm-64k-'+suite, 'check_glm_longctx.py', ['--suite', suite])

records = []
for name, script, args, mapping, plan in jobs:
    command = [sys.executable, str(ROOT/'util'/script), *args, '--extension-dir', str(a.extension_dir.resolve()), '--output', str((a.output/name).resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=mapping, EXL3_NATIVE_PLAN=str(int(plan)))
    start = time.time()
    print(f'START {name}', flush=True)
    with (a.output/f'{name}.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    records.append(dict(name=name, command=command, mapping=mapping, plan=plan, returncode=result.returncode, seconds=time.time()-start))
    (a.output/'runs.json').write_text(json.dumps(records, indent=2))
    print(f'END {name}: exit={result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

if a.phase in ('mtp', 'concurrency'):
    comparisons = []
    for plan_dir in a.output.glob('*-plan'):
        bespoke_dir = plan_dir.with_name(plan_dir.name[:-5]+'-bespoke')
        for record in plan_dir.rglob('native.json'):
            reference = bespoke_dir/record.relative_to(plan_dir)
            assert reference.exists(), reference
            assert json.loads(record.read_text()) == json.loads(reference.read_text()), f'Bespoke/plan mismatch: {record}'
            comparisons.append(str(record.relative_to(a.output)))
    assert comparisons, 'No native records compared'
    (a.output/'bespoke-plan-parity.json').write_text(json.dumps(dict(exact=True, records=comparisons), indent=2))

if a.phase == 'topology':
    comparisons = {}
    def read(arch, mapping, filename):
        return json.loads((a.output/f'{arch}-{mapping}'/filename).read_text())
    for arch in ('qwen', 'glm'):
        assert read(arch, '20', 'parity.json') == read(arch, '201', 'parity.json'), f'{arch}: state parity differs by topology'
        hashes = [[r['completion_sha256'] for r in read(arch, m, 'benchmark.json')] for m in ('20', '201')]
        assert hashes[0] == hashes[1], f'{arch}: topology completion mismatch'
        profiles = [read(arch, m, 'dispatch-1.json') for m in ('20', '201')]
        for field in ('calls', 'native_calls'):
            counts = [{r['name']: r['count'] for r in p[field]} for p in profiles]
            assert counts[0] == counts[1], f'{arch}: topology {field} mismatch'
        for field in ('eligibility', 'transfers'):
            assert profiles[0][field] == profiles[1][field], f'{arch}: topology {field} mismatch'
        memory = read(arch, '201', 'device-memory.json')
        assert not memory[2]['participating'] and memory[2]['peak_allocated'] == 0
        for mapping in ('20', '201'):
            boundaries = [read(arch, mapping, f'dispatch-{mode}.json') for mode in ('bespoke', 'plan')]
            assert boundaries[0]['native_fallbacks'] == boundaries[1]['native_fallbacks'] == 0
            for field in ('calls', 'native_calls'):
                counts = [{r['name']: r['count'] for r in p[field]} for p in boundaries]
                assert counts[0] == counts[1], f'{arch}: plan increased {field}'
            assert boundaries[0]['native_plan_invocations'] == 0 and boundaries[1]['native_plan_invocations'] > 0
        comparisons[arch] = dict(exact_state=True, exact_completions=True,
            same_call_counts=True, same_eligibility=True, same_logical_transfers=True,
            same_bespoke_plan_boundary_counts=True, zero_decode_fallbacks=True,
            unused_device_peak_allocated=memory[2]['peak_allocated'])
    (a.output/'topology-comparison.json').write_text(json.dumps(comparisons, indent=2))
