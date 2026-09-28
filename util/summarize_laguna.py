"""Write measured acceptance results into the Laguna engineering report."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'native-results/laguna/campaign-r2'
def read(name): return json.loads((V/name).read_text())
runs=read('runs.json')
assert len(runs)==11 and all(r['returncode']==0 for r in runs)
assert read('passed.json')['all_selected_gates_passed']
report=ROOT/'doc/laguna_ir_report.md'
s=report.read_text()
marker='Validation results and performance are finalized below only after completion.'
assert marker in s
lines=['### Completed acceptance gates','',
 '135 shared/Laguna guard tests passed. Actual-weight sites: 48 blocks times six',
 'shapes = 288 exact cases, including boundary dtype/layout/device/alias rejection',
 'without mutation. Existing generic native implementation needed no new primitive.',
 '',
 'Short decode: B1/B2/B3/B4 Q1, B1Q5, B2Q6, 32 steps each; all exact logits,',
 'tokens, normalized sublayer inputs, attention outputs, route IDs/weights, fused',
 'expert results, combined MLP outputs, block/transfer/export boundaries and',
 'logically valid global KV/SWA state. Each supported shape constructs48 plans',
 'and invokes1536 begin/middle/finish stages, plus1504 stock router calls.',
 'At most8 rows also make1504 run_bszN calls; B2Q6 retains the stock larger-row',
 'expert path with exact route/combined-output/state checks. No unexpected IR',
 'fallback. Q7 uses stock norm/residual stages. One unsupported norm',
 'configuration falls back for one block while the other47 remain native.',
 'Identity rebinding reconstructs a plan. All seven generator cases passed: B1..4,',
 'staggered/replacement, live cancellation/replacement and seeded sampling.',
 'Context480 plus48 generated tokens crosses both a paged-KV boundary and the',
 '512-token sliding window. Constrained cache1536 establishes real reuse:', '']
for i in (4,5):
    r=read(f'reuse/case{i}-1.json')
    lines.append(f"- Case{i}: reuse={r['reuse']}; cancellation={bool(r['cancelled'])}.")
lines+=['','Reload clears plans/bindings and reconstructs exact outputs/state. The isolated',
 'harness clears the stock process-wide workspace cache to preserve autosplit',
 'placement, as in the earlier Gemma/Devstral oracle; no loader behavior changed.',
 'Visibility2,0 matches2,0,1 exactly in placement, output/state and counters.',
 'Unused logicalGPU2 allocated/reserved/peak memory are all zero.', '']
for name in ('short-201','short-20','remapped'):
    from collections import Counter
    placement=Counter(r['device'] for r in read(name+'/placement.json') if r['key'].startswith('model.layers.'))
    lines.append(f'- {name}: {dict(placement)} blocks. Exact stock/plan parity within this placement.')
lines+=['','The remapped0,2 run uses budgets16/70 and16 steps per shape. No cross-GPU-family',
 'stock bit-equality claim. Long context8192 passes B1Q1 and B1Q5 for16 steps,',
 'ending at8208/8272 positions with768 calls per native stage; hashes include live',
 'sliding KV and valid paged KV. Maximum1048576 context is not certified.',
 '', '### Warmed performance', '',
 'Five alternating warmed pairs per B, context512,256 timed steps,8 warmup steps.',
 'No observers/hashes inside timing. B4 t/s is aggregate. Identical completion',
 'tokens required between paired runs. All individual samples:', '',
 '| Batch | Pair | Stock t/s | Plan t/s | Change |',
 '|---|---|---|---|---|']
samples=read('performance/samples.json')
for b in (1,4):
    for i in range(5):
        pair={r['plan']:r['tps'] for r in samples if r['batch']==b and r['repeat']==i}
        lines.append(f'| {b} | {i+1} | {pair[False]:.6f} | {pair[True]:.6f} | {100*(pair[True]/pair[False]-1):+.3f}% |')
lines.append('')
for r in read('performance/performance.json'):
    lines.append(f"B{r['batch']}: stock median {r['median_tps']['False']:.6f} t/s; plan median {r['median_tps']['True']:.6f} t/s; median paired change {100*r['median_change']:+.3f}%.")
repeat=V.parent/'performance-repeat'
assert json.loads((repeat/'run.json').read_text())['returncode']==1
assert 'AssertionError: Repeatable regression requires investigation' in (repeat/'run.log').read_text()
lines+=['','### Performance instability investigation','',
 'Initial unrestricted samples varied sharply for both backends (including large',
 'sign reversals), so their apparent gains are not a reliable performance claim.',
 'GPU inspection found no active thermal slowdown; the exact source of variation',
 'was not established. The host has a56-core AMD EPYC7663. Repeated ten longer',
 'pairs per B with512 timed steps, holding only the isolated benchmark process',
 'and its child to CPU affinity0xff (verified inherited), identical for both paths.',
 'No production processes, GPU clocks, power policies or dependencies were altered.',
 'Read-only one-second GPU telemetry is retained in performance-repeat/gpu.csv.',
 'During decode the secondary4090 D was observed in P5 with810MHz memory;',
 'varying power states with only two layers assigned there are a plausible',
 'contributor, not a proven attribution of every outlier.',
 'This is a controlled-host comparison, not a guarantee of unconstrained throughput.',
 'The B4 median of-3.12% triggered the unchanged harness investigation assertion.',
 'That failed diagnostic is preserved; it is not labeled a passed performance gate.',
 '', '| Batch | Pair | Stock t/s | Plan t/s | Change |','|---|---|---|---|---|']
samples=json.loads((repeat/'results/samples.json').read_text())
for b in (1,4):
    for i in range(10):
        pair={r['plan']:r['tps'] for r in samples if r['batch']==b and r['repeat']==i}
        lines.append(f'| {b} | {i+1} | {pair[False]:.6f} | {pair[True]:.6f} | {100*(pair[True]/pair[False]-1):+.3f}% |')
lines.append('')
for r in json.loads((repeat/'results/performance.json').read_text()):
    lines.append(f"Controlled B{r['batch']}: stock median {r['median_tps']['False']:.6f} t/s; plan median {r['median_tps']['True']:.6f} t/s; median paired change {100*r['median_change']:+.3f}%.")
control=V.parent/'performance-control'
cruns=json.loads((control/'runs.json').read_text())
assert len(cruns)==2 and all(r['returncode']==0 for r in cruns)
lines+=['','### Single-GPU timing control','',
 'To remove the lightly loaded secondary GPU, ran an additional exact short',
 'stock/plan gate (all six shapes plus fallback, four steps) with all48 blocks',
 'on the96GB GPU, budget90, unchanged visibility2,0,1 and CPU affinity0xff.',
 'Then five warmed alternating pairs per B, context512,256 timed steps.',
 'This separate placement is a diagnostic control, not a replacement for the',
 'required two-GPU correctness/topology/performance evidence.', '',
 '| Batch | Pair | Stock t/s | Plan t/s | Change |','|---|---|---|---|---|']
samples=json.loads((control/'performance/samples.json').read_text())
for b in (1,4):
    for i in range(5):
        pair={r['plan']:r['tps'] for r in samples if r['batch']==b and r['repeat']==i}
        lines.append(f'| {b} | {i+1} | {pair[False]:.6f} | {pair[True]:.6f} | {100*(pair[True]/pair[False]-1):+.3f}% |')
lines.append('')
for r in json.loads((control/'performance/performance.json').read_text()):
    lines.append(f"Single-GPU B{r['batch']}: stock median {r['median_tps']['False']:.6f} t/s; plan median {r['median_tps']['True']:.6f} t/s; median paired change {100*r['median_change']:+.3f}%.")
lines+=['','### Compatibility and cost','',
 'Gemma v2 and DeepSeek v1 representative short smokes passed all six supported',
 'shapes at four steps, with exact state/output and nonzero native calls.',
 'No Qwen/GLM or historical MTP campaign was run: no affected shared execution.',
 'Checkpoint/package/extracted-wheel identity and final hashes are recorded by',
 'the release README, checkpoint.json, checkout-gates.json and package-proof.json.',
 'Promotion is conditional on the three extracted-wheel jobs passing.', '']
paths=[ROOT/'exllamav3/modules/native_laguna.py',ROOT/'tests/test_native_laguna_guard.py']
paths+=sorted((ROOT/'util').glob('*laguna*.py'))
lines+=['| File | Physical LOC | Nonblank LOC |','|---|---|---|']
for f in paths:
    ls=f.read_text().splitlines()
    lines.append(f'| {f.relative_to(ROOT).as_posix()} | {len(ls)} | {sum(bool(x.strip()) for x in ls)} |')
lines+=['','Guard module includes registration and the one-line call reusing the existing',
 'emitter. Architecture registration adds two executable lines outside that module.',
 'Validation and release helpers are separately listed above and adapted from the',
 'established oracles. Elapsed effort is recorded at finalization; token/credit',
 'usage is not exposed. No unrelated stock bug was identified.', '']
report.write_text(s.replace(marker,'\n'.join(lines)))
print('Wrote measured acceptance and performance report')
