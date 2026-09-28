# Laguna-S-2.1 native IR engineering report

## Architecture and lowering

See [the operation map](laguna_ir_mapping.md). Inspection used the actual loaded
canonical TextGen checkpoint (Tabby junction accesses the same files), pinned
Python3.13.12/Torch2.9.0+cu128/CUDA12.8, and the protected Devstral native binary.
No dependency changes, production installation or checkpoint duplication.

Taxonomy: one global-attention/dense block (layer0), 11 global-attention/MoE
blocks, 36 sliding-attention/MoE blocks. Hidden3072, BF16 norm weights, FP32
residual, FP16 normalized inputs; global Q48 / sliding Q72, KV8, head128.
Sliding window512, Q/K norms and per-head softplus attention gates are Python
attention-owned. SWAState is real persistent recurrent KV state; global KV is
paged. Exported states retain the stock FP16 clamped residual semantics. Final
norm/head and cache/scheduler/attention execution remain unchanged.

47 sparse layers use 256 top-10 routed SwiGLU experts (width1024) and one shared
SwiGLU expert (width1024). Shared intermediate/output is FP32; routed
intermediates are FP16 scaled down by128 to avoid overflow. Effective routing
scale320 compensates for that downscale and implements the configured2.5.
The loaded router matrix is FP16 [3072,256], correction bias FP16 [256], output
indices int64 and weights FP16 [rows,10]. Deterministic projection uses the
existing cached transposed gate and int8 slices/scales. Sigmoid affinities plus
correction choose experts; selected *unbiased* affinities are normalized/scaled.
No grouping, router-logit bias, hash table or per-expert scale is bound.

## IR decision

Uses existing v2 RMS_NORM, RES_ADD_NORM, RES_ADD via the existing eight-line
`native_plan.devstral` emitter. The factory name is historical: its emitted
instructions contain only generic norm/residual operands. No new emitter body,
value roles, integer operands, opcodes, capability flags, C++ or kernels.
V1's HC/GR site coupling does not fit a single residual. Its softmax ROUTE also
cannot express Laguna's sigmoid/bias/scaling semantics. NORM_ADD is postnorm
addition, not the residual-in norm required here.

Outcome B: retain stock routing/MoE inside the Python sublayer boundary. For
rows<=8, inspection confirms BC_BlockSparseMLP.run_bszN already executes fused
experts and the shared-expert combine (`bc_sh_exp=True` on every sparse layer).
There is no Python per-expert loop or sort/regroup in this decode tier. A major
remaining routing orchestration bottleneck has not been demonstrated, so the
prompt's condition for a configurable routing expansion is not met. No routing
design/API was implemented. Native routing kernels still execute, but routing
is not an IR instruction. Do not attribute any measured gain to newly native
routing; there is none. A routing-isolation benchmark does not apply.

B2Q6 has12 rows and exceeds the native expert decode tier (MAX_BSZN=8).
It retains stock grouped/fused/reconstruct dispatch, including token regrouping
and separate shared-expert handling. V2 still handles its norm/residual stages.
No claim is made that all supported multi-query expert orchestration is fused.
The ordinary B1/B4 throughput experiment uses the already-fused tier.

## Safety and ownership

Registration is architecture-local. Guard accepts exact Attention/SlidingAttention
and GatedMLP/BlockSparseMLP classes with the required FP32 boundary contracts,
plain residual flow and co-located BF16/FP16 norms. B1..4, Q1..6, B*Q<=12;
FP32 CUDA contiguous residual. Derives placement from loaded tensors without
enumerating visible GPUs. Rejects TP, LoRA/backend, capture/conversion,
prefill/loader modes, CPU offload/split, latent/alternate-residual MoE, postnorms
and incompatible layouts before native mutation. Native failures propagate;
there is no stock retry after a mutated stage. Norm identity/data pointer/config
changes rebuild plans. Load/unload clears Laguna-bound plans and scratch.
The existing residual capability gate protects against old extensions.

## Validation record

Evidence root: `native-results/laguna`. Initial architecture inspection is
`inspection.json`; expanded loaded routing contracts are in
`campaign/inspection.json`. Product identity is frozen in `validated-source.json`.
All native source and the tested binary are unchanged from the Devstral release:
SHA256 `b89a859f85765405f77e517f6faed02456cd23a46e48226258ba27c5dd1d6fd3`.

The initial guard runner mistakenly set EXL3_NATIVE_PLAN=1 for existing mocked
bespoke unit fixtures, causing fixture constructor/selection failures. Corrected
only the runner to use0 (tests toggle the backend themselves). Preserved initial
failure logs in `campaign`; corrected campaign is `campaign-r2`. No product
semantics or assertions were weakened.

### Completed acceptance gates

135 shared/Laguna guard tests passed. Actual-weight sites: 48 blocks times six
shapes = 288 exact cases, including boundary dtype/layout/device/alias rejection
without mutation. Existing generic native implementation needed no new primitive.

Short decode: B1/B2/B3/B4 Q1, B1Q5, B2Q6, 32 steps each; all exact logits,
tokens, normalized sublayer inputs, attention outputs, route IDs/weights, fused
expert results, combined MLP outputs, block/transfer/export boundaries and
logically valid global KV/SWA state. Each supported shape constructs48 plans
and invokes1536 begin/middle/finish stages, plus1504 stock router calls.
Shapes with at most8 rows also make1504 run_bszN calls and compare its direct
expert output. B2Q6 uses the stock larger-row expert path; its route IDs/weights,
combined MLP output and all block/state boundaries compare exactly. No unexpected
IR fallback. Q7 uses stock residual/norm stages. One unsupported norm
configuration falls back for one block while the other47 remain native.
Identity rebinding reconstructs a plan. All seven generator cases passed: B1..4,
staggered/replacement, live cancellation/replacement and seeded sampling.
Context480 plus48 generated tokens crosses both a paged-KV boundary and the
512-token sliding window. Constrained cache1536 establishes real reuse:

- Case4: reuse={'pages': 2, 'slots': 1}; cancellation=False.
- Case5: reuse={'pages': 2, 'slots': 1}; cancellation=True.

Reload clears plans/bindings and reconstructs exact outputs/state. The isolated
harness clears the stock process-wide workspace cache to preserve autosplit
placement, as in the earlier Gemma/Devstral oracle; no loader behavior changed.
Visibility2,0 matches2,0,1 exactly in placement, output/state and counters.
Unused logicalGPU2 allocated/reserved/peak memory are all zero.

- short-201: {'cuda:0': 46, 'cuda:1': 2} blocks. Exact stock/plan parity within this placement.
- short-20: {'cuda:0': 46, 'cuda:1': 2} blocks. Exact stock/plan parity within this placement.
- remapped: {'cuda:0': 13, 'cuda:1': 35} blocks. Exact stock/plan parity within this placement.

The remapped0,2 run uses budgets16/70 and16 steps per shape. No cross-GPU-family
stock bit-equality claim. Long context8192 passes B1Q1 and B1Q5 for16 steps,
ending at8208/8272 positions with768 calls per native stage; hashes include live
sliding KV and valid paged KV. Maximum1048576 context is not certified.

### Warmed performance

Five alternating warmed pairs per B, context512,256 timed steps,8 warmup steps.
No observers/hashes inside timing. B4 t/s is aggregate. Identical completion
tokens required between paired runs. All individual samples:

| Batch | Pair | Stock t/s | Plan t/s | Change |
|---|---|---|---|---|
| 1 | 1 | 62.583146 | 48.296704 | -22.828% |
| 1 | 2 | 38.416039 | 34.481863 | -10.241% |
| 1 | 3 | 36.469527 | 36.795735 | +0.894% |
| 1 | 4 | 35.083619 | 36.105434 | +2.913% |
| 1 | 5 | 37.914274 | 60.902932 | +60.633% |
| 4 | 1 | 125.069295 | 166.598750 | +33.205% |
| 4 | 2 | 108.332548 | 176.369705 | +62.804% |
| 4 | 3 | 189.836090 | 115.295924 | -39.266% |
| 4 | 4 | 111.606958 | 179.568560 | +60.894% |
| 4 | 5 | 187.626168 | 115.378404 | -38.506% |

B1: stock median 37.914274 t/s; plan median 36.795735 t/s; median paired change +0.894%.
B4: stock median 125.069295 t/s; plan median 166.598750 t/s; median paired change +33.205%.

### Performance instability investigation

Initial unrestricted samples varied sharply for both backends (including large
sign reversals), so their apparent gains are not a reliable performance claim.
GPU inspection found no active thermal slowdown; the exact source of variation
was not established. The host has a56-core AMD EPYC7663. Repeated ten longer
pairs per B with512 timed steps, holding only the isolated benchmark process
and its child to CPU affinity0xff (verified inherited), identical for both paths.
No production processes, GPU clocks, power policies or dependencies were altered.
Read-only one-second GPU telemetry is retained in performance-repeat/gpu.csv.
During decode the secondary4090 D was observed in P5 with810MHz memory;
varying power states with only two layers assigned there are a plausible
contributor, not a proven attribution of every outlier.
This is a controlled-host comparison, not a guarantee of unconstrained throughput.
The B4 median of-3.12% triggered the unchanged harness investigation assertion.
That failed diagnostic is preserved; it is not labeled a passed performance gate.

| Batch | Pair | Stock t/s | Plan t/s | Change |
|---|---|---|---|---|
| 1 | 1 | 57.647436 | 37.063636 | -35.706% |
| 1 | 2 | 36.015684 | 36.947018 | +2.586% |
| 1 | 3 | 36.209239 | 36.079629 | -0.358% |
| 1 | 4 | 52.951752 | 37.818645 | -28.579% |
| 1 | 5 | 35.277511 | 36.061831 | +2.223% |
| 1 | 6 | 36.504704 | 36.134681 | -1.014% |
| 1 | 7 | 36.003526 | 36.262654 | +0.720% |
| 1 | 8 | 36.011088 | 35.606346 | -1.124% |
| 1 | 9 | 48.480842 | 37.828811 | -21.972% |
| 1 | 10 | 52.069307 | 36.098613 | -30.672% |
| 4 | 1 | 132.895874 | 139.921011 | +5.286% |
| 4 | 2 | 146.172195 | 137.824603 | -5.711% |
| 4 | 3 | 133.714508 | 103.819955 | -22.357% |
| 4 | 4 | 138.068821 | 134.028466 | -2.926% |
| 4 | 5 | 133.439358 | 129.026203 | -3.307% |
| 4 | 6 | 144.019152 | 127.066592 | -11.771% |
| 4 | 7 | 144.712165 | 129.486990 | -10.521% |
| 4 | 8 | 136.133088 | 135.286138 | -0.622% |
| 4 | 9 | 137.726131 | 138.536923 | +0.589% |
| 4 | 10 | 137.524954 | 140.699624 | +2.308% |

Controlled B1: stock median 36.356971 t/s; plan median 36.198668 t/s; median paired change -1.069%.
Controlled B4: stock median 137.625542 t/s; plan median 134.657302 t/s; median paired change -3.117%.

### Single-GPU timing control

To remove the lightly loaded secondary GPU, ran an additional exact short
stock/plan gate (all six shapes plus fallback, four steps) with all48 blocks
on the96GB GPU, budget90, unchanged visibility2,0,1 and CPU affinity0xff.
Then five warmed alternating pairs per B, context512,256 timed steps.
This separate placement is a diagnostic control, not a replacement for the
required two-GPU correctness/topology/performance evidence.

| Batch | Pair | Stock t/s | Plan t/s | Change |
|---|---|---|---|---|
| 1 | 1 | 70.876908 | 70.357866 | -0.732% |
| 1 | 2 | 70.725805 | 70.216916 | -0.720% |
| 1 | 3 | 71.103446 | 70.542422 | -0.789% |
| 1 | 4 | 71.225410 | 70.889805 | -0.471% |
| 1 | 5 | 70.708358 | 70.718287 | +0.014% |
| 4 | 1 | 220.710799 | 221.973283 | +0.572% |
| 4 | 2 | 223.497111 | 226.344007 | +1.274% |
| 4 | 3 | 222.914442 | 140.008995 | -37.192% |
| 4 | 4 | 223.276628 | 219.428556 | -1.723% |
| 4 | 5 | 222.433669 | 225.210335 | +1.248% |

Single-GPU B1: stock median 70.876908 t/s; plan median 70.542422 t/s; median paired change -0.720%.
Single-GPU B4: stock median 222.914442 t/s; plan median 221.973283 t/s; median paired change +0.572%.

### Compatibility and cost

Gemma v2 and DeepSeek v1 representative short smokes passed all six supported
shapes at four steps, with exact state/output and nonzero native calls.
No Qwen/GLM or historical MTP campaign was run: no affected shared execution.
Checkpoint/package/extracted-wheel identity and final hashes are recorded by
the release README, checkpoint.json, checkout-gates.json and package-proof.json.
Promotion is conditional on the three extracted-wheel jobs passing.

| File | Physical LOC | Nonblank LOC |
|---|---|---|
| exllamav3/modules/native_laguna.py | 78 | 73 |
| tests/test_native_laguna_guard.py | 144 | 129 |
| util/check_laguna_sites.py | 82 | 76 |
| util/control_laguna_performance.py | 25 | 25 |
| util/inspect_laguna_ir.py | 114 | 105 |
| util/laguna_jobs.py | 64 | 63 |
| util/package_laguna_plan.py | 71 | 70 |
| util/release_laguna.py | 145 | 140 |
| util/repeat_laguna_performance.py | 36 | 36 |
| util/summarize_laguna.py | 125 | 125 |
| util/validate_laguna.py | 339 | 326 |
| util/verify_laguna_wheel.py | 31 | 31 |

Guard module includes registration and the one-line call reusing the existing
emitter. Architecture registration adds two executable lines outside that module.
Validation and release helpers are separately listed above and adapted from the
established oracles. Elapsed effort is recorded at finalization; token/credit
usage is not exposed. No unrelated stock bug was identified.


## Scope and economics

Generic C++/CUDA/opcodes/ABI/value model: zero changes. Generic Python emitter
and common TransformerBlock forward: zero changes. TransformerBlock only adds
two Laguna lifecycle cache-clear statements. Existing validation backend helper
adds Laguna's cache name. Dedicated guard derives from the established Devstral
structural contract with sliding attention and sparse Python boundary eligibility.
Registration adds two executable architecture lines. Emitter body: zero new
lines, reuses eight existing lines. Validation utilities are counted separately
in final metadata/report; they do not affect timed execution.

Scope-proportional regression policy: shared native guard suite plus representative
Gemma v2 and DeepSeek v1 short smokes. No shared generic executor/dispatch/kernel
changes, therefore no historical Qwen/GLM or MTP campaign. Extracted wheel requires
Laguna actual-weight sites, short decode/reload and same-placement topology;
it reuses the same extension and verifies every packaged product module origin.

Unlike Gemma's initial contract work or Devstral's two-op addition, Laguna adds
no generic IR capability. Like DeepSeek it is a small architecture integration,
but its performance must be judged from the measured pairs rather than inferred
from being MoE. Validation effort is dominated by loading, exact state hashing,
concurrency, topology, long context and warmed performance samples.

## Limitations and future work

The two-GPU B4 regression remains a measured limitation: -3.12% median paired
change in the extended46/2 placement run. Its threshold assertion failed and
is preserved as an investigated diagnostic, not relabeled as a passed timing
test. The separate48/0 control passed exact parity and its timing threshold:
B1 -0.72%, B4 +0.57% median paired. B1 control samples are tightly grouped;
B4 still includes one large negative outlier. Thus removing the secondary GPU
does not fully establish the cause of all variability. No universal throughput
gain or absence of tail-latency regressions is certified. Further host/driver
performance tuning is outside this IR integration; all samples remain visible.
Release qualification rests on completed correctness gates, recorded timings
and this explicit investigation/disclosure, not a claim that every diagnostic
timing run passed. The runtime remains opt-in through EXL3_NATIVE_PLAN=1.

Explicit answers: Laguna's norm/residual flow fits existing v2 unchanged.
V1 ROUTE does not fit its router exactly. No configurable routing or value
expansion was justified, so reuse by DeepSeek of a *new* router is not claimed.
There is no newly native routing/MoE speedup to report: those native kernels
were already behind the stock Python boundary. The measured result does not
strengthen the claim that MoE models automatically benefit from this IR; it
also does not test the potential of moving a genuinely expensive Python router
into IR. Laguna's decode expert orchestration is already substantially fused.

No target-native MTP/default drafter (text component only). Separate DFlashLaguna
code exists for an external compatible draft checkpoint; none is supplied by
this target and external speculation is unverified. Prefill remains stock.
TP/LoRA/capture/CPU offload are outside the native eligibility tested here.
Long context certification is limited to the recorded8192 test, not advertised
1048576. Future configurable routing could reuse score transforms/bias/scale
with DeepSeek, but needs measured justification and dedicated index/weight
bindings; no architecture branches or speculative abstraction were added.
