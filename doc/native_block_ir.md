# Native block plan engineering report

## Source inventory (before implementation)

Reference: `glm53-native-longctx-v1` (8264969), with the Qwen implementation
retained from the protected Qwen checkpoint. Release directories are immutable.

Qwen `native_qwen.py` resolves 13 tensors: two fn/up/weight triples, four
router projection/quantization tensors, and logits/selection/routing outputs.
`BC_QwenBlock` owns a shared native MoE handle, two epsilon values and R.
Each site allocates float dots [R,fn_rows+1,H], float post [R,H], and half
mixed [R,D]. R>1 allocates private router outputs; R=1 reuses router workspaces.
Begin validates float contiguous residuals, flattens B*Q, calls gr_mix and
returns [B,Q,D]. Python attention runs next. Finish applies attention gates,
mixes MLP input, calls routing_std, native MoE run_bszN and hc_apply.
Two pybind crossings; no cache ownership. Ordinary B1..4 Q1, target MTP B1
Q1/Q5 and attached draft B1..3 Q1 are guarded in Python. Multi-job target
verification remains stock. Device placement is already derived from modules.

GLM `native_glm.py` / `native_glm_mtp.py` resolve two fn/base/scale/norm bundles.
`BC_GlmBlock` owns norm/hc epsilon, Sinkhorn iterations, B, Q and R. Scratch:
float partials [R,hc_mix_num_chunks(R,16384),25], post [R,4], combine [R,4,4],
half collapsed and normalized [R,4096], cached normalized [B,Q,4096] view.
Begin: hc_mix, rms_norm. Python attention. Middle: hc_apply, hc_mix, rms_norm.
Python dense/MoE MLP. Finish: hc_apply. Three pybind crossings. Ordinary B1..4
Q1; attached three-draft target B1..3 Q1/Q4. Draft execution stays stock.
Current guards demand 45 layers, a 35/10 Pro6000/4090D split, exact hyperconnection
and normalization configuration, valid recurrent slots and export protocol.
Both GLM guards incorrectly require exactly two globally visible devices.

Both architectures reject incompatible dtype/layout, TP, LoRA, capture,
conversion/export modes and prefill before entering native orchestration.
Qwen additionally checks all MoE configuration assumptions. GLM validates
weight identity/configuration on cached use; transformer load/unload clears
native caches. Attention, recurrent/conv state, KV pages, QSA, MLA, KDA,
DSA/indexer/pool, generator membership and speculative repair remain Python
responsibilities. These are not IR operands or generic scratch.

## Minimal resolved representation

Six primitives: GR_MIX, HC_MIX, APPLY, RMS_NORM, ROUTE, MOE. APPLY is the
existing hc_apply kernel with optional combine matrix. Views and scratch are
specialization metadata, not instructions. ROUTE and MOE reuse the existing
Qwen implementation/handle; no new math or fusion.

Python emits integer opcode/site pairs in three stage slots. Native construction
validates and resolves these to member-function pointers plus site references.
No strings, dictionaries, Python primitive callbacks or weight lookups occur
inside a stage. Qwen uses begin/finish; GLM uses begin/middle/finish, preserving
the existing transformer call sites. Each site binds tensors and scalar kernel
parameters once. The plan retains tensor and MoE ownership for its lifetime.

Specializations are keyed by device, B, Q and dtype (R is derived). Kernels
receive flattened row-local views; Python boundaries retain B,Q. Scratch holds
only temporary activations/gates/router outputs. Existing serialized-stream
execution assumptions remain; simultaneous independent forwards are unsupported.
Load/unload invalidates plans; tensor/config changes must invalidate bindings.

Architecture guards remain authoritative. Unsupported modes return stock
before begin. Native contract violations raise before the stage mutates the
residual; they must never be caught to retry stock after partial execution.
The existing bespoke wrappers remain available as an independent oracle.

Device safety must use participating block/site tensors and the existing
35/10 transfer boundary, with hardware names checked on participating devices.
An unused CUDA device must never enter placement validation or allocation.
The production split and architecture state protocols are unchanged.

## Validation and release gates

Run the existing guard, short exact, MTP, concurrency and valid-range long
context harnesses in stock, bespoke and plan modes. Compare exact tensors,
tokens, persistent-state valid ranges and transfers using established rules.
Use separate result directories, preserving failures. Compare equivalent
placement under CUDA_VISIBLE_DEVICES=2,0 and 2,0,1, including native calls,
outputs/state, transfers and unused-device memory. Benchmark bespoke versus
plan (not just stock); investigate repeatable >2–3% losses. No release tag,
wheel promotion or parity claim until all gates pass.

Build discovery already includes all .cpp files. Additive bindings preserve
legacy extension APIs. Plan selection is opt-in and an older installed
extension must fall back safely when the new binding is absent.

## Selection and migration

Set flags before importing the package. The architecture flags are still
`EXL3_NATIVE_QWEN_BLOCK` / `EXL3_NATIVE_GLM_BLOCK`:

| Mode | Architecture native flag | EXL3_NATIVE_PLAN |
| --- | --- | --- |
| Stock | 0 | either |
| Bespoke reference | 1 | 0 (default) |
| Generic plan | 1 | 1 |

The profiling tools set architecture flags explicitly during A/B runs. Select
the development extension with their `--extension-dir
native-results/block-ir/extension` option. Loading the installed older extension
with the plan flag set safely yields stock fallback; it cannot test the planner.
`--compare-plan` on ordinary/MTP/concurrency profiling tools runs warmed direct
backend comparisons. `validate_native_plan.py` coordinates these tools, records
subprocess exit codes, stops on failures, and never installs packages.

The constructor takes site tensor bundles, mix kinds, epsilon/iteration arrays,
three vectors of integer `(opcode, site)` pairs, optional router tensors/MoE
handle, and explicit B,Q. The primitive IDs are internal to this v1 extension
and emitter; use the matching Python/native build. Empty middle stages are
allowed, but calling one raises. Begin/middle return cached [B,Q,D] activations;
finish updates the provided residual. Kernels consume [B*Q,H,D] views.

A compatible architecture should add a Python emitter and eligibility guard,
then reuse this constructor and its existing Python stateful sublayer calls.
It should not add an architecture C++ class. The initial HC binding deliberately
retains the measured H=4,D=4096 contract; GR derives its dimensions from weights.
This is a restricted execution plan, not an arbitrary tensor graph API.
Plans assume immutable loaded weights and serialized forwards on the existing
streams. Reload through model/module load/unload, which clears the native caches.
The validation-only backend switch also clears caches before and after switching
and must not run concurrently with inference.

## Running results

Implemented additive `BC_BlockPlan`, six resolved operations, architecture-side
emitters and `EXL3_NATIVE_PLAN=1` selection. Existing architecture flags still
control native activation; unset plan flag preserves bespoke selection.
Production packages/dependencies and protected wheels/tags are untouched.
Build with `util/build_native_plan.bat`; the development extension goes to
`native-results/block-ir/extension`. It is not installed into Tabby.

The generic GLM guard resolves actual devices from all 45 layers, checks every
block's sublayer placement, validates the existing 35/10 hardware relationship
and final norm/head, and checks draft placement for MTP. Legacy GLM keeps its
ordinal restrictions but no longer rejects unrelated visible GPUs. Python
shape/state eligibility stays architecture-specific. Existing load/unload hooks
clear all caches. Qwen plan shape keys distinguish B,Q/device/dtype; GLM cached
signatures now include these plus backend selection and weight identities.
Weights are immutable within a loaded Qwen model, as in its reference path;
load/unload is the supported rebinding mechanism.

The generic backend retains separate site scratch (GLM legacy reuses one site's
buffers). Both are temporary only. Boundary views are cached per specialization.
Constructor resolution checks instruction/site ordering and bindings; runtime
boundary contracts are checked before residual writes. Native errors propagate,
never trigger a stock retry after partial execution.

Completed initial gates (2026-09-26):

* Qwen existing guards: 29 pass; GLM ordinary guards: 8 pass; GLM MTP: 16 pass;
  GLM concurrency: 19 pass. New generic/placement suite: 30 pass (includes
  inherited legacy guard cases; these counts are not independent coverage).
* First Qwen stock/plan 128-step test: exact logits, tokens, all 48 block
  residuals, recurrent/PLE, valid KV/QSA ranges. Initial smoke: ~72.0 t/s.
* First GLM stock/bespoke/plan 256-step test: exact all 45 block residuals,
  routing, final hidden, transfers, positions, recurrent/conv, latent KV,
  indexer and pool state. Initial smoke: ~45.1 t/s.
* Rebuilt native primitive gate: exact bespoke/plan begin/middle/finish for
  B1..4 Q1 and B1..3 Q4; invalid boundary shapes leave residuals unchanged.
* Rebuilt Qwen three-way 128-step and GLM three-way 256-step gates passed.
  Three alternating 1000-token bespoke/plan pairs passed with identical hashes.
  Median Qwen: bespoke 46.6703, plan 46.7611 t/s (+0.19%). Median GLM:
  bespoke 47.3202, plan 47.2028 t/s (-0.25%). Earlier smoke rates and stock
  rates fluctuated; all measurements are retained rather than filtered.
* Full native guard discovery: 104 pass. Initial discovery exposed a test-only
  `tests/util.py` namespace collision; the helper test now loads its utility
  by explicit file path. The failure and successful rerun are recorded here.
* Qwen MTP: natural and forced-rejection stock/plan exact gates pass (128/64
  output tokens). Bespoke reference runs and direct saved-event comparisons
  pass too. Three 1000-token direct pairs: bespoke median
  102.7391, plan 102.5868 t/s (-0.15%). All pairs have identical completion
  hashes, 666 accepted and 670 rejected drafts, and matching native/fallback
  counters (17368 native block calls, 313 expected prefill/repair fallbacks).
* GLM MTP: natural and forced-rejection exact gates pass (5905/8695 events),
  including direct bespoke/plan saved-event comparisons. Three 1000-token pairs:
  bespoke median 65.0627, plan 65.0093 t/s (-0.08%). Identical completion hashes
  and acceptance totals (646 accepted, 416 rejected). The draft stays stock.
* Extension capability is now resolved once at import; eligibility reads a
  boolean rather than adding a Python helper call per block/token. All 104
  guards passed again after this change.
* Bespoke concurrency reference suites completed for Qwen and GLM ordinary
  and MTP, including page crossing, staggered arrivals, forced rejection
  and GLM cancellation. Each
  passed its stock comparison. A fresh 104-test guard run is retained in
  `native-results/block-ir/guards-final.log`.
* Generic Qwen ordinary concurrency passed B2/B3/B4, staggered arrivals,
  stochastic sampling and page crossing against stock. Three direct paired
  measurements per batch passed completion/state checks. Median aggregate
  output t/s (bespoke / plan): B2 106.2020 / 106.3178 (+0.11%),
  B3 133.3443 / 132.8509 (-0.37%), B4 155.2250 / 155.1229 (-0.07%).
* Generic Qwen MTP concurrency passed B2/B3, staggered forced rejection and
  page crossing. Direct performance medians: B2 96.8121 / 96.7545 (-0.06%),
  B3 95.9763 / 99.3209 (+3.48%). B3 samples varied across roughly 95–100 t/s;
  all samples are retained and this does not establish a stable speedup.
* Generic GLM ordinary concurrency passed B2/B3/B4, staggered arrivals,
  stochastic sampling, page crossing and cancellation. Three paired direct
  measurements per batch passed: B2 48.7608 / 48.5129 (-0.51%),
  B3 60.7386 / 60.4814 (-0.42%), B4 68.2665 / 68.2048 (-0.09%).
* Generic GLM MTP concurrency passed B2/B3, staggered forced rejection,
  page crossing and cancellation. Direct medians: B2 71.5007 / 71.9616
  (+0.64%), B3 50.3692 / 49.9738 (-0.78%). The full concurrency phase
  completed with exit zero, including exact saved-event comparisons between
  bespoke and generic for every Qwen/GLM ordinary/MTP case.
* Constrained-cache page-reuse gates passed for Qwen and GLM with 1024-token
  caches, staggered jobs and required physical-page reassignment.
* Real topology regression passed for both models under `2,0` versus `2,0,1`:
  exact state/completions, identical eligibility, module/native calls and
  participating-device transfers; unused logical CUDA2 peak Torch allocation
  was zero. Bespoke/plan boundary counts matched, generic invocations were
  positive, and ordinary decode fallback counts were zero in both backends.
* Qwen long-context gates passed with the final observer: 65536-token ordinary
  and 32768-token MTP prompts, each with 64 output tokens and exact three-way
  event/state/sparse-selection records. Inputs reuse protected prompt IDs;
  only logically valid KV/pooled and recurrent-history ranges are compared.
* GLM 64K ordinary passed exact stock/bespoke/generic event and sparse-selection
  comparison with the production 35/10 placement, 65216 prompt tokens and
  256 output tokens in a 65536-token cache, including transfer parity.
* GLM 64K MTP passed exact three-way event and sparse-selection comparison:
  13133 state events, 1076 selection events / 3568 rows, with rewind distances
  0,1,2,3 represented and exact participating-device transfers. The draft
  remains stock. All four long gates exited zero.
* Final primitive check passed all seven HC specializations, including B/Q
  distinction and rejection of invalid boundaries before residual mutation.

Diagnostics are retained under `native-results/block-ir`. The first GLM command
was rejected by the benchmark minimum before loading; an explicit `--smoke`
option now distinguishes short gates from release measurements. Its retry
produced successful parity/benchmark/dispatch files but PowerShell returned 1
after reporting Triton stderr remarks. The sequential validation runner now
records Python subprocess exit codes directly and stops on failure.
The first 64K Qwen long-context adapter stopped after its stock pass because it
watched eager Python sparse selection only; single-job attention uses existing
captured native QSA. The adapter now observes that path's selection scratch
immediately after its native call as well. The failed run is preserved under
`remaining/long-failed-adapter`; no inference behavior or parity requirement
was changed.
A second adapter run passed Qwen 64K ordinary, then correctly rejected MTP
sparse records containing initial-prefill positions (minimum 2049 rather than
32768). MTP prefill enters through `forward` too. The observer now wraps both
entry points and identifies initial prefill by logical cache position, matching
the established GLM harness and retaining draft-repair observations. This run
is preserved in `remaining/long-failed-mtp-adapter`; the long gates are rerun.

`util/native_plan_validation.py` shares backend switching/clearing and direct
alternating benchmarks. Ordinary exact harnesses run the bespoke oracle as well
when plans are selected. Native profiling reports calls, fallback eligibility
and logical transfers outside measured runs. The topology phase compares exact
state records/hashes, completion hashes, calls, eligibility, transfers and
unused-device allocator peaks across both requested visibility mappings.

All model correctness and performance gates passed. Repeated direct benchmark
median changes range from -0.79% to +3.48%; the positive outlier has visible
run-to-run variation. Full samples and observational GPU utilization/power are
in `native-results/block-ir/performance-summary.json` and its Markdown table.
The `1.5.1+nativeplan.v1` wheel was packaged without compilation; its extension
hash matches the tested binary and every packaged Python file matches source.
Extracted-wheel Qwen and GLM three-way ordinary gates both passed under
`CUDA_VISIBLE_DEVICES=2,0,1`. Metadata confirms both package and extension were
loaded from the extracted wheel. The production environment is unchanged.
The release README and SHA-256 manifest identify the resulting checkpoint and
artifacts; see [benchmark details](native_block_ir_benchmarks.md) and
[release instructions](native_block_ir_release.md).
Both bespoke implementations remain intact as opt-in comparison oracles.
