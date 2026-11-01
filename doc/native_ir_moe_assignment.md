# Bounded native IR routed-MoE assignment — implementation and validation

Implementation branch: `feature/native-ir-moe-assignment`, based on
`d62891dee3e4cf233207c0a7ec765adba4b7bc6f`. No commits or merges are part of this work.

This report accompanies `nemotron_moe_performance.md`. It concerns Option B only:
device-resident assignment preparation using the existing expert and gather kernels.

## Executive result

**Retain Option B as an opt-in native IR stage.** It removes a measured concurrent
decode bottleneck without changing expert arithmetic. Three-session aggregate
throughput improved by **73–76%** and four-session throughput by **60–69%** in the
final short/4K/16K paired measurements. Net decode-round savings were **39–45 ms**.
These are measured results on the specified Windows/RTX PRO 6000 system, not
projections to other hardware or 128K contexts.

Same-shape stock/IR comparisons were exact: metadata, expert contributions,
routed outputs, full-model logits, generated tokens, acceptance/rejection,
recurrent settlement and cache writes. The relevant regression suite passed
**1,083 tests and 100 subtests**, plus a subsequently added workspace-canary test.
The implementation stays disabled by default, uses no new CUDA graphs, and does
not enable other models. The installed inference extension remains unchanged;
the rebuilt extension and all results are in an isolated evidence directory.

## Architecture and contract

`exllamav3/modules/native_moe.py` owns a modular execution stage, independent of
whole-block native IR plans. Enable it explicitly with `EXL3_NATIVE_MOE_ASSIGN=1`.
The default is stock execution. An older extension without assignment ABI 1 falls
back before allocation or dispatch. Existing v1/v2 native plan APIs are unchanged.

The new CUDA operation consumes contiguous CUDA int64 selected IDs `[R,K]` and
unchanged fp16 routing weights. Bounds are `1 <= R <= 16`, `1 <= K <= 32`, and
`1 <= E <= 512`. It produces int64 counts `[E+1]`, base/start/kind tables
`[3,E+1]`, packed token IDs and weights, inverse mapping, and zeroes the fp32
output `[R,H]`. It checks dtype, device, layout, shape and writable-buffer
nonoverlap before launching. Invalid IDs join sentinel bucket E, which is never
executed or gathered. Duplicate assignments remain separate, in original order.

Two bounded CUDA kernels perform histogram/prefix scan and stable packing. Only
integer histogram updates use atomics. There are no floating-point atomics, host
count readbacks, sorting, NumPy tables, per-call workspace uploads, or changed
expert arithmetic. Stable packing differs from stock's unspecified order within
each expert; inverse mapping restores original token/top-k order for gathering.
The existing `exl3_moe` writes each weighted contribution and existing
`exl3_moe_gather` accumulates it in original k order.

The low-level metadata operation does **not** prove the expert kernel's capacity.
The first consumer establishes that separately: unchanged `routing_dots` top-k
selection retires each selected expert, giving at most one valid assignment per
token per expert. Thus R12/R16 require at most 16 rows per expert, despite 264/352
total assignments. Arbitrary, hash, duplicated or replaced routing functions are
not admitted by the production adapter. The metadata tests can exercise those
inputs without asserting they satisfy the production capacity proof.

The adapter admits only Nemotron Omni's 512 local experts, top-22, latent width
1024, nongated ReLU-squared, EXL3 mul1 K=4 expert pointer tables, and the validated
RTX PRO 6000 Blackwell geometry (SM120, 188 SMs). At least 22 active experts gives
the same eight-CTA reduction width for known-active stock and unknown-active IR
launches on that device. Tile size remains 16. Other architectures, GPUs, shapes,
quantization/codebooks, calibration/reconstruction, tensor parallelism, expert
offload/slicing, replaced routing functions, nondefault streams and worker
threads fall back. This is an initial validation scope, not a universal GPU
limitation of the metadata builder.

## Integration and lifetime

The upstream-facing change is one import, one optional branch at the existing
routed-expert boundary in `modules/block_sparse_mlp.py`, and one unload cleanup.
The hook receives already-selected IDs/weights and the already-projected routed
input. Router, latent projections, shared expert, Mamba, attention, speculative
acceptance, and cache code are not duplicated or changed. Bindings add one ABI
marker and one operation. The existing extension source discovery picks up the
new `.cu`; no existing native kernel implementation changes.

An IR-owned arena per device and supported shape holds metadata, contribution
slots, output and four 16-row expert buffers. Modules retain arena references;
the shared registry uses weak references. Unloading drops the module's references.
No arena retains model weights or cache pointers; execution reads current expert
tables, so reloading does not replay stale bindings. Consumption and reuse are
serial on the default CUDA stream. Eligibility is checked before mutation;
exceptions after a launch propagate instead of retrying stock after partial work.

CUDA graph capture/replay is deliberately absent from this first implementation.
Existing Mamba/attention graph paths remain in use. A native MoE graph is not
required to remove the assignment readbacks and would add pointer/lifetime state.

## Evidence

Builds, validation records, benchmark repetitions and traces are isolated under
`native-results/native-ir-moe-assignment/`. The live inference extension is not
installed over or replaced. Primary performance files are `final-short-bench/`
and `long-bench/`; `short-bench/` retains the initial non-graph performance
checkpoint. `host/` contains lightweight CPU timing and `trace/` contains the
separate CPU/CUPTI traces and their analysis. None of the instrumented runs is
used to calculate the throughput improvements above.

## Correctness and regression results

| Check | Result |
|---|---|
| New primitive/guard tests in the regression run | 139 passed |
| Additional canary test, after the regression run | Passed; prefix/suffix guards intact at R8/9/12/16/12, including A=512 and all-invalid/concentrated picks |
| Full target MoE comparisons | 240 exact comparisons across all 40 blocks at R12/16/8/9/16/12, repeated in final validation |
| Restored-prefix target forward | Six B/Q cases; exact logits, full recurrent/cache bytes, metadata and divergent per-row settlement; repeated in final validation |
| Initial generation matrix | 32 cohorts, 12,288 generated tokens, exact stock/IR comparisons |
| Final validation including reload | 18 cohorts, 6,912 tokens; exact comparisons; 16 sampled draft-refresh write-window/replay audits |
| Actual unload/reload | All IR arenas released; stock/IR generation exact after reload and exact against pre-reload logits/tokens/acceptance |
| Existing plus new regression suite | 1,083 passed, 100 subtests passed; no failures or skips |
| Existing v1 primitive oracle | Seven production B/Q shapes exact, with transactional invalid-boundary rejection |
| Existing v2 residual oracle | FP16/BF16 on physical GPUs 2 and 0 exact; malformed, aliased, moved/resized operands rejected safely |

Native tests cover unique, concentrated, empty, invalid/sentinel, duplicate,
zero-weight and permuted assignments, stable offsets/packing/inverse mapping,
fixed-k-order gather, repeated execution, malformed/overlapping buffers, and
unchanged quantized expert outputs. Duplicate **metadata** is supported generally;
expert execution with duplicates is tested only when occupancy is independently
bounded. Production does not admit arbitrary duplicate-producing routers.

The generation matrices use independent and identical coding prompts, B1/B3/B4,
three MTP drafts, fixed greedy decoding, 128-token completions, both accepted and
rejected drafts, and no-MTP/capped-MTP fallback controls. Full logits are hashed
bitwise, not just compared by argmax. Every audited settled state satisfies
`last_history == 0` and `state.position == kv_position == sequence_length - 1`.
The final refresh checks verify unchanged bytes outside the legal write window
and identical cache bytes on replay. R8/R9 remain stock fallback; R12/R16 dispatch
was verified explicitly. Fresh arenas/graphs after real reload cover cold
execution; repeated cohorts cover warmed execution and shape transitions.

Regression coverage includes existing native plan/model registration and fallback
guards, v1/v2 residual semantics, native graph replay and process-level graph
policy, HC/GR kernels, all routed-MoE kernel tests (including half-rate and
deterministic gather tests), routing paths, Mamba recurrence and batched rewind.
Other model checkpoints were not loaded for end-to-end generation: their native
guards and synthetic execution interfaces were tested, and they remain disabled
in this stage. ROCm, TP/offload and arbitrary multistream execution are unvalidated
and unsupported by the initial adapter.

Two initial build attempts needed local fixes in the new file's CUDA error-check
and header includes. The first model diagnostic needed an outer inference-mode
context before directly manipulating cache tensors. These diagnostic/build
failures are preserved in the evidence directory; none required changing stock
execution or relaxing a correctness expectation.

## Decode measurement method

Environment: Python 3.13.12, PyTorch 2.9.0+cu128, CUDA 12.8, physical GPU 2
`NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`, exposed as `cuda:0`.
Checkpoint: `turboderp_NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-exl3_sc_4.23bpw`.
Its 40 target routed blocks use K=4 mul1 experts, latent width 1024 and padded
intermediate width 2688. The same isolated extension is used for stock and IR.

Each fresh cohort creates three or four jobs with independent coding prompts,
greedy sampling, seeds `1234+i`, three fixed MTP drafts, uncapped verification,
and a 2,048-token job limit. Every job finishes prefill before 12 warmup rounds;
then 48 decode rounds are timed, with synchronization at the window boundaries.
One priming stock/IR pair warms both implementations. Three measured pairs follow
in alternating order: stock/IR, IR/stock, stock/IR. No job finishes during these
timed windows. Prefill is recorded separately and excluded from throughput.

Short prompts are 42/32/30/34 tokens. Longer prompts contain exactly 4,096 or
16,384 tokens of repeated coding context followed by the same coding task.
These are controlled synthetic long contexts, not full real repositories.
The input tuning cache and every measured output cache have SHA-256
`8a2a3ca2fe28c0f21beb9db93b1ce72502c4d0550dee5df71cd627ee1ca74fef`.
Tokens, warmup/window boundaries, draft acceptance/rejection and completion
hashes match within every stock/IR pair and across repetitions. Cross-batch
or cross-shape token identity is not asserted.

| Prompt context / sessions | Tokens in each 48-round window | Generated-token counts at window start → end, per session | Accepted / rejected drafts in window |
|---|---:|---|---|
| Short / 1 control | 165 | `[42] → [207]` | `[117] / [27]` |
| Short / 3 | 527 | `[42,38,43] → [221,206,223]` | `[131,120,132] / [13,24,12]` |
| Short / 4 | 708 | `[42,38,43,38] → [221,206,223,219]` | `[131,120,132,133] / [13,24,12,11]` |
| 4K / 3 | 499 | `[45,45,46] → [206,213,216]` | `[113,120,122] / [31,24,22]` |
| 4K / 4 | 665 | `[45,49,43,49] → [203,217,213,218]` | `[110,120,122,121] / [34,24,22,23]` |
| 16K / 3 | 496 | `[45,45,46] → [211,212,209]` | `[118,119,115] / [26,25,29]` |
| 16K / 4 | 663 | `[45,45,46,49] → [203,215,219,211]` | `[110,122,125,114] / [34,22,19,30]` |

Settled context length at either boundary is prompt length plus the listed
generated-token count minus one. Measured verification shapes are `(3,4)` and
`(4,4)`; the control is `(1,4)`. Cache capacity is 16,384 token slots in short
runs and 81,920 in long runs. Four 128K contexts were not measured, and these
results do not establish 128K throughput or latency.

## Throughput and delivery latency

All entries below are uninstrumented. Throughput is aggregate generated tokens/s;
round times are medians of the three per-run mean round times.

| Context / sessions | Stock median [range] | IR median [range] | Median throughput change | Stock → IR round ms |
|---|---:|---:|---:|---:|
| Short / 1 | 106.82 [105.80–106.88] | 106.04 [105.30–106.24] | −0.7% | 32.18 → 32.42 |
| Short / 3 | 110.43 [109.48–115.12] | 194.79 [194.44–199.42] | +76.4% | 99.42 → 56.36 |
| Short / 4 | 132.95 [131.80–135.42] | 224.85 [210.76–225.92] | +69.1% | 110.95 → 65.60 |
| 4K / 3 | 108.33 [92.73–108.55] | 187.57 [187.31–188.67] | +73.1% | 95.97 → 55.42 |
| 4K / 4 | 134.82 [133.76–135.73] | 215.85 [215.02–216.90] | +60.1% | 102.76 → 64.19 |
| 16K / 3 | 105.98 [104.06–108.43] | 186.01 [185.60–187.46] | +75.5% | 97.51 → 55.55 |
| 16K / 4 | 132.60 [112.45–133.71] | 212.42 [202.75–220.85] | +60.2% | 104.16 → 65.02 |

The single-session control executes no native assignment stage. Its small
measured regression is consistent with an extra opt-in dispatch check and timing
variation; no single-session improvement is claimed. At short context, average
per-session throughput is approximately 36.8 → 64.9 tokens/s for three sessions
and 33.2 → 56.2 for four.

Individual repetitions, in order r0/r1/r2:

| Context / sessions | Stock tokens/s | IR tokens/s |
|---|---|---|
| Short / 1 | 106.88, 106.82, 105.80 | 106.04, 106.24, 105.30 |
| Short / 3 | 109.48, 110.43, 115.12 | 194.79, 194.44, 199.42 |
| Short / 4 | 131.80, 132.95, 135.42 | 224.85, 210.76, 225.92 |
| 4K / 3 | 108.33, 92.73, 108.55 | 187.57, 188.67, 187.31 |
| 4K / 4 | 133.76, 134.82, 135.73 | 216.90, 215.85, 215.02 |
| 16K / 3 | 104.06, 108.43, 105.98 | 186.01, 185.60, 187.46 |
| 16K / 4 | 112.45, 133.71, 132.60 | 212.42, 220.85, 202.75 |

No slow repetition was removed. Notable variability includes stock 4K/B3 r1,
stock 16K/B4 r0, and IR short/B4 r1. The earlier checkpoint run is also retained:
short B3 median 112.67 → 199.04 and B4 135.64 → 228.31 tokens/s. The consistent
gain across alternating pairs and both short runs is stronger evidence than any
single best run; these measurements are not a statistical confidence interval.

Per-session positive-delivery interarrival times, pooled over all repetitions:

| Context / sessions | Stock p50 / p95 ms | IR p50 / p95 ms |
|---|---:|---:|
| Short / 1 | 32.39 / 33.23 | 32.61 / 33.34 |
| Short / 3 | 97.55 / 103.09 | 55.96 / 58.33 |
| Short / 4 | 108.43 / 128.40 | 66.04 / 71.33 |
| 4K / 3 | 96.63 / 128.82 | 55.13 / 58.69 |
| 4K / 4 | 103.56 / 106.81 | 64.08 / 66.93 |
| 16K / 3 | 96.78 / 103.15 | 55.56 / 58.23 |
| 16K / 4 | 105.13 / 131.23 | 65.06 / 71.28 |

MTP delivers bursts of up to four tokens; these are burst-arrival latencies,
not separately spaced latency for every token in a burst. Per-round burst p50/p95
and full per-session token counts/timestamps are also in `measurements.json` and
the raw cohort JSON.

## Attribution: what actually became cheaper

Lightweight host-only timing over 32 matched short-context rounds gives the
following **inclusive CPU call spans**, in ms per round. They include dispatch
and any blocking within those calls; they are not additive with GPU time.

| Span | B3 stock → IR | B4 stock → IR |
|---|---:|---:|
| Target forward | 83.88 → 36.00 | 87.03 → 45.61 |
| Assignment preparation boundary | 43.65 → 4.34 | 47.23 → 5.96 |
| Stock argsort | 5.62 → absent | 5.33 → absent |
| Stock bincount | 10.06 → absent | 13.34 → absent |
| Explicit count readback | approximately 3.00 → absent | approximately 3.01 → absent |
| New assignment binding, within preparation | absent → 1.59 | absent → 2.13 |
| Grouped-expert dispatch | 1.91 → 1.03 | 1.91 → 1.25 |
| Gather dispatch | 1.03 → 0.80 | 1.07 → 0.92 |
| Shared expert | 6.46 → 6.20 | 6.47 → 7.45 |
| Three draft forwards | 4.08 → 4.13 | 4.09 → 6.21 |

Preparation includes indexing, slot-table construction/upload and intervening
Python work. The new boundary includes capability checks and scratch lookup.
Thus its reduction of approximately **39–41 ms** is a measured host-span result,
not a sum of nominal readback waits. Draft execution was not optimized; its B4
variation illustrates why stage timings should not replace matched whole-cycle
measurements.

Separate eight-round CPU/CUPTI traces establish the device and synchronization
mechanism. Values below are per verification round:

| Trace metric | B3 stock → IR | B4 stock → IR |
|---|---:|---:|
| Target GPU activity time, ms | 40.302 → 38.833 | 45.616 → 44.163 |
| Preparation GPU activity time, ms | 1.874 → 0.407 | 1.883 → 0.444 |
| Expert GPU time, ms | 27.478 → 27.526 | 32.538 → 32.570 |
| Gather GPU time, ms | 0.256 → 0.254 | 0.257 → 0.257 |
| Shared-expert GPU time, ms | 2.148 → 2.147 | 2.203 → 2.207 |
| Preparation stream synchronizations | 120 → 0 | 120 → 0 |
| Preparation `cudaLaunchKernel` calls | 400 → 80 | 400 → 80 |
| Target `cudaLaunchKernel` calls | 978 → 618 | 978 → 618 |
| Target cooperative-launch calls | 161 → 161 | 161 → 161 |
| Target GPU activities, including copies/memsets | 1,700 → 1,060 | 1,700 → 1,060 |

The 120 removed synchronizations are the two internal bincount scalar readbacks
plus one count-vector readback in each of 40 blocks. Target preparation has zero
such calls after dispatch. Whole-cycle sampling/scheduling synchronizations
remain. The new operation launches two kernels per layer, 80 per target forward.
Expert and gather GPU time remain effectively unchanged.

The full profiler substantially slows execution: traced round spans are roughly
166 → 114 ms for B3 and 178 → 108 ms for B4, versus the uninstrumented timings
above. Within those traces, target CPU spans with no recorded device activity
fall from 103.25 → 50.61 ms and 106.21 → 36.84 ms. Preparation alone accounts for
60.31 → 9.34 ms and 66.04 → 8.56 ms of those gaps. These are **profile-inflated
idle observations**, not estimates of removable production time or SM utilization.
The actual CUDA synchronization API spans total only about 0.9 ms per round in
these heavily instrumented traces. Neither synchronization counts nor inclusive
host spans may be converted directly into a speedup.

The defensible critical-path result is the **39–45 ms whole decode-round saving
in the uninstrumented paired runs**, corroborated by lower lightweight preparation
time, removed readbacks/launches and unchanged expert computation. The earlier
15–30 ms estimate was an unverified hypothesis; this implementation's measured
benefit on this environment is larger.

## Memory, graphs and remaining opportunities

Exact owned tensor storage is **6,615,728 bytes at R12** and **6,994,144 bytes at
R16**, approximately **12.98 MiB combined**. This includes counts, tables, packed
IDs/weights, inverse mapping, output, contribution scratch and four 16-row expert
buffers. It is shared across all eligible layers, not multiplied by 40. Existing
stock buffers remain available for fallback. Paired memory readings include both
warmed arenas even when the stock flag is selected, so their difference is not
an independent measurement of persistent overhead. Raw allocated/peak memory
readings are retained in each cohort record.

No new CUDA graphs were implemented. The non-graphed stage already clears the
performance checkpoint. Its 0.4 ms of assignment GPU work is small relative to
27.5–32.6 ms of expert computation. Remaining host enqueue cost does not establish
an equal graph speedup because it overlaps device work. Graph capture would add
binding, lifetime and fallback complexity; there is no separately measured
incremental benefit here to justify expanding scope now.

Shared-expert GPU work remains approximately 2.15–2.21 ms per target verification.
Its inclusive host span is about 6–7 ms and its percentage of the now-shorter
target forward is larger, but those figures are not an achievable saving. Keep
multi-row shared-expert graphing secondary and separately benchmark it before
implementation. Mamba and attention execution remain unchanged.

## Reuse and maintenance boundaries

The reusable result is the assignment ABI and its owned buffers, not a new router
or generalized MoE subsystem. A future gated consumer can pass its existing gate,
up and down tables to the same stock expert kernel after separate validation.
The prior investigation verified potential consumers in GLM 5.3 Flash (top-8),
Qwen 3.8 Flash Next (top-10), MiMo V2.6 (top-8), and DeepSeek V4 Flash (top-6,
including hash-selected layers). They remain disabled here.

Capacity alone is insufficient for reuse. Stock's `num_active` can change the
CTA reduction width and therefore quantized arithmetic. Nemotron's top-22 and
188-SM bound proves the same width for every valid finite routing distribution.
Lower-top-k models need their own scheduling/parity proof; copying this adapter
would not establish it. Hash duplicates also need an explicit occupancy bound
(possibly total assignments) rather than assuming uniqueness. Gated activations,
expert biases, other codebooks, different GPUs, TP and expert offload require
separate guards and tests. No throughput benefit for those models is claimed.

Upstream merge exposure is confined to the routed-output hook, unload cleanup,
binding additions, the `MultiLinear` metadata interface, and the existing expert/
gather signatures. Rebase validation must check those signatures, table semantics,
kernel capacity and launch geometry before retaining ABI compatibility. The new
stage does not fork model implementations, routing, Mamba, attention, or expert
arithmetic. That makes this substantially smaller to maintain than a copied
Nemotron forward path, while still requiring native compilation and regression
testing after upstream kernel changes.

## Files and source locations

| Ownership | File / location | Change |
|---|---|---|
| New IR implementation | `exllamav3/modules/native_moe.py:26` | Capability proof and opt-in fallback guards |
| New IR implementation | `exllamav3/modules/native_moe.py:82`, `:106` | Arena ownership and assignment/expert/gather execution stage |
| New IR implementation | `exllamav3/exllamav3_ext/libtorch/native_moe_assignment.h:1` | Reusable metadata ABI contract |
| New IR implementation | `exllamav3/exllamav3_ext/libtorch/native_moe_assignment.cu:16`, `:51`, `:70` | Histogram/scan, stable packing, argument/alias checks |
| Minimal integration | `exllamav3/modules/block_sparse_mlp.py:21`, `:1023`, `:1151` | Import, arena cleanup, optional dispatch; six inserted lines total |
| Minimal integration | `exllamav3/exllamav3_ext/bindings.cpp:72`, `:337` | Header, ABI marker, binding; three inserted lines total |
| New tests | `tests/kernels/moe/test_native_moe_assignment.py` | Metadata, gather, exact expert arithmetic, validation and canary coverage |
| New tests | `tests/modules/moe/test_native_moe_guards.py` | Unsupported cases, opt-in/missing ABI, stream/thread guards and arena release |
| External utilities | `util/build_native_moe_assignment.bat`, `util/validate_native_moe.py`, `util/summarize_native_moe.py` | Isolated build, reproducible validation/benchmark/profiling, analysis |
| Report | `doc/native_ir_moe_assignment.md` | This implementation report |

Unmodified dependencies underpinning the proof: `routing.cu:91` retires each
top-k winner; `modules/block_sparse_mlp_routing.py:249` selects the existing dots
router; `quant/exl3_moe.cu:329` onward validates buffer dimensions, `:491` chooses
CUDA group geometry, and `:593` gathers contributions in original k order.
Paths for these CUDA files are beneath `exllamav3/exllamav3_ext/`.

The working tree had no tracked edits at branch preparation. Existing untracked
reports, prompts and diagnostics were preserved. Prior diagnostic/report hashes
and the installed extension hash are rechecked in `integrity.json`. There are no
stock native-kernel edits, no model-architecture edits, no modifications to
existing native-plan implementations, and no commits or merges.

## Reproduction and opt-in use

From `D:\exllamav3`, the isolated build is:

```powershell
cmd /c util\build_native_moe_assignment.bat
```

It compiles SM89 and SM120 using the established TextGen Python, VS 2022 Build
Tools and CUDA 12.8, without installing over the live extension. For this build:

```powershell
$py = 'D:\text-generation-webui\installer_files\env\python.exe'
$env:CUDA_VISIBLE_DEVICES = '2'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'D:\exllamav3\native-results\native-ir-moe-assignment\extension;D:\exllamav3;D:\exllamav3\native-results\nemotron-shape-correctness\test-deps'
$modelPath = 'D:\text-generation-webui\user_data\models\turboderp_NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-exl3_sc_4.23bpw'
$tunePath = 'native-results/nemotron-moe-performance/long-cuda/final-tune.bin'
& $py -c "from exllamav3.ext import exllamav3_ext as e; print(e.__file__); assert e.native_moe_assignment_abi == 1"
```

For normal inference, set `EXL3_NATIVE_MOE_ASSIGN=1` before importing ExLlamaV3,
with the rebuilt extension selected as above. `EXL3_NATIVE_PLAN=1` is not required
for this modular stage. Unset the new flag or set it to `0` for stock execution.
An old extension safely falls back; the flag alone does not rebuild or install it.
This validation uses one serial generator on the main Python thread/default
stream, batching independent sessions. Worker-thread, separate-process serving,
or arbitrary multistream throughput is not established by these results.

Use fresh output directory names when rerunning:

```powershell
& $py util/validate_native_moe.py --model $modelPath --output native-results/native-ir-moe-assignment/reproduce-validation --tune-from $tunePath --phase validate --repeats 2 --tokens 128 --reload
& $py util/validate_native_moe.py --model $modelPath --output native-results/native-ir-moe-assignment/reproduce-short --tune-from $tunePath --phase bench --batches 1,3,4 --rounds 48 --warmup 12 --repeats 3
& $py util/validate_native_moe.py --model $modelPath --output native-results/native-ir-moe-assignment/reproduce-long --tune-from $tunePath --phase bench --contexts 4096,16384 --rounds 48 --warmup 12 --repeats 3
& $py util/validate_native_moe.py --model $modelPath --output native-results/native-ir-moe-assignment/reproduce-host --tune-from $tunePath --phase host --rounds 32 --warmup 12
& $py util/validate_native_moe.py --model $modelPath --output native-results/native-ir-moe-assignment/reproduce-trace --tune-from $tunePath --phase trace --rounds 8 --warmup 12
& $py util/summarize_native_moe.py native-results/native-ir-moe-assignment/reproduce-short
& $py util/summarize_native_moe.py native-results/native-ir-moe-assignment/reproduce-long
& $py util/summarize_native_moe.py native-results/native-ir-moe-assignment/reproduce-host
& $py util/summarize_native_moe.py native-results/native-ir-moe-assignment/reproduce-trace
```

The validation utility switches the opt-in flag internally for each paired run.
No throughput run has correctness/timing wrappers or a profiler attached.
The final short rerun follows the final K=4/layout guard tightening; long runs
used the same validated kernel/stage path before those additional rejection-only
checks. All checkpoint experts already satisfy both checks.

The regression commands used the same isolated extension:

```powershell
$env:EXL3_NATIVE_MOE_ASSIGN = '1'
$env:EXLLAMAV3_TUNE_CACHE = 'D:\exllamav3\native-results\native-ir-moe-assignment\regression-tune.bin'
$env:TRITON_CACHE_DIR = 'D:\exllamav3\native-results\native-ir-moe-assignment\regression-triton'
$nativeTests = (Get-ChildItem tests/test_native*.py).FullName
& $py -m pytest @nativeTests tests/kernels/moe tests/modules/moe/test_routing_paths.py tests/modules/moe/test_native_moe_guards.py tests/kernels/recurrent/test_mamba2_recurrent.py tests/kernels/recurrent/test_recurrent_rewind_batched.py tests/kernels/hyperconnections -q --basetemp=native-results/native-ir-moe-assignment/reproduce-test-tmp
$env:CUDA_VISIBLE_DEVICES = '2,0'
& $py util/check_native_plan_primitives.py --extension-dir native-results/native-ir-moe-assignment/extension --output native-results/native-ir-moe-assignment/reproduce-v1.json
& $py util/check_residual_plan_ops.py --extension-dir native-results/native-ir-moe-assignment/extension --output native-results/native-ir-moe-assignment/reproduce-v2.json
```

With the subsequently added canary included, the pytest command now selects
1,084 tests plus the same subtests. Original logs retain the 1,083-test run and
the separate passing canary check.

## Recommendation

1. **Retain and use the bounded stage for the validated concurrent Nemotron
   workload, opt-in.** The measured improvement is large relative to roughly
   13 MiB of scratch and the small integration surface. Exact same-shape parity
   avoids the main numerical risk of changing expert math.
2. **Keep graph capture and shared-expert graphing as separate future proposals.**
   Neither has a measured incremental return here. Do not expand this change
   into a broader MoE subsystem or model-forward fork.
3. **Reuse the metadata operation only after per-model validation.** The operation
   is reusable; capacity, scheduling geometry, routing semantics, quantization
   and serving-thread requirements still need explicit proof for each adapter.

No further optimization, model enablement, deployment, commit or merge is part
of this implementation task.
