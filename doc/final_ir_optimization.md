# Final native IR optimization investigation

The retained optimization is GLM-only HC stage graphs. The clean build improves
matched 8192-token GLM TP2 MTP decode from 82.79 to 90.02 tok/s (+8.73%), with
exact completions and acceptance. All 196 regression tests and final isolated
cold-start/restart checks passed. No installed package has been changed.
DeepSeek graph activation was rejected
because long MTP requests also fail repeatability with the unchanged old binary.
The versioned evidence index is [final_ir_measurements.json](final_ir_measurements.json);
full local logs, profiles and binaries remain under this worktree's ignored
`native-results` directory.

## Provenance and isolation

Started 2026-10-25 from `68e4c7954a74ebea0df14e5bb7cce42619d406c5` on
`feature/native-ir-tensor-parallel`. The investigation branch is
`feature/native-ir-final-optimization`, in
`D:/exllamav3/native-results/final-ir-worktree`. Original commits, the unstaged
`convert.py` change, untracked prompts and investigation files are untouched.
No upstream merge, dependency upgrade, package installation, push or merge.
No applicable AGENTS.md was found.

Python is 3.13.12, Torch 2.9.0+cu128, CUDA runtime 12.8, and Tabby's Triton is
3.8.0.post28. Tabby and WebUI both have ExLlamaV3 1.5.4+nativeplan.v2.tp2;
WebUI's independently inspected Triton metadata is 3.5.1.post24. Tests import
Python from the new worktree. Baselines use the existing extracted tp2 extension at
`D:/exllamav3/native-results/tp-cold-start/package/extracted`; final tests use the
clean candidate build described below. Each run records
the binary hash, all Python source hashes, model config, quantized file sizes
and modification times, device UUIDs and actual TP shard allocation.

Physical devices 0/1 are 48 GB RTX 4090 D, device 2 is the 96 GB RTX PRO 6000
Blackwell Max-Q. Driver 610.88, Windows WDDM, AMD EPYC 7663. Device selection
is confined to test commands through CUDA_VISIBLE_DEVICES. Initial compute
devices were idle. The campaign refuses to overlap another GPU Python process.

## Current architecture and coverage audit

Reviewed the native block v1 design/benchmarks, v2 design/report, TP integration
and cold-start reports, Qwen/GLM experiments, v154 integration/investigation,
DeepSeek implementation and completed validation, Gemma assessment, Devstral
mapping/report and Laguna mapping/report. Earlier negative Gemma v1 and
DeepSeek state results are superseded by their later completed validation;
they are not current failures. Historical throughput is not a baseline here.

| Family | Native stages | Remaining Python boundary | Current limit |
| --- | --- | --- | --- |
| Qwen3.8 | GR/APPLY; single-GPU also ROUTE/MOE | Attention, state; TP keeps routing/experts/reductions in upstream MLP | TP B1Q5 or B1..4Q1, at most 5 rows |
| GLM-5.3 | HC/RMS/APPLY | KDA/MLA and state, routing/experts, upstream communication, draft | B<=4, Q<=6, B*Q<=12 |
| DeepSeek V4 | HC/RMS/APPLY, including supported draft sites | DSA/compression/state, hash/configured routing, experts, communication | B<=4, Q<=6, B*Q<=12 |
| Gemma 4 | Four RMS/postnorm-residual operations | Sliding/global attention and state, dense MLP, exports/layer scalar | B<=4, Q<=6, B*Q<=12 |
| Devstral 2 | RMS, fused residual-in RMS, residual add | Attention/cache, dense MLP, export | B<=4, Q<=6, B*Q<=12 |
| Laguna | Existing v2 norm/residual operations | Attention/SWA, sigmoid router/experts, export | Existing non-TP registration; no checkpoint locally |

Attention/state and communication boundaries are required by the current
architecture. In TP, invoking a local fused MoE instruction without the
upstream distributed module would omit collectives or empty-shard participation.
GLM/DeepSeek routing cannot use the existing softmax ROUTE opcode without
changing semantics. Local expert execution is already fused/graph-backed for
up to eight rows. A 12-row HC plan does not imply a 12-row fused expert path.

Unsupported prefill, overrides/LoRA/capture, incompatible operands, block-local
norm exports and larger shapes retain stock fallback. These checks execute
before native mutation. Native exceptions are not retried through stock.
Loaded binding signatures cover identity, storage, layout, versions and norm
parameters. Invariant operand validation is already outside cached execution.
Per-block/shape scratch persists, and reload clears bindings. No extra
synchronization or collective is inserted by native begin/middle/finish.

TP dispatch already sends child ranks before the inline output rank and
defers child/reducer acknowledgments. `all_reduce` enqueues the upstream
CPU-assisted operation; its host-call duration is not communication latency.
Reduction kernels may include rank skew and host scheduling waits. Three-rank
arrival-order rounding is preserved; stock itself is not bitwise repeatable.

## Measurement method

`util/profile_final_ir.py` uses actual Generator/Job generation, greedy sampling,
fixed prompts/seeds, no ngram speculation, fixed checkpoint-default MTP draft
length, fresh generator/page tables and 65536-token target and draft caches.
Load chunks are 256 tokens, matching the existing TP validation driver. MTP
draft loads on logical cuda:0 before the target; target allocations are recorded.
No offload or alternative sharding is introduced to force a model to fit.
Qwen's checkpoint includes its existing 39,040,214,152-byte host-resident
`ngram_embedding.safetensors` table (108,970,746,600 bytes across all weight
files). Its single-GPU result includes that standard host lookup path; it is
not an entirely VRAM-resident baseline.

Decode throughput is the existing generator metric `new_tokens/time_generate`;
its interval starts at the first token, so this differs slightly from the
strict `(new_tokens-1)/time_generate` convention. Full-request wall throughput
includes job enqueue, prefill, sampling, text delivery and loop work. It excludes
generator construction/automatic startup warmup and model load. Windowed rates
use actual delivered token deltas, with thresholds crossed at speculative
iteration boundaries. The first window includes prefill. Exact window counts,
accepted/rejected drafts and cumulative wall rates are saved rather than
pretending every window has exactly 512 tokens.

CPU and CUDA profiles are separate from final throughput samples. Later profile
passes start after a requested generated prefix to exclude prefill and inspect
the sustained context. Per-layer host timers do not synchronize the GPU and
are inclusive/overlapping. They must not be summed as a partition of wall time.
TP native execution/fallback/build counters are the existing production counters;
diagnostic RPCs synchronize only outside timed intervals. All samples are retained.
The `stock` mode means stock upstream **TP target** execution. It disables the
v2 plan flag, while a supported single-device draft can still use the earlier
specialized native executor. Graph-versus-eager-IR pairs keep that flag enabled
in both modes, so the draft path is identical and cannot explain a graph gain.

The first `glm-screen` profile attempt collided with a parent cProfile because
the output rank is an inline pseudo-worker; its completed timing pairs are
retained, but the profile failed. `glm-long` exposed a keyword-argument mismatch
in the observer (`ids` versus `input_ids`). Both harness issues were corrected;
neither changed production code. `glm-long-r2` includes early CPU/CUDA profiles
and later uninstrumented samples. A separate fresh-process `glm-clean` campaign
does no profiling and warms the longest request before paired timings.

## Baseline results

All rates below are tokens/second. Ranges include every retained sample; no
outliers were removed. These compare existing IR to stock, not a new optimization.

| Model / configuration | Outputs | Stock median (range) | Existing IR median (range) | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| GLM TP2 MTP, fresh unprofiled process | 128 | 76.756 (76.590–76.923) | 77.998 (77.806–78.190) | +1.62% |
| GLM TP2 MTP, same process | 2048 | 81.258 (81.012–81.503) | 81.650 (81.401–81.900) | +0.48% |
| GLM TP2 MTP, same process | 8192 | 81.453 (81.303–81.604) | 82.382 (81.951–82.812) | +1.14% |
| GLM TP2 ordinary decode | 128 | 42.666 (42.560–42.703) | 54.694 (54.390–54.805) | +28.34% |
| Qwen3.8 single GPU, MTP | 128 | — | 103.633 (102.646–105.168) | No stock comparison |
| DeepSeek V4 TP2 MTP | 128 | 31.114 (29.803–35.613) | 33.652 (28.772–37.534) | −5.51%; unstable |
| Gemma 4 TP2 ordinary decode | 128 | 42.650 (42.634–42.684) | 45.525 (45.435–45.567) | +6.74% |
| Devstral 2 TP3 ordinary decode | 128 | 24.646 (24.612–24.847) | 24.691 (24.548–24.745) | −0.26% |

GLM's clean MTP campaign has two alternating pairs per length, all other rows
have three repetitions. Its stock/native full-request median wall rates are
66.335/67.454 at 128, 80.451/80.842 at 2048 and 81.234/82.162 at 8192.
The corresponding native decode standard deviations are 0.271, 0.352 and
0.609; stock deviations are 0.235, 0.347 and 0.213. Other models' full-request
native medians are Qwen 93.207, DeepSeek 31.477, Gemma 44.291 and Devstral
24.059. Full variances, individual pairs and windowed rates are saved in
`native-results/summary.json` and each campaign's `results.json`.

DeepSeek's individual paired changes are +25.94%, −5.51%, −7.53%. That range
does not support either a speedup or a regression claim. A ratio of aggregate
medians would misleadingly conceal the changing runtime state. Devstral's
sub-percent variation is also not a meaningful effect.

The first long GLM process, which collected profiles before timings, measured
native medians/ranges at all five requested output lengths:

| Outputs | Native median | Range | Standard deviation |
| ---: | ---: | ---: | ---: |
| 128 | 75.944 | 75.938–76.222 | 0.162 |
| 1024 | 80.538 | 74.266–80.985 | 3.757 |
| 2048 | 51.022 | 49.047–52.217 | 1.601 |
| 4096 | 50.682 | 49.567–50.831 | 0.691 |
| 8192 | 52.896 | 52.856–52.924 | 0.034 |

All three 8192 samples have 5428 accepted/2864 rejected draft tokens and
2764 target verifications, 124380 native block executions per rank, zero
new plans, and 45 expected prefill fallbacks. Completions and acceptance
match the fresh unprofiled 81–83 tok/s process exactly. Model allocation,
checkpoint metadata, extension and production Python hashes also match.
The slow state persists into the beginning of subsequent short-context
requests. Therefore the observed decline is **not explained by context growth,
MTP acceptance or lost IR coverage alone**. Compilation appeared at an initial
threshold crossing; profiler/runtime/OS scheduling effects remain unresolved.
No sampled thermal/power throttling was reported. A fresh stock warmup itself
sustained 83.02 tok/s for 8192 outputs: this is not a new IR speedup.

Timestamp-aligned telemetry adds a useful constraint: during the graph screen's
1024-token eager decline from 81.49 to 56.11 tok/s, median Blackwell SM clocks
were approximately 2152 then 2197 MHz, with memory at 13365 MHz throughout.
The peer changed from 2700 to 2685 MHz with memory at 10501 MHz. Utilization
shifted between ranks, so a simple GPU clock collapse does not explain that
transition. Other slow-process samples show different peer clock/utilization
patterns. Power management and CPU affinity were left at their existing settings;
these measurements do not identify the runtime-state cause. The per-request
clock observations are preserved in `native-results/clock-correlations.json`.

Local checkpoint directories and quantization labels:

| Model | Directory under `D:/tabbyAPI/models` |
| --- | --- |
| GLM | `GLM-5.3-Flash-Uncensored-EXL3-3.05bpw` |
| Qwen | `Qwen3.8-Flash-Next-Uncensored-exl3-4bpw` |
| DeepSeek | `turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw` |
| Gemma | `turboderp_gemma-4-31b-it-exl3_6.00bpw` |
| Devstral | `turboderp_Devstral-2-123B-Instruct-2512-exl3_4.00bpw` |

Exact config and quantized-file metadata are in each run's `inputs.json`;
the directory bpw labels are not a newly calculated effective bit rate.
Prompt lengths reported by the generator are GLM 23, Qwen 25, DeepSeek 25,
Gemma 25 and Devstral 20 tokens. `prompt.json` preserves the rendered prompt
and token IDs. The prefill forward may consume one fewer token because decode
starts with the final prompt token. MTP draft settings are GLM three tokens,
Qwen four, DeepSeek five. GLM target verification is B1Q4; DeepSeek's observed
verification lengths range from one through six. For 128 outputs, GLM records
81 accepted/60 rejected, Qwen 88/72 and DeepSeek 83/37 draft tokens.

## Bottleneck evidence

The first current GLM TP2 MTP comparison measured 76.282 stock / 77.278 native
tok/s at 128 outputs and 80.767 / 81.352 at 1024. Each is one pair, with matching
completion hashes and acceptance totals; neither establishes a repeatable gain.

The 128-token profile records 47 B1Q4 target verification calls, 141 B1Q1 draft
calls and one 22-token target prefill. Every target verification executes all
45 native blocks on both ranks: 2115 executions/rank, zero new plans, and 45
expected prefill fallbacks/rank. B1 MTP verification is already covered.

Native signature generation accounts for about 74 ms in the instrumented
1.75-second decode interval, of which tensor keys account for 59 ms. Even
eliminating the entire measured signature cost would bound that isolated gain
near 4.4%, before accounting for profiler overhead and CPU/GPU overlap. This
does not justify weakening binding checks or building a new native signature
API. Some cProfile frames have inconsistent cumulative attribution across
extension boundaries; explicit host timers are used as an additional check.

The initial CUDA trace includes prefill and profiler startup. Its raw dominant
large reduction durations must not be attributed to decode. After the final
large-prefill reduction, the output GPU's approximately 249 ms trace region
contains 43.2 ms of communication kernels and 5.0 ms of HC mix/apply kernels;
the peer region contains 57.4 ms and 3.6 ms respectively. These instrumented
regions include drafting, host gaps and overlapping activity. They suggest
that a new HC math fusion is unlikely to provide a large gain; they do not
separate pure PCIe cost from rank-arrival or CPU scheduling waits.

In the first long campaign, later 2048/4096-token requests also start slowly
at short context, with unchanged acceptance totals, zero plan builds and no
verification fallback. An initial threshold crossing compiled attention
kernels, but later repeats remained slower. Sampled GPU status reported no
thermal/power throttle. This does not establish a context-only or IR-only
cause; the fresh unprofiled comparison is necessary.

## Opportunity decisions

| Opportunity | Assessment |
| --- | --- |
| Extend single-request GLM MTP coverage | Already fully native at its default verification shape; no missing IR work to recover |
| Remove repeated signature checks | Measured fraction small; identity/version/layout checks protect rebinding correctness; no unsafe shortcut |
| Fuse HC math | GPU arithmetic fraction small in short trace; requires credible end-to-end evidence before adding kernels |
| Graph entire transformer blocks | Existing attention/GDN/MLP graphs already cover expensive sublayers; full blocks cross dynamic state and CPU-assisted collectives |
| Graph just IR stages | Retained for GLM: +8.73% in the final sustained comparison; pointer/dtype/binding, lifecycle and cold-start gates passed. DeepSeek activation rejected below |
| Move TP routing/MoE into existing IR opcode | Existing opcode does not preserve all router types or distributed semantics; not a safe reuse |
| Broaden batched verification shapes | Real gap: Qwen multi-job verification and GLM B4Q4 exceed native bounds; also exceed existing fused expert tier, so benefit cannot be inferred from IR counts |
| Change TP collective backend/order | Explicitly outside scope; current arrival-order rounding and communication protocol preserved |

## Validation

All 184 existing native/dispatch/attention-export tests passed, no skips, plus
the TP worker terminal-signal regression. `baseline-glm-parity/passed.json`
records exact TP2 logits, block residuals and live state for B1..4Q1 and B1Q4,
and repeated MTP generator requests at B1..4. B4 can execute some native calls
as the active batch shrinks; this is not full B4 verification coverage.

The baseline sequential campaign completed all eight cases. Fresh isolated
Triton caches completed GLM TP2 MTP and Gemma TP3 startup, followed by repeated
requests; no CPU reduction timeout, cache race or startup deadlock occurred.
Additional independent warm-cache processes completed long GLM, ordinary GLM,
and the representative model runs. No timeout was enlarged. The normal caches
were not deleted or replaced. Cold runs are startup checks, not speed claims.

Automatic rank warmup is serialized using the null backend and deliberately
skips native plans. Its "long context" sequence depends on the load chunk size;
with chunk 256 it does not precompile every 8K attention regime. Successful
startup is not proof that every later shape is compilation-free.

## Profiling evidence and stage-graph design

The explicit timers change the decision about stage graphs: in 21 verifications
after 4096 generated tokens, the output rank spends 451.1 ms in target forward
and 51.8 ms in 63 draft forwards. Native selection is 33.2 ms; begin, middle and
finish account for 44.8, 48.9 and 18.2 ms of inclusive host time. These are about
1.58 ms selection and 5.33 ms native-stage host work per target verification.
The other rank's stage costs are similar. Small GPU arithmetic time therefore
does not imply small launch/orchestration cost.

The late-context CUDA trace excludes prefill and contains five target verifies.
The output GPU spans 232.5 ms with 27.5 ms of communication kernels, 5.0 ms HC
mix/apply, and 104.0 ms other kernels; the peer spans 215.9 ms with 85.6, 3.6
and 49.7 ms respectively. These are sums of kernel durations, not additive
wall-time partitions. Peer receive kernels account for 62.4 ms, while output
rank small-gather kernels account for 17.5 ms. The asymmetry demonstrates
rank waits, but cannot separate PCIe transport from CPU reducer and scheduling
latency. Measured host calls per verification are 79 all-reduces, 11 broadcasts,
six small gathers and one gather on each rank. This communication is upstream
work, not extra IR synchronization.

Output-rank attention and MLP host totals in the 21-verification timer pass are
82.1 and 181.2 ms. Draft forwards total 51.8 ms, target forwards 451.1 ms.
Sublayer/collective/stage measurements overlap and must not be added together.
These identify substantial work outside IR; they do not certify a single pure
communication bottleneck. GPU traces also show thousands of existing launches
and existing graph replays; host launch overhead remains a credible constraint.

Capturing the existing three-kernel begin and four-kernel middle reduces seven
launches to two replays per block (five launches saved), leaving the one-kernel
finish eager. This reduces host submissions; the arithmetic kernels and their
memory traffic remain the same. Eliminating all measured begin/middle host work is an unattainable
upper bound of 4.46 ms/verification; a partial reduction could credibly exceed
5% of whole-request decode time. This was the pre-implementation estimate;
matched end-to-end measurements below establish the actual benefit.

The implementation reuses the existing Graph implementation and HC/RMS/APPLY kernels,
adds dynamic residual and attention-boundary pointer patching, and specializes
middle-stage graphs for FP16 versus FP32 boundaries. It validates fixed tensor
storage/layout/device/dtype before replay; the existing Python binding signature
and native dynamic boundary checks stay intact. Graphs own no attention state,
KV pages, routing or collectives. No new architecture or math kernel is added.
The prototype was initially opt-in through a native setter. The final integration
enables it for validated GLM TP plans, with the opt-out described below.

Risks/gates: exact output at each stage across both GPU families and all existing
HC row limits; changing x/y storage and boundary dtype; rejection before mutation
when fixed storage/layout or exposed scratch changes; two-rank logits/live state
and MTP parity; three-rank local-stage parity; isolated cold-cache startup; repeat
graphs-off/on 128/1024/8192 timings and memory/plan/graph counters. GLM passes
these gates; the DeepSeek activation that failed the matching gate was removed.

### Prototype measurements and corrections

The first graph screen passed exact completions and acceptance in all six
pairs. At 128 outputs, eager IR median/range was 77.250 (76.789–77.414),
stage graphs 83.969 (83.052–84.930). Paired improvements were 9.94%, 9.35%,
7.28%. At 1024, the pairs improved 9.27%, 12.55%, 10.22%, despite both modes
entering a slower runtime state during later pairs. Aggregate medians for that
length would obscure this state change. These are graph-versus-existing-IR
comparisons, not graph-versus-stock comparisons.

A fresh process then warmed the full 8192-token request in both modes and ran
two alternating pairs at every requested length. All completions and acceptance
totals match, with no plan builds or graph captures in timed requests:

| Outputs | Eager IR median (range) | Stage graphs median (range) | Median paired change |
| ---: | ---: | ---: | ---: |
| 128 | 54.205 (53.045–55.365) | 59.554 (58.286–60.822) | +9.97% |
| 1024 | 57.009 (56.998–57.020) | 62.863 (62.849–62.876) | +10.27% |
| 2048 | 55.381 (53.773–56.989) | 62.915 (62.817–63.014) | +13.70% |
| 4096 | 56.428 (56.317–56.539) | 62.422 (62.351–62.493) | +10.62% |
| 8192 | 58.617 (58.538–58.696) | 64.576 (64.191–64.960) | +10.16% |

The two 8192 improvements are +10.67% and +9.66%; full-request wall medians
are 58.477 and 64.409. This process was already in the slower runtime state
at warmup. The paired gain is credible; comparing its absolute rates against
the earlier fast-process stock baseline is not valid. The larger 2048 effect
includes one slower eager sample, so approximately 10% is the more conservative
description of this campaign. Raw data are in `native-results/graphs/glm-long`.

The first graph-enabled 128-token warmup captures 90 graphs per rank (two for
each of 45 HC blocks), with 4230 replays. The long warmup replays 248760 graphs
per rank with zero captures. Driver free-memory snapshots around capture show
approximately 10 MiB extra on the peer and 12 MiB on the output device; the
subsequent 8192-token warmups show no additional driver-memory loss. These are
whole-process snapshots, not a precise allocation accounting of CUDA graph
internals. Existing stage tensors are reused. Each plan stores at most three
graphs: begin and two middle variants for FP16/FP32 attention output. Graphs
share the existing bounded shape-plan cache and are released with their plans.

Primitive tests cover both GPU families, supported row limits, changing input
addresses, alternating boundary dtypes, fixed-storage/layout and exposed-scratch
invalidation. Initial GLM TP2 tests passed exact logits, residual traces and
live state at B1..4Q1 and B1Q4, plus concurrent MTP B1..4. The validation wrapper
initially missed the upstream validator's inference-mode context; that failed
before graph execution and was corrected. An experimental global emitter hook
also retained its inline rank context after unload. That hook was removed from
the production integration, and final lifecycle tests verify memory release.

The final source adds pre-capture rejection for unsupported norm dtype/layout,
and uses the immutable captured device when destroying graphs after a caller
changes weight storage/device. These safety changes do not change the measured
replay path. The clean full build and final regression/performance campaign
passed with these safety changes included.

### Validated production integration

`native_tp.py` enables stage graphs only when constructing existing GLM
HC plans, only with a graph-capable extension, and only when
`EXL3_NATIVE_TP_GRAPHS` is unset or `1`. Setting it to `0` before process startup
keeps eager stages. This is subordinate to the existing `EXL3_NATIVE_PLAN=1`
switch. Older extensions keep the eager path. DeepSeek remains eager because
its long MTP benchmark did not satisfy the matching gate described below.
Single-GPU registration, Qwen, dense plans, fallback domains and upstream TP
scheduling are unchanged.

The benchmark helper now controls this construction flag and toggles cached
plans directly. It no longer installs a global emitter closure or changes the
single-device DeepSeek draft. Production diagnostics expose graph capture and
replay counts outside timed execution. A clean full build compiles all 194
translation units for sm89 and sm120 into an isolated output directory; no
package is installed.

The final host-timer comparison starts after 4096 outputs and observes the same
21 target verifications and 63 draft forwards in each mode. Output-rank
begin/middle inclusive host time falls from 84.87 to 42.59 ms; the peer falls
from 87.53 to 48.00 ms. Selection remains 30.78 versus 30.95 ms on the output
rank, and finish remains approximately 17 ms. Parent target-forward time falls
from 436.17 to 394.70 ms, while draft time remains approximately 51 ms.
This supports the intended launch-overhead mechanism. These instrumented
intervals overlap and are not final throughput measurements.

## Files and operating behavior

The final clean-binary performance pass, after narrowing activation to GLM,
uses a fresh process, full-length warmup, and two alternating pairs:

| Outputs | Eager IR median (range) | Graph median (range) | Median paired change |
| ---: | ---: | ---: | ---: |
| 128 | 73.029 (72.531–73.528) | 79.503 (79.478–79.527) | +8.87% |
| 8192 | 82.787 (82.751–82.824) | 90.015 (89.952–90.078) | +8.73% |

Both 8192 pairs independently improve by 8.76% and 8.70%; decode standard
deviations are 0.051 eager and 0.090 graphed. Full-request wall medians are
82.493 and 89.695. Every sample has identical completion hashes and the same
5428 accepted/2864 rejected draft tokens. Each rank records 124380 native block
executions and, with graphs enabled, 248760 replays. There are zero new plans or
captures during timing and only 45 expected prefill fallbacks per rank. Driver
free memory is unchanged across each full request.

The final windows remain faster: eager 81.56/81.45 versus graphed 88.20/88.30
tok/s. Excluding the initial prefill window, median window rates are 81.66 and
88.59; ranges are 75.36–91.59 and 82.28–99.60 as draft acceptance changes along
the same trajectory. The benefit is sustained, not just a startup effect.

Ordinary GLM decode also improves in its matched final process: 28.045 eager
versus 32.086 graphed median tok/s at 128 outputs, median paired +13.91%
(individual +20.21%, +13.91%, +13.38%). That process was in a slower runtime
state than the earlier ordinary baseline; cross-process absolute comparisons
are invalid. The primary result remains sustained MTP, not ordinary decode.

| File | Change |
| --- | --- |
| `exllamav3/exllamav3_ext/graph.cuh` | Append HC pointer parameter identifiers without renumbering existing identifiers |
| `exllamav3/exllamav3_ext/hc_mix.cu` and `.cuh` | Capture entry points for existing HC mix/apply kernels; preserve eager entry points and all math |
| `exllamav3/exllamav3_ext/libtorch/block_plan.cpp` and `.h` | Own bounded stage graphs, fixed-binding guards, dynamic pointer patches, boundary dtype variants, counters and device-safe destruction |
| `exllamav3/exllamav3_ext/bindings.cpp` | Expose graph enable/disable and diagnostic counts |
| `exllamav3/modules/native_tp.py` | Enable validated GLM HC plans at construction, retain eager DeepSeek and old-extension compatibility/opt-out, report graph counts |
| `tests/test_native_plan_graphs.py` | Eight GPU regressions, including every supported TP batch/query combination on both GPU families |
| `tests/test_native_tp_graphs.py` | Four construction-policy/legacy-extension/opt-out regressions, including eager DeepSeek |
| `util/profile_final_ir.py`, `util/summarize_final_ir.py`, `util/validate_native_graphs.py` | Isolated generation, instrumentation, paired summaries and explicit graph validation |

The build uses the existing setup script, MSVC 14.44, CUDA 12.8 and sm89/sm120.
The local invocation is preserved in `native-results/graphs/build_full.bat`;
the candidate is `native-results/graphs/full-extension/exllamav3_ext.cp313-win_amd64.pyd`.
The full build succeeded in 743.1 seconds. Its SHA256 is
`2ab9b32a6d222a37b14c8cd74211c109b9c317a85d91bfb3bdf67aa7294e2cc7`.
All 196 native/graph/dispatch/attention-export tests pass with zero skips, plus
the worker signal regression. Separately, all 25 TP policy/boundary tests pass
against the old tp2 extension with the updated Python code.

The clean binary's default-path GLM TP2 validation passes exact logits, block
traces and live state at B1Q1, B1Q4 and B2Q1, with concurrent B1/B2 MTP requests,
before and after reload. Each eight-step run reports 720 graph replays per rank.
Final unload plus garbage collection leaves 86,391,296 bytes allocated on the
parent GPU, versus approximately 83.5 GB before collection of the validator's
cycles. Before reload, the still-loaded draft intentionally remains resident.
The experimental emitter-context retention is absent from the final integration.

Final startup checks used new isolated Triton caches and separate fresh-process
restarts against the clean binary. Each process completed all 15 serialized
warmup passes, a generation warmup and two 32-token MTP requests:

| Model/path | TP ranks | Fresh-cache rank warmup | Restart rank warmup | Result |
| --- | ---: | ---: | ---: | --- |
| GLM stage graphs | 2 | 315.23 s | 6.48 s | Pass |
| GLM stage graphs | 3 | 318.79 s | 9.54 s | Pass |
| DeepSeek eager | 2 | 29.82 s | 5.51 s | Pass |

No reduction timeout, cache replacement race, startup deadlock or first-request
failure occurred. Timeouts and the user's normal compiler caches were unchanged.
These are startup checks, not sustained performance claims. DeepSeek's final
default-path short parity check also passes with zero graph replays.

GLM TP3 also passes the existing exact **local-stage** oracle, replicated-rank
checks, generator checks and stock-repeat rounding envelope. Whole-model logits
and live-state hashes are not bitwise identical across independent three-rank
runs; the unchanged CPU reducer accumulates in rank-arrival order. For B1Q1,
B1Q4 and B2Q1, graph/stock RMSE is 0.1914/0.1944, 0.1627/0.1513 and
0.3894/0.3515 respectively, where the second number is stock-versus-stock noise.
Maximum differences are 1.881/1.942, 1.018/0.931 and 6.885/5.640. The existing
validator's thresholds were not changed. Local graph arithmetic is exact using
the same reduced inputs, and every four-step run reports 360 replays per rank.

The feature requires the new extension to gain graph replay. Keeping an old
extension preserves eager behavior. No installed package has been replaced.
First use captures graphs once for each encountered plan shape/dtype; steady
requests replay them. This adds a small initialization and memory cost. Plan
selection checks, unsupported-mode fallback, speculative decoding and collective
order remain intact.

Example matched benchmark from this worktree, using the existing Python:

```powershell
$env:CUDA_VISIBLE_DEVICES = '2,0'
$env:PYTHONPATH = "$PWD;$PWD\native-results\graphs\full-extension"
& D:\tabbyAPI\venv\Scripts\python.exe util\profile_final_ir.py --model D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw --extension-dir native-results\graphs\full-extension --output native-results\my-graph-comparison --modes native graphs --mtp --lengths 128 8192 --runs 2 --warm-longest
```

The output directory must be new. `native` explicitly disables stage graphs in
this benchmark; `graphs` enables them. Ordinary native TP operation with the
new binary enables eligible HC graphs by default. Set
`EXL3_NATIVE_TP_GRAPHS=0` before starting the process to opt out.

## Rejected DeepSeek graph activation

The clean binary passes exact DeepSeek TP2 logits and live-state checks at
B1Q1, B1Q6 and B2Q1, plus concurrent short MTP requests, with 688 graph replays
per rank in each eight-step validation run. The short 128-token graph comparison
also matches completions and acceptance. However, the 1024-token pair diverges:
eager records 694 accepted/267 rejected, graph replay 675/275, with different
completion hashes. Zero new plans or captures occur during either timed run.
Those rates are not a valid speed comparison.

A separate process using the **unchanged old tp2 extension**, with graphs
unavailable, also fails repeatability across successive 1024-token eager MTP
requests: 681/262 versus 694/267 accepted/rejected, with different completions.
One eager-control completion hash matches the graph warmup's hash, and the other
matches the earlier eager timed request. The mismatch is therefore not specific
to graph execution. The root cause of this existing long-generation issue has
not been identified, and no speculative-decoding or state algorithm was changed.

Without a repeatable sustained matching benchmark, automatic DeepSeek graph
activation is rejected. Production construction and benchmark mode switching
now leave DeepSeek eager, with a dedicated regression test. The failed candidate
and old-binary control are retained under `native-results/graphs/final-deepseek-mtp`
and `deepseek-eager-control`; they are not omitted as outliers. GLM's fully
matched long results support retaining its graph path independently.

## Recommendations from the measurements

1. **The previous IR was not at its practical ceiling.** HC stage graphs recover
   8.73% in the final controlled sustained GLM comparison. This does not
   establish unlimited further gains; the remaining paths have less removable
   work and more state/communication constraints.
2. **Significant Python boundaries remain.** Attention/state and MLP/collectives
   dominate useful portions of execution, but their boundaries have correctness
   responsibilities. Combining them requires a larger integration project.
3. **Single-request GLM MTP verification was already fully native.** The gain
   comes from host submission overhead, with identical acceptance and rejection,
   not a new speculative method or newly covered verification calls.
4. **Broader coverage is a concurrency opportunity, not a demonstrated sustained
   B1 win.** Qwen multi-job verification and GLM B4Q4 still fall back. Existing
   expert kernels have a separate eight-row limit. Raising IR counters alone
   would not demonstrate that these workloads become faster.
5. **CPU dispatch still matters.** Stage graphs provide direct end-to-end evidence.
   Repeated signatures remain measurable, but their isolated measured upper
   bound does not justify removing binding safety checks or adding a new API.
6. **Substantial limits are outside IR.** MLP execution, rank waits, CPU-assisted
   collectives and drafting are visible in profiles. The data do not isolate
   pure PCIe transport as the sole bottleneck. The fast/slow runtime-state change
   remains unresolved and should be investigated separately from IR arithmetic.
7. **Further IR work should be selective.** Retain the validated GLM launch
   reduction. Further work should begin
   with a concrete workload showing substantial removable overhead; broader
   fusion, shape expansion and TP protocol redesign are not justified by the
   current evidence. A dedicated upstream TP/Windows scheduling investigation
   is more promising than speculative additions to the IR instruction set.
