# DeepSeek V4.1 native IR investigation

Work in progress; no optimization or release is validated yet.

## Provenance

The feature branch `feature/native-ir-deepseek-v41` starts at
`fde10e2e6af1edfce4ea17a8ea78ca2e5adb7e67`, the existing v1.6.6 native IR/MoE
integration. Its upstream baseline is v1.6.6 (`bd1f5191`). The pre-existing
`convert.py` edit and unrelated untracked files are preserved. A copy of that
edit and the initial status are under `native-results/deepseek-v41/prework`.

## Architecture findings

The exclusion is capability based, rather than an architecture-name cutoff:
`native_deepseek.register`, `native_deepseek.get_native_block`, and native TP
export/selection reject `carry_pre`. The existing GLM guards also reject it.
This protects against applying same-site collapse to a model that collapses
with the previous site's weights. Cached-plan checks occur after the guard.

The checkpoint has four FP32 residual streams of width 5120; the existing
DeepSeek HC plan accepts width 4096 and collapses to FP16. V4.1 explicitly
requires FP32 collapse before RMS normalization because massive activations
can exceed FP16 range. Removing only the carry guard would therefore be wrong
in multiple ways.

V4.1 additionally shares compressed KV pools and index selections across
layers, uses latent-derived index keys and candidate block selection, omits
per-head query normalization, and has two Engram injection layers (1 and 14).
Layer-split attention is pinned to one GPU, even for transformer blocks whose
residual and experts live on another. A native block must preserve the stock
attention input/output transfers; the old native block path assumes colocation.

MTP uses the DSpark block drafter (`dflash_draft` generator path), with three
draft tokens in this experiment. Its expert count/top-k are 128/3, versus
384/6 for the target. It also carries pre weights. Target state taps are at
the outputs preceding layers 37, 38 and 39. Python must retain ownership of
state export, shared pools, verification, and rewind.

Native MoE assignment is independent of the main IR flag but is currently
restricted to Nemotron, width 1024, top-k 22, and 12/16 rows. It is not active
for this checkpoint. Upstream already uses fused expert execution for the
small target/verification rows, avoiding the assignment sort and host count
readback of its larger-row path. Existing attention graphs and shared-expert
graphs remain enabled with the native IR flag off. However, the whole-step
attention graph explicitly declines V4.1 pool readers, latent-index pool
owners, and candidate selections: most of V4.1 attention therefore remains
eager, even though its projection kernels and small-row experts are optimized.

R2 contains two separate issues: the resolved narrow MGEMM tile defect, whose
fix remains in this branch, and a later sustained-MTP repeatability failure
observed even with eager execution on older DeepSeek V4. That historical
evidence does not establish either failure or correctness on V4.1. Repeated
stock V4.1 runs and exact same-input stage comparisons are required before
considering HC graph activation. The older architecture's graph restriction
must remain unchanged.

## Experiment controls and exclusions

All runtime commands use `D:/tabbyAPI/venv/Scripts/python.exe`. The isolated
Tabby server binds localhost port 5008 and uses existing authentication.
Persistent configuration and credentials are not edited. Visibility is
`2,0,1`: logical 0 is the RTX PRO 6000, logical 1 the first 48 GiB 4090D,
and logical 2 the unused 4090D. No TP or CPU expert offload is introduced.
Both Engram tables report `fp16_disk`, with `EXL3_NGRAM_STREAM=1` and
`ngram_ram=false`. Vision and MTP load successfully. Context capacity is
65536; actual prompt occupancy is recorded separately.

The first load attempt reserved only 6 GiB for the draft component and was
rejected by the loader. The successful load uses target budgets [94,46,0]
GiB and draft budgets [90,0,0]; budgets are cumulative allocator ceilings,
not extra weight allocations. Target layers 0–29 live on logical 0, layers
30–39 on logical 1, and all target attention on logical 0. Initial reported
driver memory was 95011/33143 MiB on physical GPUs 2/0.

The raw-completion calibration runs in `baseline-load2` are **excluded from
performance claims**: omitting the chat delimiters made the model repeat the
request and produced approximately 99% MTP acceptance. The corrected workload
uses the checkpoint's begin/user/assistant delimiters and closed think tag.
These failed experiment records are retained rather than overwritten.

## Initial controlled baseline and bottlenecks

The corrected short HTTP prompt occupies **177 tokens** after text API
roundtrip. Three warmed greedy generations of 512 tokens per mode, with fresh
generators and no prefix-cache reuse, gave:

| Mode | Median tok/s | Range | Accepted/rejected drafts |
| --- | ---: | ---: | --- |
| Stock, zero draft tokens (see correction below) | 10.397 | 10.209–10.527 | 0/0 |
| Stock, MTP on | 19.545 | 19.513–19.615 | 275/78 each run |

All token IDs repeat exactly within each mode. IDs differ between MTP off/on;
these are distinct numerical/query-shape workloads, not an exact-token MTP
speedup experiment. Native comparisons must keep MTP setting fixed.

Separate cProfile samples begin after 64 generated tokens. In the MTP sample,
HC mixing consumed 0.407 s and HC apply 0.113 s of 2.627 s measured host time
(19.8% combined). All RMSNorm calls together consumed another 0.191 s (7.3%;
includes norms outside HC). The common tensor-workspace cache, views, and
Python dispatch contribute substantial cost. These inclusive host timings
are not independent GPU latency measurements and must not be summed across
parent/child frames. Profiling overhead makes them unsuitable throughput data.
The disk gather extension consumed 0.073 s (2.8%) in that MTP sample and
0.115 s (2.1%) in the ordinary sample. SSD gathers are synchronous small-row
random-read staging, but this evidence does not identify disk bandwidth as
the primary bottleneck. Streaming remains enabled throughout.

Short CUDA traces cover 32 ordinary target forwards and 13 MTP target
verification forwards. Summed actual kernel/copy durations (excluding GPU
annotation spans) are 1976.709 ms ordinary and 477.332 ms MTP. Ordinary GPU
activity is about 51.6% expert kernels, 39.6% GEMM/GEMV, 4.4% HC and 1.3%
normalization; MTP is about 56.3%, 32.2%, 4.0% and 1.5%, respectively.
Kernel categories reflect symbol names and are approximate: GEMM includes
attention projections and output projections; this is not a semantic model
stage partition. Copies account for about 11.4 ms ordinary and 4.4 ms MTP;
their kernel time does not capture the full host synchronization penalty.
Profiling disturbs submission and wall latency substantially, so these are
diagnostic activity shares, not claimed uninstrumented speed limits.

The first candidate therefore extends the existing BC_BlockPlan with carried
pre weights and FP32 collapse, using the unchanged stock HC/RMS kernels.
Python retains both attention transfers, MTP state export, and all stateful
operations. It is restricted to this architecture, B1, queries 1–5, and exact
supported bindings. Existing DeepSeek V4/GLM behavior and their guards remain.
The new plan owns only small row-local workspaces; measured allocation costs
and full numerical gates are pending. Graph replay is a separate next gate,
not yet enabled in the first implementation.

The initial zero-draft-token control retained upstream DSpark cache refresh
and hidden-state export. It is therefore not the final fully-disabled-MTP
baseline. The corrected `eager-http2` controls temporarily detach the draft
model/cache while creating each non-MTP generator, then restore the container's
loaded component references. The generator itself retains no draft model.
This keeps physical weights/placement fixed while eliminating all draft
execution. The MTP-enabled baseline above is unaffected. An early audit client
also assumed non-streaming HTTP `usage` was populated; this Tabby version
returns null there, so exact output length is verified from generator diagnostics.

## Eager correctness checkpoint

The first candidate extension built successfully for sm_89 and sm_120.
`eager-tests.json` records 221 shared native tests, zero failures/errors/skips.
This includes new exact carry-stage cases on both GPU architectures and the
existing graph policy, TP, deferred residual, Qwen, GLM, Gemma and Devstral
guard/regression tests. Actual V4.1 checkpoint HC weights passed 215 target/
draft specializations and 860 exact cases, including absent/present carry and
large FP32 activations. Older DeepSeek V4 passed 414 actual-weight stage
specializations; the seven GLM production primitive shapes also passed.

The first full HTTP audit pairs in `eager-http2/audit` match exactly in recorded
target/draft outputs and final token IDs. Ordinary generation executes all 40
target blocks and checks 30,720 native boundary results against stock with the
same inputs. MTP executes all 43 target/draft blocks and checks another 14,448
results. Both MTP paths accept/reject 72/11 drafts in these 128-token audits.
These instrumented timings are excluded from performance claims.
