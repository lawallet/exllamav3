# DeepSeek V4.1 native IR investigation

Validation in progress; no release has been installed yet.

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

R2 records three distinct observations: the resolved narrow MGEMM tile defect, whose
fix remains in this branch, and a later sustained-MTP repeatability failure
observed even with eager execution on older DeepSeek V4, plus a version-to-version
throughput change with different token trajectories and acceptance counts.
The latter was not a matched-work measurement of an IR regression. That historical
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
GiB and draft budgets [90,0,0]. The loader interprets these as requested
additional room for the component, caps that request by available memory,
and adds current reservations to set a cumulative allocator ceiling. They
are not guaranteed weight allocations. Target layers 0–29 live on logical 0, layers
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

The row-table implementation waits for prior pinned-buffer upload events before
reuse, resolves CPU row IDs, and gathers the requested disk rows. Small hinted
requests read inline; larger requests can use the prefetch worker. The measured
gather time includes host work and reads served by the OS page cache, so it is
not a direct physical-SSD latency or bandwidth measurement. Warm repeated
prompts can benefit from that cache naturally. Cold-cache random-read latency
has not been isolated, and no Engram table was moved to an application RAM mode.

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

## Graph implementation and validation

The graph extension captures the existing HC collapse plus RMSNorm begin stage,
and HC apply plus collapse plus RMSNorm middle stage. Attention, expert execution,
Engram, transfers, MTP exports and finish remain in their established paths.
It uses the existing `BC_BlockPlan` and upstream graph parameter machinery.
Carried pre-weight pointers and the first-stream/weighted-collapse scalar mode
are patched on every replay. Owned pre-weight outputs participate in fixed
binding checks. This is necessary because the first block starts without a
previous pre tensor, and later calls can change the input storage.

The graph candidate passes 223 shared native tests with no failures, errors or
skips, including both GPU architectures, dynamic carry pointer/mode transitions,
graph opt-out and existing TP/graph policies. Actual V4.1 weights pass another
215 specializations / 860 exact cases with graphs enabled. Two paired HTTP
audits per MTP setting match all recorded forward inputs/output hashes, tokens,
and draft acceptance, with 90,336 exact same-input boundary comparisons across
all 40 target and three draft blocks (`graph-http/audit/parity.json`).

Additional regression evidence: 360 Gemma actual-weight stages, 528 Devstral
stages, generic residual primitives, and 140 Nemotron assignment/guard tests
passed. These are targeted operation/integration checks; they do not constitute
full generation tests of every older model. Existing older-DeepSeek TP graph
restrictions remain intact.

The eager-only candidate is not the selected performance result. It improved
ordinary decode (10.787 to 12.377 median tok/s) but MTP samples ranged from
17.836 to 21.921 tok/s, with a median of 18.685 versus stock 20.066. Exact outputs
and allocator stability did not identify a correctness failure. That timing
variation prevents recommending the eager-only change for the primary MTP
workload. Graph and eager comparisons, device clock telemetry, longer contexts,
and sustained generation are evaluated separately below.

## Controlled short-context results

The graph candidate's completed `graph-http/timing` cohort has three warmed
512-token samples for each configuration, with alternating order and exactly
matching tokens within each MTP mode. Prompt occupancy is 177, cache capacity
65536. MTP acceptance is 275/78 in every paired sample.

| Mode | Stock median (range) tok/s | Native graphs median (range) tok/s | Median change |
| --- | ---: | ---: | ---: |
| MTP disabled | 10.743 (10.737–10.962) | 13.706 (13.494–13.992) | +27.6% |
| MTP enabled | 20.012 (19.714–20.364) | 22.373 (22.173–22.551) | +11.8% |

A separate alternating native eager/graph MTP cohort gives 21.692 versus
22.649 median tok/s (+4.4%), with wider ranges of 18.304–21.751 and
21.927–25.274. This isolates graph replay within the same implementation,
but its variability makes the primary stock/graph cohort the clearer result.
Telemetry records changing WDDM clocks: the 4090D median sampled memory clock
is 810 MHz, and the 6000's is 7001 MHz, with bursts to 10501/13365 MHz.
That is an observed environmental variable, not proof of the cause of any
individual slow sample. No power, clock or driver policy was changed.

Native warmup adds approximately 23.7/8.0 MiB of live tensor allocations on
logical GPUs 0/1 across ordinary and MTP shape creation. These are observed
before/after warmup deltas, not a complete isolated accounting of CUDA graph
driver allocations. The final 163 plans own 326 captured stages. Warmed timed
requests hold allocated memory at approximately 91260.5/31359.6 MiB and
reserved memory at 94604/32758 MiB. Driver telemetry stays at 95311/33191 MiB
through the principal timing cohort. CUDA-reported free memory (1380/5546 MiB)
differs from NVML physical free memory under WDDM; neither is presented as
the other. No capture count or persistent allocation growth occurs during
the warmed repeated short requests.

## Sustained repeatability

`graph-http/r2-audit` contains two stock and two native-graph 1024-token MTP
requests. All target and draft forward inputs/output hashes, final IDs, and
569 accepted / 155 rejected drafts match exactly. Native boundary comparisons
add 234,780 exact results. This does not reproduce the historical R2 divergence
on the tested V4.1 layer split. It does not close R2 for older DeepSeek TP,
whose guard remains unchanged. Audit timings are excluded from throughput.

The separate uninstrumented 1024-token cohort repeats three paired requests:
stock median 17.973 tok/s (17.595–18.360), native graphs 19.593
(18.484–20.093), a 9.0% median improvement. All six trajectories and acceptance
counts match. Native sample variation is greater here than in the 512-token
cohort, and the last pair is nearly tied; the median is not a guaranteed
per-request gain. No timing outlier was discarded.

## Populated-context measurements

The longer prompts contain numbered Python functions and are freshly prefilled
for every sample; recorded cached prompt tokens are zero. Long-context round
zero is retained as context-specialization warmup, because new attention
specializations compile on first use. At **8192 actual prompt tokens**, the
two subsequent 512-token generations per configuration give:

| Mode | Stock median (range) tok/s | Native graphs median (range) tok/s | Median change |
| --- | ---: | ---: | ---: |
| MTP disabled | 10.466 (10.383–10.549) | 13.037 (12.952–13.123) | +24.6% |
| MTP enabled | 19.862 (19.700–20.023) | 22.015 (21.346–22.683) | +10.8% |

All six outputs within each MTP mode, including warmup, match exactly. The
all-three-round MTP medians are 19.700/22.473 (+14.1%); the more conservative
warmed-only comparison above is used in the final summary. Each MTP sample accepts
306 and rejects 94 drafts. Prefill generally takes 15.3–15.8 seconds; the first
stock non-MTP sample takes 13.25 seconds. This implementation does not target
large-row prefill, and no prefill acceleration is claimed. Native plans and
captures remain at 163/326. Prefill grows general allocator reservations and
NVML memory on the 6000 to about 97233 MiB, very close to its physical limit.
That headroom constraint motivates the separate placement experiments.

At **64511 actual prompt tokens**, leaving room for 512 generated tokens in
the same 65536-token cache, both warmed post-warmup samples also pass:

| Mode | Stock median (range) tok/s | Native graphs median (range) tok/s | Median change |
| --- | ---: | ---: | ---: |
| MTP disabled | 9.176 (9.066–9.286) | 10.152 (10.145–10.158) | +10.6% |
| MTP enabled | 16.177 (16.127–16.228) | 18.242 (18.166–18.318) | +12.8% |

All six outputs per mode match exactly, including warmup. Every MTP request
accepts 288 and rejects 91 drafts. Prefill is approximately 101 seconds, with
the first stock sample taking 104.17 seconds. These times are reported separately
from decode. The native cache remains 163 plans / 326 captures, with no allocation
failure across the full long-context campaign. Near-capacity physical use remains
a deployment headroom concern even though these requests complete successfully.

## Placement investigation

Requested split budgets do not translate directly into block placement. Budgets
94, 90 and 86 GiB all place target blocks 0–29 on the 6000 and 30–39 on the
4090D. The per-load budget is capped after already-loaded vision, draft and
pinned attention allocations. The diagnostic harness records each requested
budget, prior reservation and resulting allocator limit; reported CUDA budgets
also vary between placements under WDDM. Physical NVML free memory must not be
substituted for the loader's effective budget.

The 90/86 runs are therefore unchanged-placement controls, not weight-movement
successes. Their three-repeat short MTP medians are respectively stock/native
19.622/22.353 (+13.9%) and 19.924/22.066 (+10.8%), with exact tokens and
275/78 acceptance throughout. Budget 82 first produces a real 29/11 block split,
moving about 2.39 GiB of live allocation to the 4090D and freeing 2.51 GiB on
the 6000 at load. These allocation deltas include device-dependent workspaces;
they are not claimed to be a byte-for-byte accounting of weight tensors alone.

The 29/11 split's short MTP medians are 17.980 stock (17.594–19.655) and
21.375 native graphs (20.366–22.111), +18.9% within that placement. Its six
timed outputs match each other, with 281/74 acceptance, and its 128-token
audit matches all recorded target/draft outputs and native boundaries. The
changed placement changes the longer token trajectory relative to the baseline
275/78 workload. Cross-placement throughput is consequently an application
workload comparison, not matched-token evidence of a pure placement speedup.

The completed short-context matrix has three paired 512-token MTP samples per
row, preceded by an exact 128-token audit. All within-row token sequences,
acceptance counts and audited forward outputs match.

| GPU0 budget | Target blocks 6000/4090D | Added live GiB on 4090D | Stock median tok/s | Native median tok/s | Same-placement IR gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 94 baseline | 30/10 | 0 | 20.012 | 22.373 | +11.8% |
| 82 | 29/11 | 2.387 | 17.980 | 21.375 | +18.9% |
| 80 | 28/12 | 4.773 | 19.891 | 23.693 | +19.1% |
| 78 | 27/13 | 7.285 | 20.347 | 23.900 | +17.5% |
| 76 | 26/14 | 9.671 | 21.852 | 27.344 | +25.1% |
| 72 | 25/15 | 12.183 | 23.128 | 26.628 | +15.1% |

GPU1's requested budget remains 46 GiB, GPU2 zero, and the draft budget
[90,0,0]. Budget 76 is the best candidate for installation validation. Its
stock range is 21.514–21.992 and native range 27.147–27.425 tok/s, with 288/77
acceptance throughout. Budget 72's native range is 26.490–26.862 and stock
21.172–23.735, with 283/94 acceptance. Shifting the most weight does not win.

On budget 76, warmed live allocation is 80.316/40.306 GiB and reservation
83.521/42.264 GiB on the 6000/4090D; telemetry reports 86227/43715–43717 MiB.
Budget 72 raises 4090D driver use to 47113–47115 MiB, leaving less headroom.
The sampled 4090D memory clock median rises from 810 MHz in budgets 80/78 to
5001 MHz in 76/72. This correlates with the throughput change, but clock policy
was not independently controlled, so it is not established as the sole cause.
These results do not justify attributing the entire cross-placement gain to IR.

## Reproduction and interpretation

`util/deepseek_v41_campaign.py` starts the isolated authenticated Tabby server,
runs deterministic requests, and stores before/after status and generator
statistics. `--compare --graphs` alternates stock/native order each round;
`--compare-eager` isolates eager versus graphed native execution. `--audit`
adds exact stock oracles and target/draft output hashes and must not be used
as throughput evidence. `--telemetry` samples NVML in a separate process.
`util/verify_deepseek_v41_cohorts.py` checks complete token sequences and
acceptance within each context/MTP group. `util/summarize_deepseek_v41.py`
summarizes all recorded timing rounds; `--skip-first` separately produces
`summary-warmed.json` for context-specialization warmup exclusion.

Primary evidence lives under `native-results/deepseek-v41/graph-http`, including
`audit`, `timing`, `eager-ab`, `r2-audit`, `r2-timing`, `context8k` and `context65k`.
The JSON records include actual prompt occupancy, uncached prompt length,
prefill/generation durations, IDs, acceptance, allocator metrics, placement,
Engram mode and graph counters. Raw artifacts are kept locally outside Git;
the committed report and tools document their interpretation and reproduction.

The measured workload is one deterministic tutorial prompt per context length,
batch one, with three speculative tokens. Gains are not universal across
prompts, sampling, concurrency or TP. Larger batches and unsupported bindings
fall back to stock; V4.1 TP is not enabled by this optimization. Historical
older-model restrictions and native MoE assignment guards are unchanged.
Remaining opportunities include reducing repeated Python binding checks and
carefully extending attention graph coverage around shared pool/index state.
Those need separate correctness and profiling work. No speculative algorithm,
quantization arithmetic, expert assignment scheme or Engram storage mode was
changed to obtain the reported gain.

The final release gate reran all 223 shared native tests on both GPU architectures
with zero failures, errors or skips (`final-tests.json`). Against the same final
extension, 414 older-DeepSeek actual-weight specializations, seven GLM production
primitive shapes and 140 Nemotron tests also passed. Earlier Gemma/Devstral
actual-weight checks exercise unchanged v2 code; full older-model generation
campaigns were not repeated. The final native extension SHA-256 is
`12cd3cbf117c5782065651cbcafa62b7b1798178f174f57b14f46c7d715d0139`.

## Runtime selection

The existing opt-in remains `EXL3_NATIVE_PLAN=1`. With that enabled, the new
V4.1 carried-HC binding selects graphs by default after the validation gates
above. `EXL3_NATIVE_DSV41_GRAPHS=0` selects the diagnostic eager variant;
`EXL3_GRAPHS=0` remains the upstream global graph opt-out. Setting
`EXL3_NATIVE_PLAN=0` restores stock execution. No persistent Tabby configuration
or launcher is edited by the experiment.

The native guard requires the exact V4.1 architecture and carried-HC semantics,
batch one, query lengths 1–5, contiguous four-stream FP32 residuals of width
5120, supported norms and CUDA bindings, and non-TP ownership. Conversion,
prefill, unsupported overrides, LoRA bindings and unsupported shapes fall back
before native mutation. Cached bindings are invalidated when tensor storage or
supported scalar attributes change. The carry ABI and graph ABI are separately
checked, so an older extension does not silently execute the new opcode.
