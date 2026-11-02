# DeepSeek V4.1 native IR investigation

The opt-in OpenMP policy is committed and the separately versioned
`1.6.6+nativeplan.v2.tp.moe1.ds41.1.omp1` wheel is installed in `D:/tabbyAPI/venv`.
Final fresh installed-wheel throughput and unset-compatibility checks have passed.
New evidence is under `native-results/deepseek-v41/omp-policy/`; the earlier
release wheel and all prior evidence remain preserved. The unset/default
behavior and persistent Tabby configuration are unchanged. The entries below
retain the investigation's intermediate findings as well as release results.

The original release follow-up reproduced 27.34 tok/s: three fresh processes give pooled native
median 27.327 versus stock 23.427 tok/s (+16.65%), with exact paired outputs.
Controlled CPU runtime interventions reach 33–34 tok/s without changing the IR
or arithmetic. The six follow-up sections below separate established effects
from the still-unresolved trigger of the historical slow runtime state.

## OpenMP release recommendation and installed decode result

Use **`EXL3_OMP_BLOCKTIME_MS=0` for this system's GPU-heavy DeepSeek V4.1
workload**. It preserves 56 Torch threads, exact outputs/MTP acceptance, native
IR graphs and SSD Engram streaming. The five-run 0/1 sweep differs by only
0.36% in median throughput; later fresh repeats sometimes favor 1 by 0.4–1.8%.
Zero is the lower-CPU choice, not a claim that it wins every timing sample:
separate decode telemetry measured about 1.1 cores at zero versus 3.4 at 1 ms
and 43 at 50 ms. Fifty milliseconds retains substantial worker spinning.

Keep the feature **opt-in**. For many short parallel CPU operations, choose
**1 ms or leave the setting unset**: the CPU control found approximately a
threefold wall-time regression at zero for repeated small parallel regions.
Large matrix multiplication and two CPU submitters remained parallel and exact.
No known-good CPU-expert-offload configuration was available; these controls
are not a substitute for a real offloaded-model benchmark.

The final installed-wheel short benchmark uses the unchanged 177-token prompt,
512 outputs, `[76,46,0]` / 26+14 blocks, MTP3, 65536 capacity and graphs.
Warmup and monitored requests are excluded from this table.

| Release / setting | Mode | n | Median tok/s | Range | Sample SD | Sample variance |
|---|---|---:|---:|---|---:|---:|
| Prior installed `ds41.1`, default 200 ms | Native | 3 | 26.828 | 26.604–26.958 | 0.179 | 0.0321 |
| New installed `ds41.1.omp1`, 0 ms | Native | 5 | **33.162** | **32.996–33.366** | 0.179 | 0.0322 |
| Prior installed `ds41.1`, default 200 ms | Stock | 3 | 21.407 | 21.072–22.446 | 0.716 | 0.5132 |
| New installed `ds41.1.omp1`, 0 ms | Stock | 5 | **29.648** | **29.253–29.944** | 0.290 | 0.0842 |
| New installed `ds41.1.omp1`, unset / 200 ms | Native | 2 | 27.086 | 26.982–27.190 | 0.147 | 0.0217 |
| New installed `ds41.1.omp1`, unset / 200 ms | Stock | 2 | 23.219 | 23.145–23.293 | 0.105 | 0.0110 |

The nearby old/new comparison is **+23.61% native decode** and **+38.50% stock**.
These are fresh-process release measurements, supported by the controlled
within-process setter sweep; they do not explain every source of process
variance. Every short sequence and the 288/77 draft counts are exact. Native
IR remains independently beneficial after the CPU policy change.

The same-wheel unset control gives **+22.43% native** for 0 ms versus unset.
Unset reports requested `null`, actual 200, and no setter application. Its stock
rate also illustrates the remaining process variability; do not treat the
38.50% old/new stock comparison as a universal policy gain.

Cross-model unmasked short medians below use three 256-token runs per setting
and each model's established placement/capacity. MTP is enabled where supported;
Gemma and Devstral use ordinary decoding. Full ranges, variances, 4K prefill,
concurrency and masked cases are in
[the measured tables](../native-results/deepseek-v41/omp-policy/measured-tables.md).

| Model / mode | 200 ms | 0 ms | 1 ms | 50 ms | Interpretation |
|---|---:|---:|---:|---:|---|
| Qwen 3.8 Flash Next | 103.876 | 104.601 | 104.994 | 104.667 | Small unmasked change; CPU saving |
| GLM 5.3 Flash, layer split | 58.096 | 59.020 | 58.597 | 58.872 | Small unmasked improvement |
| GLM 5.3 Flash, native TP | 80.079 | 82.833 | 82.586 | 82.131 | Short gain; substantial independent process variance |
| Nemotron 3.5 Super VL | 109.977 | 110.907 | 111.240 | 110.713 | Small unmasked change; B3 native MoE remains functional |
| Older DeepSeek V4 Flash | 84.182 | 89.675 | 90.827 | 88.831 | Gain with wide default variability |
| Gemma 4 | 37.075 | 36.592 | 36.703 | 36.251 | No established decode gain |
| Devstral 2 | 12.592 | 12.605 | 12.573 | 12.552 | Effectively unchanged decode |

Request constraints matter. With minimum-length EOS suppression, 0 ms improves
the two-run Nemotron median from 83.931 to 105.967 tok/s and GLM TP from 58.591
to 80.201. Gemma/Devstral instead save CPU without a clear decode gain. Fresh
five-round B2 rechecks do not reproduce the initial Qwen/older-DeepSeek
concurrency slowdown; zero's aggregate medians exceed default in both. Do not
generalize a no-stop benchmark to every serving request or arrival pattern.

The product implementation is a reusable, process-local selection, applied once
on each actual model/generator/TP execution thread through the loaded kmp API.
It verifies readback, never loads a second OpenMP runtime, does not change Torch
thread counts, and preserves unset behavior. Unsupported runtimes warn once
only when explicitly selected. No kernels, arithmetic, MTP algorithms or native
correctness guards changed. See [activation and API details](openmp_wait_policy.md).

Implementation: `f30d73e37e9396fa71b28297a030df2ba9898416`.
Wheel: [Windows CPython 3.13 wheel](../native-results/deepseek-v41/omp-policy/package/wheels/exllamav3-1.6.6+nativeplan.v2.tp.moe1.ds41.1.omp1-cp313-cp313-win_amd64.whl).
All 645 installed code payloads and the unchanged SM89/SM120 extension match
the package proof. Other package versions, persistent Tabby configuration,
authentication files and the pre-existing `convert.py` edit are unchanged.
The original validated wheel remains available for rollback.

Regression evidence includes 234 native/policy tests, real-runtime checks of
unset/0/1/50 on different execution threads, all seven staged model configurations,
and **720 saved request outputs across 50 workload cohorts**, including final
installed validation. Non-Windows fallback is unit-tested, not hardware-tested; multimodal
checkpoints were exercised through text generation. No new models were downloaded,
converted or configured for CPU offloading. GLM TP retains unexplained rate
variation even with fixed policy, identical plans/tokens/MTP counts and similar
GPU clocks, including within the old-wheel process. No clock, WDDM or scheduling
mechanism is claimed as its established cause.

Installed long-context validation passes with exact outputs: 8K prefill
**3.830 s**, decode **31.749 tok/s**; 64511-token prefill **27.462 s**, decode
**28.101 tok/s**. These are warmed single verification samples after separate
warmups; the earlier sweep supplies the repeated prefill comparison. No material
prefill or allocation-stability regression was observed. The separate installed
telemetry pair records **1.16 CPU core-equivalents** for native and **1.12** for
stock, with 56 Torch threads retained; native monitored decode is 33.160 tok/s.
The fresh unset-control check passed with exact outputs and the original 200 ms
policy. All temporary benchmark servers and monitors have exited; validation
port 5008 is free. The new wheel remains installed, with the option unset until
explicitly selected for an application process.
[Raw evidence and reproducible commands](../native-results/deepseek-v41/omp-policy/README.md).

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

| Mode | Stock median (range) tok/s | Native graphs median (range) tok/s | Median change | Initial live allocation increase |
| --- | ---: | ---: | ---: | --- |
| MTP disabled | 10.743 (10.737–10.962) | 13.706 (13.494–13.992) | +27.6% | approximately 3.3 MiB for Q1 plans |
| MTP enabled | 20.012 (19.714–20.364) | 22.373 (22.173–22.551) | +11.8% | approximately 32 MiB total after all tested shapes |

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

## Release artifacts and installation

- Branch: `feature/native-ir-deepseek-v41`.
- Eager compatibility commit: `e19e0ec4050b67b49311d8064846449cac7426b1`.
- Graph implementation commit: `81987208`.
- Release source commit: `6f4a7208c10d36d3a32e906bb90a248c5d435fa8`.
- Installed version: `1.6.6+nativeplan.v2.tp.moe1.ds41.1`.
- Wheel: `D:/exllamav3/dist/native-ir-deepseek-v41/exllamav3-1.6.6+nativeplan.v2.tp.moe1.ds41.1-cp313-cp313-win_amd64.whl`.
- Wheel SHA-256: `8cae2ec51d8a7e73276447d7cb4a970e1bd0a529f32a835439cd1e5b8950c139`.
- Compiled architectures: sm_89 and sm_120; Python 3.13, Torch 2.9.0+cu128,
  CUDA 12.8, MSVC 2022, compressed extension.

The wheel was built with the existing project build procedure, preserving the
exact tested extension and refreshing Python sources. ZIP payload validation
checks every tracked packaged Python/C++/CUDA source against the committed
checkout, and the extension against the tested binary. After `pip install
--no-deps --force-reinstall` in `D:/tabbyAPI/venv`, every checked installed
payload file matches the wheel byte-for-byte. Proof files are beside the release
wheel. No dependency packages or other Python environments were modified.

The installed startup proof (`installed/startup-proof.json`) confirms package
and extension imports from Tabby's `site-packages`, normal environment-selected
IR activation without a control toggle, 326 captures / 4902 replays, loaded
vision, both `fp16_disk` Engram tables, and 128 exact tokens against the
same-placement stock audit. Persistent `config.yml` and `api_tokens.yml` hashes
remain unchanged, and the pre-existing `convert.py` matches its saved copy.

The previous wheel is preserved at
`D:/exllamav3/dist/native-ir-tp-moe-1.6.6/exllamav3-1.6.6+nativeplan.v2.tp.moe1-cp313-cp313-win_amd64.whl`
for package rollback in the same Tabby environment. The release source commit
above identifies the wheel's code; subsequent commits contain validation tools
and the completed report rather than changes to that packaged code.

## Installed-wheel validation

The fresh installed process uses target budgets [76,46,0], yielding 26/14 target
blocks, with all target attention still on the 6000. Two stock and two native
1024-token audits match all target/draft output hashes, token IDs and 589/179
acceptance; 224,460 native boundaries match exactly. This independently repeats
the sustained correctness gate at the selected placement.

The installed short-context timing cohort does **not** reproduce the earlier
27.344 tok/s absolute rate. Its three-sample stock median is 17.951
(15.498–19.711), native 20.412 (19.934–24.522), a 13.7% median improvement.
Each paired native request is faster, but both paths vary considerably between
processes and runs. No slow sample is removed from this short-context summary.
The earlier 27.344 figure is a measured candidate-process result, not a promised
runtime rate. The supported conclusion is a repeated same-placement IR benefit;
precisely ranking placements under fixed clock/CPU scheduling conditions remains
an opportunity for further investigation.

At 8192 tokens, the two warmed installed samples give stock median 19.407
(18.021–20.794), native 22.617 (19.664–25.570), +16.5%. All six outputs including
warmup match, with 288/122 acceptance. Prefill is 3.69–3.85 seconds after warmup,
versus approximately 15.5 seconds at the near-full baseline placement. Both
stock and native benefit, so this is reported as a placement/environment effect,
not an IR prefill improvement. The mechanism was not isolated with paging or
fixed-clock measurements. Telemetry at 8K is about 89259/45285 MiB on the
6000/4090D, providing a more balanced physical-memory margin.

At 64511 tokens, the two warmed installed samples give stock median 17.812
(17.659–17.964), native 20.280 (19.769–20.790), +13.9%. All six outputs match,
including warmup, with 288/81 acceptance. Warmed prefill is 27.32–27.72 seconds
in both modes, versus approximately 101 seconds at baseline placement. This
large placement-related prefill difference is separate from native decode gain.

Final live allocation is 80.344/40.307 GiB and reservation 86.482/44.293 GiB;
request peak live allocation is 81.675/41.611 GiB on the 6000/4090D. NVML stays
around 89261/45795 MiB through the warmed near-65K cohort. The native cache remains
163 plans / 326 captures. Engram remains `fp16_disk`, vision and MTP stay loaded,
and no allocation failure or continued plan growth occurs. The temporary port
5008 validation server was stopped after the campaign; only the verified wheel
installation remains.

| Installed MTP workload, split [76,46,0] | Stock tok/s | Native tok/s | Improvement | Native VRAM cost / warmed live GiB, 6000/4090D |
| --- | ---: | ---: | ---: | --- |
| 177-token prompt, 3 samples | 17.951 | 20.412 | +13.7% | shared ~32 MiB footprint; 80.316 / 40.306 |
| 8192-token prompt, 2 after warmup | 19.407 | 22.617 | +16.5% | same footprint; 80.328 / 40.307 |
| 64511-token prompt, 2 after warmup | 17.812 | 20.280 | +13.9% | same footprint; 80.344 / 40.307 |

The native tensor footprint was measured during initial shape creation at about
32 MiB combined; it persists when switching back to stock in the same process.
Consequently paired warmed allocation deltas are near zero and must not be
misrepresented as zero implementation cost. CUDA graph driver allocations are
additional and not fully isolated from other warmup allocations. Ordinary decode
at the baseline split remains independently validated at 10.743 stock versus
13.706 native (+27.6%); the installed selected-split timing focus is MTP.

## Final diagnostic profiles

Installed profiles cover generation after token 64, with complete iterate call
scopes, and exclude exact-audit hooks. Stock ordinary HC mix/apply costs are
2.573/0.704 s out of 19.451 s profiled host time; MTP costs are 1.251/0.348 s
out of 9.520 s (16.8% combined). Stock MTP all-RMSNorm cost is 0.294 s.
The native MTP profile has 6.936 s total host time and 1.728 million calls versus
1.799 million stock; these instrumented totals are diagnostic, not throughput
benchmarks. Some decorated forward frames still have inconsistent cumulative
accounting, so no attention/MLP wall-time partition is inferred from those frames.

Actual disk gather calls cost 0.224 s stock and 0.221 s native in matched MTP
samples; true non-MTP gather costs are 0.347/0.348 s. Streaming was not removed.
Tensor `.cpu()` call count stays 277, `.to()` stays 5093, and the native binding
guard costs 0.493 s cumulative. This leaves substantial dispatch, transfer and
readback work outside the new HC graph boundary.

Matched CUDA traces cover 15 target iterations and 15 DSpark draft forwards,
counted from first-attention-block annotations. `cudaLaunchKernel` calls fall
from 17072 to 12557 (4515 fewer, 26.4%), while `cudaGraphLaunch` rises from 30
to 1320. The 1290 added graph launches equal two HC stages × 43 blocks × 15
iterations. Cooperative launches remain 2429; `cudaMemcpyAsync` remains 1591,
stream synchronizations 86, event synchronizations 60 and device synchronizations
four. No structural copy or synchronization elimination is claimed.

Summed traced GPU kernel/copy activity is approximately 677.1 ms stock and
540.9 ms native. Expert/GEMM activity remains dominant (roughly 60%/29% stock,
58%/31% native); HC is about 4%. Kernel counts and numerical work are effectively
unchanged. Timing changes in unchanged kernels, especially on the 4090D, show why
clock/submission/profiling effects prevent treating this difference as reduced
arithmetic. DMA activity itself is only about 5.8/5.7 ms in the trace, whereas
host submission/readback can wait substantially longer. Native graphs address
repeated small-kernel host submission; broader attention/state graph work remains
outside this change.

## Follow-up: fresh-process throughput reproducibility and variance

Investigation on the existing branch, with the installed release
binary unchanged. The primary cohort uses three independent installed-wheel server
processes at requested split [76,46,0], verified as 26/14 blocks, MTP3,
65536-token capacity, and both Engram tables in `fp16_disk` mode. Each process
runs two full 512-token stock/native warmup pairs followed by five measured
pairs, alternating order and reversing initial order in the middle process.
The 177-token workload is loaded verbatim from the saved prompt artifact.
Complete sequences, acceptance, server timings, client timestamps and boundary
memory/placement snapshots are retained. No polling telemetry, profiler, audit,
or experimental implementation is active in these primary runs.

Reproduction: `D:/tabbyAPI/venv/Scripts/python.exe -I
util/deepseek_v41_followup.py`. Evidence: `native-results/deepseek-v41/followup/fresh`.
The new client intentionally does not import Torch or tokenize the prompt;
this is an explicit harness difference to investigate separately, not assumed
to explain previous results. Native-plan selection alone changes within each
pair; graph policy, MTP, placement, request format and installed binary remain
fixed. Generator recreation excludes prefix-cache reuse, checked from actual
`cached_tokens=0`. Warmup is excluded prospectively, not by deleting slow runs.
The existing validation endpoint changes `native_plan.enabled`, the runtime
selector initialized by `EXL3_NATIVE_PLAN`; this permits same-process pairing
without importing different binaries. It does not rely on changing an
environment variable after its import-time value has already been read.

The three-process primary cohort is complete (15 steady pairs, 12 additional
warmup requests). No profile/telemetry hooks were used in these server processes.

| Fresh process | Stock median (range), tok/s | Native median (range), tok/s | Stock / native sample SD | Stock / native sample variance |
|---|---|---|---|---|
| 0 | 23.461 (23.021–23.499) | 27.429 (26.909–27.610) | 0.238 / 0.290 | 0.0566 / 0.0839 |
| 1 | 23.427 (21.344–23.792) | 27.581 (26.568–27.701) | 1.130 / 0.465 | 1.2769 / 0.2160 |
| 2 | 21.818 (21.433–23.854) | 27.189 (26.064–27.581) | 1.248 / 0.677 | 1.5573 / 0.4582 |
| Pooled, n=15 each | 23.427 (21.344–23.854) | 27.327 (26.064–27.701) | 0.974 / 0.512 | 0.9480 / 0.2624 |

The pooled median ratio is +16.65%; pooled coefficients of variation are 4.16%
stock and 1.87% native. All 42 requests have the original exact complete token
sequence, 288 accepted / 77 rejected drafts, 177 actual prompt tokens and no
cached tokens. Native high performance therefore reproduces across all three
fresh **installed-wheel** processes, without changing clocks, power policy,
affinity or the release. It is not an isolated candidate-build maximum.
Raw process summaries and the pooled summary are under `followup/fresh` and
`followup/fresh-summary.json`.

Server readiness takes 33.143 / 33.090 / 33.109 s, before the separate
specialization warmups. Warmup reaches 326 HC graph captures / 163 cached plans;
these counts remain stable during timing. Requests are non-streaming, with
server `time_generate` as the throughput denominator; HTTP wall time and request
timestamps are also saved. HTTP parsing/network time is not counted as decode.
The earlier benchmark uses the same server generation/prefill timers. Prompt
occupancy is verified from the actual job, not the misleading historical `c178`
filename. No output truncation, changed acceptance trajectory, prefix-cache hit,
or changing graph-capture count explains the primary throughput differences.

## Follow-up: GPU and CPU telemetry correlations

Separate instrumented cohorts follow the completed primary measurements.
Telemetry correlations are distinguished below from controlled interventions.

The separate diagnostic server records 4096-token maximum prefill chunks
(shared chunk 2048), 56 Torch intra-op/inter-op threads, and successful direct
copies in both GPU directions (`bounced=0`, one successful pair probe). This
rules out a host-bounce fallback in this process. The CPU is an AMD EPYC 7663,
56 cores / 112 logical processors. One-second PDH counters are available for
per-core utilization/performance, run queue, context switching, host paging,
physical disk reads and GPU dedicated/shared memory. NVML PCIe RX/TX counters
are available on both active GPUs. These are recorded only in the separate
diagnostic cohorts. Nominal WMI/psutil frequency is not treated as effective
CPU frequency; PDH performance and activity are recorded separately.

Initial telemetry found a Windows venv-launcher pitfall: the PID returned by
`Popen` is a waiting launcher, with the actual inference in its Python child.
The first cohort's per-process memory/IO/thread fields are excluded; its child
CPU totals are recoverable from the recorded all-process CPU snapshots.
`host/telemetry/process-resolution.json` marks the limitation. Subsequent monitor
starts select the Python descendant with the largest resident set. System/GPU
counters in the first cohort remain usable.

The first monitored cohort reproduced native 27.039–27.242 tok/s. Its inference
child accumulated approximately 54.8–55.3 CPU-seconds per elapsed second, despite
batch size one. Busy-core PDH performance was about 166% of nominal (nominal
2 GHz), with no corresponding evidence of sustained host hard paging. This
does not show that all those CPU cycles lie on the critical path; worker spin
and parallel mask operations are hypotheses for a separate thread-count test.

Corrected per-thread sampling finds 56 continuously active threads, each using
nearly one CPU-second per second (both user and kernel time). They report primary
group 0 with its 64-logical-processor mask; the process spans groups 0 and 1.
These API results do not establish actual core residency or exclude the modern
Windows default of scheduling across groups. `libiomp5md.dll` is loaded; this alone does not prove
that every hot thread belongs to OpenMP. The process-local 56-versus-1 Torch-thread
test and 2×2 thread-count/main-thread-group comparison below separate
parallel-worker pressure from placement of the submitting thread. All such
settings disappeared with the isolated server; no system affinity policy changed.

API interpretation follows Microsoft's documentation: on modern Windows,
[GetThreadGroupAffinity reports the primary group](https://learn.microsoft.com/en-us/windows/win32/api/processtopologyapi/nf-processtopologyapi-getthreadgroupaffinity),
whereas [SetThreadGroupAffinity explicitly confines the thread to one group](https://learn.microsoft.com/en-us/windows/win32/api/processtopologyapi/nf-processtopologyapi-setthreadgroupaffinity).
The default unrestricted condition is therefore measured before any explicit
group binding; ending the isolated server removes all experimental bindings.

Engram's actual Windows disk handle uses `FILE_FLAG_OVERLAPPED |
FILE_FLAG_NO_BUFFERING` (`loader/safetensors.py` and `ngram_gather_win.cpp`).
Thus these gathers bypass the Windows file-data cache; generic Linux page-cache
warming is not an appropriate explanation for this storage path. Physical D:
reads persist during repeated identical prompts (roughly 13.3 MB/s in the fast
native decode windows). Controller caching and other host paging remain distinct
possibilities; no storage mode or system cache policy was changed.

The completed scheduling and OpenMP interventions provide causal evidence for
host runtime policy, beyond those correlations. Default 56-thread execution
consumes ~55.6 core-equivalents. Keeping 56 threads but changing only the Intel
runtime's block time from 200 ms to zero reduces that to 1.10–1.23, while native
decode rises from ~27 to ~33 tok/s. Busy-core performance remains ~165–169% of
the 2 GHz nominal frequency. This is a PDH performance estimate, not a sampled
instruction-level effective-clock trace. Both GPUs retain median memory clocks
5001 / 13365 MHz. Typical sampled core clocks/power increase with faster host
submission (4090D ~360–375 to 465–510 MHz, ~49 to 52 W; RTX PRO 6000 ~2060 to
2100–2130 MHz, ~114 to 127 W). The controlled cause is the CPU runtime policy;
some resulting GPU activity/clock change can mediate its end-to-end benefit.

The low historical state instead has memory clocks 810 / 7001 MHz. Those clocks
are useful indicators, but were never locked or independently manipulated.
GPU P-states, clocks, utilization, power, physical VRAM and PCIe rates are saved
for both active adapters. Host process/thread CPU, memory/IO, per-core PDH
counters, other process CPU, disk reads and paging are synchronized to request
and decode windows. Host snapshots are actually about two seconds apart because
counter collection adds to the one-second sleep; GPU polling is one second.
Short bursts can be missed. No hard-paging spike or SSD saturation explains
the controlled CPU-policy gain. Affinity and worker-policy results should not
be added together: they address overlapping effects.

## Follow-up: root cause of the 27.34 versus 20.41 tok/s discrepancy

The original high native rate is reproducible with the installed wheel in all
three fresh processes. A matching requested split alone does not imply identical
runtime state; the historical low-rate trigger cannot be identified uniquely.

Historical artifact comparison establishes identical full token IDs (SHA256
`b64dd5a15ca2a89f2ce19e37a506bc028b84725b909b9160d90a0bcfe9e8e98c`),
288/77 acceptance/rejection, 177 prompt tokens, serialized placement, generator
configuration and server CLI. Both had 326 graph captures without subsequent
growth during timing. Candidate and installed runs both used a Torch/tokenizer
client and 500 ms NVIDIA polling. Their warmup histories differed: the installed
process had additionally executed long audited workloads before timing.

The client-import crossover does not reproduce the large regression: native
with the Torch/tokenizer client was 26.895 median (26.788–27.102, n=3), versus
26.572 before monitoring and 27.170 in the preceding monitored block. Later
lightweight-client runs slowed further, rather than recovering on client exit.
This rules out importing Torch in the client as a sufficient explanation here.
The telemetry-on/off blocks likewise do not explain a 25% regression, although
they do not establish zero monitoring overhead.

Offline decode-window alignment of the historical NVIDIA CSVs is preserved in
`followup/historical-telemetry.json` (response creation timestamps have one-second
resolution; one second at each boundary is excluded). The two slow native
installed runs, 20.412 and 19.934 tok/s, had median 4090D memory clock 810 MHz
and RTX PRO 6000 memory clock 7001 MHz. The three candidate native runs had
5001 and 13365 MHz respectively. Physical VRAM usage was effectively identical
(43717 / 86227 MiB). The faster installed native sample, 24.522 tok/s, also
had the higher median memory clocks. Core clocks, power and utilization differed
as well. This is evidence of different runtime activity/power states, **not proof
that clock policy caused the change**: host pacing can change those states.

The longer diagnostic process also transitions from native ~27.1 tok/s early
to 24–25 tok/s after the client crossover and 20.324 tok/s after profiling,
without an installation, placement or token change. Corrected child telemetry
during the 20.324 run again shows median memory clocks 810 / 7001 MHz, CPU
performance ~166% of nominal, and physical VRAM only ~58 / 84 MiB higher after
profiling. Profiling and elapsed session history are confounded here; this is
not proof of a persistent profiler defect. It does establish that the low
throughput is reproducible **within one process**, rather than being an inherent
property of a fresh installed wheel. The CPU/thread interventions below establish
a substantial host-runtime bottleneck, but do not reconstruct the missing
historical CPU scheduling state.

Established conclusions: the wheel is capable of the original high throughput;
the low state can also arise later inside a process; host worker policy and
submitting-thread placement can change throughput substantially with the same
binary and generated workload. Lower GPU clocks correlate with the historical
slow state, but are not established as its initiating cause. The exact trigger
of the original 20.41 tok/s session remains unresolved because it lacks CPU
thread/scheduling and GPU residency traces. Neither WDDM alone, SSD streaming,
nor wheel installation is an adequate root-cause claim. The repeatable high-rate
conditions are a fresh process, the recorded warmup, unchanged 26/14 placement
and graph policy, and this saved workload; they are not a guarantee that an
arbitrarily long session will retain the same rate.

## Follow-up: prefill-performance investigation

Controlled same-binary placement/order/warmup tests are complete. Decode and
prefill are analyzed separately.

The first fresh 26/14 prefill cohort reproduces 8K at 3.717 s initially,
then 3.629 / 3.728 / 3.627 s; 64.5K starts at 27.280 s. Actual occupancies
are 8192 and 64511 with zero cached tokens. Chunk size is 4096. These are
instrumented prefill cohorts with 64 output tokens, not decode comparisons.
The outer generator prefill scope for the first 64.5K request is 27.387 s,
consistent with server prefill 27.280 s plus the first decode iteration; the
reported difference is not a client-timing artifact. Engram gather scopes total
1.777 s across worker/main threads and may overlap model execution, so they
must not be added to the outer wall time. The 30/10-placement and allocator-cache
interventions follow below, including a fresh return to 26/14.

The fresh 30/10 process is slower again, though it does **not** reproduce the
historical slowdown's full magnitude: 8K steady 8.900–8.995 s versus the old
~15.5 s; its first 64.5K prefill is 63.865 s versus the old ~101 s. Its RTX PRO
6000 reserved bytes match the historical 101483282432-byte value and CUDA reports
zero free bytes. Read-only DXGI queries show process local usage ~102.20 GB
against a 101.37 GB WDDM local budget. By contrast, 26/14 has about 8 GiB physical
headroom during long prefill. The slow placement runs at **higher** core clocks
(~2212 versus ~1935 MHz) and the same 13365 MHz median memory clock, so a lower
GPU clock is not the explanation for this prefill difference.

The slow 64.5K run has ~98% median GPU utilization but lower power, increased
RTX PRO 6000 PCIe TX (~5.8 million KiB/s sampled mean versus ~2.3 million), and
almost no host hard paging. Disk read volume is similar (~4.8 GB estimated from
sampled rates × duration), with lower read rate in the slower run. This strongly
supports GPU memory pressure/residency traffic rather than SSD throughput or CPU
pagefile stalls. PCIe counters do not label transfers as migration, so that last
mechanism is an inference rather than a direct paging-packet trace.

The DXGI interpretation is consistent with Microsoft's
[QueryVideoMemoryInfo documentation](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo),
which identifies process usage above its assigned budget as a source of paging
and stalls. The measurements here, rather than the platform label alone, are
the basis for the memory-pressure hypothesis.

Releasing unused allocator blocks before 8K temporarily restores ~1.61 GB CUDA
headroom, but execution grows back over budget: prefills remain 9.022 / 8.989 /
9.060 s. A further controlled process kept the 30/10 placement and loaded
weights but reduced runtime prefill chunks to 2048, testing the transient
workspace/headroom explanation before returning to 26/14.

Completed prefill comparison (first request at each context excluded prospectively
as specialization warmup; seconds, sample SD/variance):

| Placement / runtime chunk | Warmed n | 8K median (range), SD / variance | 64.5K median (range), SD / variance |
|---|---:|---|---|
| 26/14, 4096 | 3 | 3.629 (3.627–3.728), 0.058 / 0.00334 | 27.430 (27.366–27.477), 0.055 / 0.00307 |
| 30/10, 4096 | 3 | 8.922 (8.900–8.995), 0.050 / 0.00245 | 63.858 (63.717–65.355), 0.908 / 0.82445 |
| 30/10, 2048 + cache release | 2 | 4.058 (4.034–4.082), 0.034 / 0.00114 | 30.619 (30.583–30.655), 0.051 / 0.00260 |
| Fresh return to 26/14, 4096 | 1 + cold | 3.626 (cold 3.770) | 27.370 (cold 27.419) |

At fixed 30/10 placement, reducing transient workspace cuts prefill by ~54.5%
at 8K and ~52.1% at 64.5K, whereas cache release alone does not. The 8K local
usage moves just below budget (~128 MB headroom); 64.5K remains slightly over
(~143 MB instead of ~840 MB). This intervention, budget measurements and excess
PCIe traffic strongly support memory-pressure stalls as the large placement
penalty. Different chunk kernels are an additional changed variable, so no exact
migration-time decomposition is claimed. The old 15.5/101 s severity is not
reproduced exactly; the original and current 30/10 allocation footprints match,
but no historical DXGI/paging trace exists to resolve the remaining severity.

Each fixed placement/chunk cohort is token-repeatable, and the return to 26/14
matches its first cohort exactly. Changing chunk size changes the generated
trajectory/acceptance, so it is not a parity-preserving production optimization.
Keep the validated 26/14 placement with its substantially larger headroom.
Evidence: `followup/prefill-summary.json` and the four `prefill*` directories.

CPU-policy cross-checks are separate from that placement comparison. With
56 threads retained, default/zero block-time prefills after context warmup are
3.630/3.713 s at 8K and 27.436/27.499 s at 64.5K, with exact tokens and MTP
counts. These single pairs show +2.3%/+0.23% prefill time, not enough samples to
claim either a regression or exact equivalence. A separate 1/56-thread check
gives 3.818/3.633 and 27.378/27.418 s; its first 8K request is not equivalently
warmed. Neither intervention recreates a fourfold prefill penalty. Initial and
warmed selected-placement runs are similar, while the overloaded placement
remains slow after warmup, ruling out compilation/warmup as a sufficient
explanation for the large persistent placement penalty.

## Follow-up: remaining host overhead and ranked opportunities

Targeted exclusive-time/call-site measurements and reversible experiments follow.
Call counts and cumulative timings are diagnostic leads, not direct estimates of
achievable speedup. Existing correctness guards remain.

Source inspection identified a concrete upstream harness interaction:
the benchmark's `min_tokens=max_tokens=512` makes the generator's existing
`job.new_tokens >= job.min_new_tokens` batched-readback condition false throughout
decode. It therefore reads samples back serially and rebuilds/uploads the
stop-token mask after accepted positions. Separate diagnostic requests with
`min_tokens=0` exercised the existing batching path, and the existing
`_NO_BATCH_READBACK` test switch forced its serial fallback while holding
that request fixed. These are **not replacements for the primary benchmark**;
exact full output and acceptance parity was verified before comparison.

The first targeted guard microbenchmark is complete. On loaded target/draft
blocks with valid cached plans and representative query lengths 2/4/5, seven
batches of 2000 calls give original medians 24.16–26.77 microseconds/call.
The earlier cProfile cumulative 0.493 s / 3698 calls was 133.3 microseconds/call,
so it cannot be used as an uninstrumented saving estimate. A temporary standalone
copy with only the three imports hoisted (all checks retained, identical cached
plan returned) takes about 21.9–23.6 microseconds/call. This is a microbenchmark,
not an end-to-end optimization result. Even eliminating the entire Python guard
would save only roughly 0.23–0.26 s at ~9632 calls/512-token decode, before overlap
and second-order effects; import hoisting alone is on the order of 0.02–0.05 s.
No guard or installed source has been changed. Evidence: `host/guard-microbench.json`.

The existing readback mechanism has a controlled end-to-end result: at
`min_tokens=0`, toggling only the upstream `_NO_BATCH_READBACK` fallback yields
serial/batched pairs 20.964/23.612, 20.415/21.615 and 21.807/23.579 tok/s.
All complete tokens and 288/77 MTP counts match the original benchmark.
Paired gains are about 12.6%, 5.9% and 8.1%. This demonstrates a real readback
opportunity on this workload, rather than interpreting cumulative `.cpu()` time
as a saving. It does not establish that arbitrary constrained/minimum-length
requests may use that path: the existing eligibility checks remain untouched.
These later diagnostic runs include the corrected external telemetry monitor
and occur in the slower runtime state, so their absolute rates are separate
from the primary uninstrumented cohort.

The first uninstrumented CPU-thread intervention is large and repeatable. With
the request, binary, native graphs, placement and GPU policies fixed, 56/1-thread
pairs are 27.273/34.386, 27.213/34.355 and 27.402/34.813 tok/s. Every sequence
and 288/77 acceptance result is exact. The median rises ~26.1%, saving roughly
3.9 seconds per 512-token decode. This establishes avoidable host-runtime
overhead independently of Python call counts. Prefill costs, explicit processor
group effects, and OpenMP spin-time controls are reported below. No persistent
thread setting has been changed.

The 2×2 scheduling intervention now separates the effects. With 56 Torch
threads, explicitly placing only the submitting thread in group 1 raises native
throughput from 26.966–27.202 to 32.537–32.776 tok/s (three runs each), while
process CPU consumption remains ~55.5 core-equivalents. With one Torch thread,
group 0 and group 1 both give ~33.7–34.0 tok/s and process CPU falls to ~1.0–1.1
core-equivalents. Both groups have the same 5001/13365 MHz median GPU memory
clocks in these comparisons. Thus host worker/scheduling interaction is a
demonstrated throughput limiter; GPU clocks alone cannot explain these gains.
The precise historical scheduling state was not captured, so this does not
retroactively assign every second of the original 27.34-to-20.41 discrepancy.

The completed OpenMP experiment independently reproduces the CPU result while
retaining 56 Torch threads. It calls `kmp_set_blocktime(0)` on the submitting
thread, with `kmp_get_blocktime()` confirming 200 ms before and zero afterward.
No affinity changes occur in this process. The
[Intel runtime API](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-2/intel-compiler-extension-routines-to-openmp.html)
controls the worker wait interval for the calling thread and its teams.
Returning to 200 ms after the prefill checks gives 25.798 tok/s, with exact
output, showing that the improvement was not an irreversible warmup transition.
The earlier reversed-order pairs are the stronger controls for elapsed time.

| Native CPU condition | n | Median tok/s (range) | Sample SD / variance |
|---|---:|---|---|
| Default 56 threads, uninstrumented CPU cohort | 3 | 27.273 (27.213–27.402) | 0.096 / 0.00930 |
| One thread, uninstrumented CPU cohort | 3 | 34.386 (34.355–34.813) | 0.256 / 0.06554 |
| 56 threads, 200 ms block time, uninstrumented | 3 | 26.503 (26.441–27.264) | 0.458 / 0.20991 |
| 56 threads, zero block time, uninstrumented | 3 | 33.130 (33.090–33.688) | 0.334 / 0.11164 |
| 56 threads, 200 ms block time, monitored | 3 | 26.972 (26.931–27.330) | 0.220 / 0.04823 |
| 56 threads, zero block time, monitored | 3 | 33.022 (32.211–33.322) | 0.575 / 0.33047 |
| 56 threads, main thread group 0, monitored | 3 | 27.055 (26.966–27.202) | 0.119 / 0.01427 |
| 56 threads, main thread group 1, monitored | 3 | 32.721 (32.537–32.776) | 0.126 / 0.01576 |
| One thread, main thread group 0, monitored | 3 | 33.829 (33.704–34.000) | 0.149 / 0.02207 |
| One thread, main thread group 1, monitored | 3 | 33.917 (33.851–33.950) | 0.050 / 0.00255 |

Zero block time gives +25.00% by the ratio of uninstrumented medians and
paired median +24.85% (23.56–25.30%), saving paired median 3.846 s per 512
tokens. Monitored paired gains are 19.60–23.54%. One thread gives +26.08% by
median ratio and paired median +26.25%, saving 3.911 s. These are separate
cohorts, not additive improvements or a statistically decisive comparison
between one thread and zero block time. The latter retains CPU parallelism.
Full statistics, including decode-time variance, are in `experiment-summary.json`.

Host attribution uses three complementary measurements: end-to-end interventions,
exclusive call-site wall timers, and a short CUDA trace. The instrumented window
starts after 64 generated tokens and covers 195 target/draft iterations. The
following exclusive scopes do not include their instrumented children, but they
still include blocking waits inside the called function. Columns come from
separate processes/runtime states and must not be interpreted as a matched
speedup decomposition.

| Scope, count in window | Default host profile, s | One-thread profile, s | Zero-block-time profile, s |
|---|---:|---:|---:|
| Outer generator wall time (inclusive reference) | 21.597 | 14.439 | 14.962 |
| Target model residual, 195 | 11.002 | 8.344 | 8.399 |
| Draft model residual, 195 | 1.049 | 0.792 | 0.885 |
| GPU sample `.cpu()`, 402 | 3.527 | 1.462 | 1.568 |
| CPU-to-GPU `device_copy.to`, 975 | 1.256 | 0.404 | 0.431 |
| Cross-GPU `device_copy.to`, 6825 | 1.096 | 0.829 | 0.857 |
| Python native binding guard, 8385 | 0.644 | 0.478 | 0.473 |
| Engram gather, 780 | 0.526 | 0.433 | 0.433 |
| Explicit `cuda.synchronize`, 45 | 0.523 | 0.140 | 0.150 |

The model residual is **not pure Python interpreter time**: it includes unwrapped
Python, extension dispatch, CUDA calls and waits. The new cProfile workload has
3.979 million calls / 19.498 s, versus 3.972 million / 13.384 s with one thread,
over a longer window than the old 1.728-million-call report. Pure interpreter
cycles cannot be isolated accurately here. Some decorated/resumed functions
have cProfile cumulative time smaller than self time, so summing parent and
child timings would be especially misleading. Per-call `thread_time_ns` also
has 15.625 ms Windows accounting granularity; small-call CPU values can be zero
or exceed wall time and are not used as precise per-call CPU measurements.

The `.cpu()` distinction is concrete. `job.py:803` copies an eight-byte sampled
token from GPU 1 and blocks for readiness: 402 copies transfer just 3216 bytes,
yet occupy 3.527 s in the default profile. The 195 `dspark.py:249` `.cpu()` calls
already operate on CPU tensors and return the same object, totaling 0.000540 s;
another 45 CPU-resident sample calls total 0.000144 s. They are not meaningful
copy bottlenecks. At `device_copy.py:86`, 195 eight-byte CPU-to-GPU copies alone
occupy 1.199 s with `non_blocking=False`. That duration reflects runtime/stream
readiness, not bandwidth for 1560 bytes. The remaining host-to-GPU calls and
the cross-GPU calls are real transfers; cross-GPU volume totals ~793 MB in this
window. No host-bounce fallback was observed. Same-device `.to()` can also be
a real dtype conversion; identity-returning CPU `.to()` sites cost only
milliseconds. Sample readiness waits cannot simply be deleted while preserving
the serial MTP dependency chain.

Stock and native host profiles have identical counts and bytes for these major
copy sites (including 402 sample readbacks, 975 CPU uploads and 6825 cross-GPU
copies). They are inherited from upstream ExLlamaV3. Native IR adds its binding
guard and graph launches but does not introduce those transfers. Logit-mask
preparation costs only 0.061 s cumulative / 0.021 s self in the default cProfile
window; `zero_` totals 0.010 s. Those operations can wake workers, but their own
execution time does not justify attributing 55 busy CPU cores to necessary mask
computation. The spin-time intervention isolates the avoidable runtime effect.

Binding inspection finds repeated capability, shape/dtype/contiguity/device,
HC/RMS parameters and tensor identity/data-pointer checks before the cached-plan
lookup. Three repeated imports and construction of small tuples/signatures are
avoidable dispatch work. The standalone import-hoisting microbenchmark retains
all checks; its small saving is reported above. Caching more immutable binding
metadata requires explicit invalidation on weight/device/parameter changes.
Dynamic input, `hc_pre`, override and alias checks remain necessary. C++ execute
and graph replay also validate their bindings; their validation-only time is
not separately isolated from extension/runtime execution. No guard was weakened.

The short native CUDA trace contains 15 target/draft steps over ~1.865 s:
12557 ordinary kernel launch calls (275.3 ms host API time), 2429 cooperative
launches (99.2 ms), 1320 graph launches (55.8 ms), 1591 asynchronous copies
(108.3 ms), 86 stream synchronizations (34.3 ms), 60 event synchronizations
(0.23 ms) and four device synchronizations (6.74 ms). Per-device unions of GPU
activity are 277.5 and 330.5 ms, with ~5.8 ms summed DMA activity. Device unions
can overlap each other and the CPU; none of these durations should be added
to manufacture total wall time or a speedup. Launch-end-to-GPU-start p90 is
~171 microseconds on GPU 0 and 1258 on GPU 1, with maxima 13.5/10.0 ms. This
includes earlier queued work and dependencies, **not isolated WDDM scheduling
delay**; a kernel can start before its host launch call returns.

Existing HC graphs capture the begin mix/normalization and middle
apply/mix/normalization stages. Final carried apply, attention/KV/index state,
Engram, sampling and the transfers above remain outside them. The measured
26.4% reduction in individual kernel-launch calls is real (17072 to 12557),
but cooperative launches and copies remain. V4.1 shared-pool readers, latent
index owners and candidate selection prevent assuming the older whole-attention
graph path is applicable. Scratch-cache key construction also has measurable
Python work: `make_key` self 0.243 s and `get_bucketed` self 0.223 s; the latter's
0.466 s cumulative includes children and must not be added to their self times.
This suggests narrowly caching state metadata before attempting wider capture,
but no controlled end-to-end gain for that change is demonstrated here.

Ranked opportunities below use measured costs as **conditional upper bounds**,
not additive savings. Profile-derived ceilings apply only to their instrumented
window; unprofiled benefit may be much smaller. Estimates beyond rank 1 and the
existing eligible readback path are not demonstrated performance improvements.

| Rank / candidate | Current cost and maximum removable envelope | Realistic decode benefit | Complexity, correctness and compatibility |
|---|---|---|---|
| **1. Process-local OpenMP worker wait policy** | Controlled excess 3.58–3.91 s/512 tokens; removing the measured excess gives 23.6–25.3% more tok/s. No evidence supports a larger universal ceiling. | **About 20–25% on this host/workload**, measured with zero block time; ~33 tok/s instead of ~26.5–27. | Low implementation complexity as an opt-in runtime setting. Exact short/long outputs and MTP counts here; no IR ABI or math change. Runtime/vendor/platform detection and CPU-heavy/batched workloads need checks before making it a default for other models. |
| 2. Batch readbacks for requests with provably invariant constraints | Default profile spends 3.53 s in GPU `.cpu()`; tuned profile 1.57 s. Eliminating all tuned readback wall time would give at most ~12% in that instrumented window, but includes necessary GPU readiness. | Existing eligible path measured +5.9–12.6% in slow state. A new safe constrained-request extension might yield 3–8%; unproven and likely less after CPU tuning. | Medium. Preserve EOS/minimum-length boundaries, mask/filter changes, RNG consumption, acceptance and rewind. Upstream generator affects other IR models; no eligibility guard may simply be removed. |
| 3. Narrow transfer/submission consolidation | Tuned cross-GPU call envelope 0.857 s/~15 s window: hypothetical removal <6.1% throughput; 793 MB and dependencies still required. Small CPU token upload adds a separate wait envelope, overlapping readback/runtime effects. | 0–3% planning estimate until a fixed-output end-to-end test; no demonstrated gain. | Medium/high. Maintain stream ordering, pinned-buffer lifetime, shared-index ownership and direct-copy semantics. Upstream/shared attention compatibility risk. |
| 4. Cache immutable scratch/attention metadata | Two scratch-cache functions total 0.465 s **self**, ~2.4% of the 19.5 s cProfile window; removing all would give ~2.5% there. Profiling inflation means this is not a production bound. | 0–1% provisional; no measured gain. Wider graph capture has no defensible standalone estimate from this trace. | Medium for bounded cache, high for graphs. Invalidate on shape/device/cache state, MTP rewind and shared KV/index mutation; architecture-specific gating required. |
| 5. Python binding import hoisting / metadata caching | Unprofiled full guard ~0.23–0.26 s/512 tokens, roughly 1–2% absolute elimination ceiling; import-only saving ~0.02–0.05 s, <0.3%. | Import hoisting <0.3%; more caching plausibly <1%, not end-to-end established. | Low for imports, medium for binding-version cache. Keep all dynamic guards and C++ checks; stale tensor identity/device metadata risks silent numerical errors across native models. |
| 6. Engram host gather bookkeeping | Tuned gather scope 0.433 s/~15 s window; hypothetical entire elimination ~3%, including real IO. Prefill worker work can overlap and cannot be summed. | Likely <1% for bookkeeping alone; no validated change. | Medium. Preserve disk mode, row order, buffer lifetime and prefetch synchronization. Storage/compression redesign is outside scope. |

**Recommendation:** implement and validate an opt-in process-local OpenMP wait
policy next, starting with zero block time on this Windows Intel-OpenMP runtime.
It is worthwhile: the controlled benefit is about 20–25%, much larger than the
binding-guard opportunity, with exact output parity and little prefill change
in the checks above. Retain the 26/14 placement and existing correctness guards.
The tested API is the runtime setter; a startup environment-variable equivalent
has not been separately benchmarked. Do not make this an unconditional global
default or add it to persistent Tabby configuration on this evidence alone.
No permanent performance change is shipped by this investigation.

## Follow-up: limitations and unresolved questions

CPU stack sampling through WPR could not start: Windows returned
`0xc5585011`, "Failed to enable the policy to profile system performance."
No system profiling policy was changed to bypass that restriction. The saved
`cpu/wpr-start.json` records the failure; per-thread CPU counters, cProfile,
call-site timers and CUDA traces remain available. Exact native-stack CPU-cycle
attribution and OS scheduling/paging packet attribution are therefore limited.

The exact initiator of the historical slow state and the old 15.5/101 s prefill
severity remain unresolved. CPU sampling permission and lack of an OS GPU paging
trace limit attribution; CUDA queue delays do not substitute for such a trace.
Monitoring/profile overhead, coarse sampling, background processes and evolving
session state preclude a complete additive CPU/GPU wall-time budget. Primary
throughput uses uninstrumented processes; instrumented results are labeled and
not pooled with it. One prompt and batch size one cannot establish behavior on
all workloads. CPU-policy cohorts have three pairs each, and their long-prefill
cross-check has one warmed pair per context. Benefits are not additive.

Final offline checks find **127 complete 177/512 requests**, including warmups,
profiles and controlled variants, with original exact tokens, 288/77 MTP counts
and zero cached prompt tokens. Eleven 8K and eleven 64.5K requests at unchanged
26/14 placement/chunk also match within context, including the CPU interventions.
The deliberate 30/10 chunk-size diagnostic changes output trajectory and is
excluded from that parity claim. No completed model correctness suite was rerun.

Preservation checks pass for all **644 installed wheel payload files**, the
release wheel and extension SHA256, persistent Tabby configuration/authentication
hashes, original `convert.py` edit, branch and original HEAD
`e5f428727b54f41dc0250c50615c81dfa9e0e72f`. No installation, release source,
clock policy, power setting or system affinity setting changed. All temporary
servers/monitors have exited, port 5008 is free, and WPR is not recording.
Process-local thread/affinity/OpenMP changes ended with those servers. The
report and new investigation utilities remain as working-tree changes on the
existing branch; unrelated changes are preserved. Verification evidence is
`followup/final-preservation.json`.

Reproducible entry points (PowerShell, repository root; output directories must
be new to avoid replacing evidence):

```powershell
$py = 'D:/tabbyAPI/venv/Scripts/python.exe'
& $py -I util/deepseek_v41_followup.py --output native-results/deepseek-v41/followup/fresh-repeat
& $py -I util/deepseek_v41_experiments.py --phase host --output native-results/deepseek-v41/followup/host-repeat
& $py -I util/deepseek_v41_experiments.py --phase blocktime --output native-results/deepseek-v41/followup/blocktime-repeat
& $py -I util/deepseek_v41_experiments.py --phase cpu --output native-results/deepseek-v41/followup/cpu-repeat
& $py -I util/deepseek_v41_experiments.py --phase prefill76 --output native-results/deepseek-v41/followup/prefill76-repeat
& $py -I util/deepseek_v41_experiments.py --phase prefill94 --output native-results/deepseek-v41/followup/prefill94-repeat
& $py -I util/deepseek_v41_experiments.py --phase prefill94-small --output native-results/deepseek-v41/followup/prefill94-small-repeat
& $py -I util/deepseek_v41_experiments.py --phase prefill76-return --output native-results/deepseek-v41/followup/prefill76-return-repeat
& $py -I util/analyze_deepseek_v41_followup.py
& $py -I util/analyze_deepseek_v41_experiments.py
& $py -I util/analyze_deepseek_v41_profiles.py native-results/deepseek-v41/followup/host
& $py -I util/analyze_deepseek_v41_telemetry.py native-results/deepseek-v41/followup/cpu
& $py -I util/verify_deepseek_v41_followup.py
```

Raw evidence lives under `native-results/deepseek-v41/followup/`: `fresh`, `host`
(including `torch-client`, original and corrected-child telemetry), `cpu`,
`blocktime`, `prefill76`, `prefill94`, `prefill94-small`, and `prefill76-return`.
Saved request JSON includes full tokens, acceptance, actual occupancy, server and
client times, placement and boundary memory state. Diagnostic folders also keep
script snapshots, server commands/logs, raw counters and profiles. Historical
inputs remain in `splits/budget76/timing` and `installed/timing`. Analyzers are
offline and do not import Torch. Profile/telemetry analyzers accept a cohort
directory; `--telemetry telemetry-child` selects the corrected late host monitor.

## OpenMP policy release: 0 / 1 / 50 ms sweep

The new campaign uses the preserved installed `ds41.1` wheel and the same
177-token prompt, 512 outputs, 26/14 blocks, MTP3, graphs and disk Engram.
Five warmed rotating/reversed-order runs per setting are complete:

| Block time | Median tok/s | Range | Sample SD |
|---|---:|---|---:|
| 0 ms | 33.543 | 32.611–34.086 | 0.534 |
| 1 ms | 33.663 | 33.158–33.960 | 0.311 |
| 50 ms | 28.081 | 19.189–28.387 | 4.025 |

All full tokens and MTP counts are exact. The 50 ms outlier is retained. Limited
200 ms baseline controls give 24.805 cold/warmup and 24.136 warmed tok/s in this
process; this campaign does not assume the previous process's absolute baseline.
The ~0.36% median difference between 0 and 1 ms is smaller than run variation.
Separate monitored decode windows use 1.09–1.11 core-equivalents at 0 ms,
3.36–3.42 at 1 ms, and 42.91–43.68 at 50 ms. Throughput is ~33 tok/s at both
0/1 and ~28 at 50. Zero is therefore the provisional choice on CPU consumption;
there is no evidence that 1 ms buys a meaningful decode advantage.
Separate telemetry, populated 8K/64.5K tests, and two further fresh-process
0/1 comparisons follow before selection. Evidence: `omp-policy/deepseek-sweep`.

## OpenMP policy release: cross-model and concurrency coverage

Local checkpoints are available for GLM-5.3 Flash, Qwen3.8 Flash Next, Nemotron
3.5 Super VL, older DeepSeek V4 Flash, Gemma 4 and Devstral 2. The campaign uses
their prior model-specific loaders: GLM 90/44 layer split and separate native
TP; Qwen and Nemotron on the RTX PRO 6000; DeepSeek V4 85/40; Gemma 20/40;
Devstral 55/22. These are independent of the V4.1 budgets. MTP windows follow
the existing harnesses (GLM/Nemotron 3, Qwen 4, older DeepSeek 5). Small concurrent
cohorts include Nemotron B3/MTP so its 12-row native assignment path is eligible.
No new checkpoint, conversion or CPU-expert offload mode is introduced.

## OpenMP policy release: implementation and validation progress

The candidate is an opt-in `EXL3_OMP_BLOCKTIME_MS` setting; unset behavior remains
unchanged. Source inspection confirms model loading and generation can execute
on different threads. The policy therefore initializes each actual inference
thread once, including direct model calls, generator iteration and TP workers.
Repeated token calls take a thread-local fast path and do not call the native
setter/getter again. No Torch thread-count or affinity policy changes.
Candidate source changes are not installed; all initial model measurements
continue to use the preserved wheel and experimental runtime setter.

The 11 focused policy unit tests pass: unset/no-side-effects, explicit 0/1/50,
single setter per thread, new execution threads, readback without mutation,
unsupported-runtime fallback, invalid input, rejected-setting restoration and
fork-marker reset. Runtime initialization is outside any architecture-specific
IR guard. The next integration gates use the real Intel API and model entry
points on different Python threads, followed by the unchanged shared native
regression suite and staged/installed model checks.

Long-context sweep complete (two warmed runs per setting, 512 outputs each):
8K prefill medians are 3.703/3.732/3.777 s for 0/1/50 ms; decode medians are
31.117/31.726/27.116 tok/s. Near-65K prefill remains about 27.4 s for every
setting, while 0/1 decode remains around 27–28 tok/s and 50 ms around 22.9.
All full sequences and acceptance counts match within context. No material
prefill or allocation-stability penalty from zero block time is observed.

The staged candidate passes **234 shared native/policy regression tests**, with
zero failures, errors or skips, using the preserved extension on SM120 and both
SM89 adapters. Real-runtime fresh-process checks pass for unset, 0, 1 and 50 ms.
Unset stays at 200 ms; forward, prefill and generator calls on newly created
threads select the requested value, while the initiating thread remains
unchanged. All retain 56 Torch threads. Repeated forward calls respect an
external later setter rather than repeatedly overriding it. Evidence:
`omp-policy/native-regressions.json` and `runtime-integration/`.

The two initial fresh-process repeats give 0/1 medians 32.774/32.910 and
32.952/33.400 tok/s, with exact full output. Review found the two-value ordering
helper combined rotation and reversal into repeated 0-then-1 order. These
results are retained and labeled; an additional fresh cohort uses explicitly
alternating 0/1 order. The five-round three-value sweep already varied order.

Qwen's completed primary cohorts (three runs/setting) give default/0/1/50 ms
medians 103.876/104.601/104.994/104.667 tok/s. Whole-request CPU consumption
falls from ~9.02 core-equivalents to ~1.17 with zero waiting. At 4K, default
decode is ~110.3 versus ~115.5 tok/s at zero; prefill also improves in these
samples. Concurrent rates vary substantially across rounds and require their
full ranges, not a single winning sample. Complete tokens and MTP counts match.
A harness `KeyError` on absent draft-count fields in an ordinary-generation
result stopped the later control section; it was fixed to record zero for
non-draft jobs. Completed primary data is preserved, and only the missing
ordinary/stock/telemetry controls are rerun in a separate directory.
## OpenMP policy follow-up: additional evidence (in progress)

The completed CPU control (`omp-policy/baseline-next-cpu/results.json`) establishes
why the setting must remain opt-in. With 56 Torch threads, 13 large 2048-square
matrix multiplies took median 0.1227/0.1181/0.1206/0.1201 seconds at 200/0/1/50 ms
for one submitter. Two concurrent submitters took 0.1942/0.1847/0.1923/0.1988 s.
However, 200 short parallel 4M-element additions took 0.0140/0.0434/0.0160/0.0160 s
for one submitter and 0.0278/0.0796/0.0273/0.0297 s for two: immediate sleeping
caused roughly a threefold regression for frequent short CPU regions. All outputs
were exact; the Torch thread count remained 56. These short tests have coarse
process-CPU accounting, so their instantaneous core-equivalent estimates should
not be interpreted precisely. They demonstrate retained parallel CPU execution
and a workload-dependent wakeup cost, not a measured CPU-offload-model result.

Cross-model primary cohorts use a direct generator with no stop conditions.
This differs from the DeepSeek HTTP request, whose minimum length suppresses EOS
through a CPU logit mask. Separate, clearly labeled `masked-r*` controls now test
the minimum-length/EOS-mask case. They must not be pooled with no-stop primary
results or used to attribute every cross-model difference to architecture alone.
The previously completed no-stop cohorts are retained unchanged. Instrumented
requests remain separate from primary timings.

The additional correctly alternating fresh V4.1 process gives 0/1 medians
32.677/33.280 tok/s (ranges 32.091–32.868 / 33.020–33.412; SD 0.405/0.199).
Across fresh repeats, 1 ms sometimes has a small 0.4–1.8% advantage. Zero should
therefore be described as the low-CPU GPU-inference choice, not a claim that it
wins every throughput sample. The much larger difference is between either
0/1 and a long spin period; the CPU-heavy control favors 1 over 0.

Separate V4.1 telemetry records Blackwell mean SM clocks around 2,111–2,140 MHz
across all three policies, versus 662–811 MHz on the 4090D. At 0/1, Blackwell
power is roughly 124–130 W and utilization 24–27%; at 50, about 119–120 W and
23%. These are synchronized correlations, not evidence that clock policy caused
the gain. No clock, power, affinity or system policy was changed. CPU worker
consumption changes by tens of core-equivalents under the controlled setter.

Cross-model primary results completed so far (three 256-token runs per value):

| Model | 200 ms median | 0 ms median | 1 ms median | 50 ms median | 200 → 0 whole-request CPU cores |
|---|---:|---:|---:|---:|---:|
| Qwen 3.8 Flash Next | 103.876 | 104.601 | 104.994 | 104.667 | 9.02 → 1.17 |
| GLM 5.3 Flash, layer split | 58.096 | 59.020 | 58.597 | 58.872 | 5.10 → 1.00 |
| Nemotron 3.5 Super VL | 109.977 | 110.907 | 111.240 | 110.713 | 7.01 → 0.99 |
| DeepSeek V4 Flash | 84.182 | 89.675 | 90.827 | 88.831 | 55.34 → 1.06 |

These are modest single-request changes outside V4.1, with substantial variance
in older DeepSeek's default samples (72.838–93.236 tok/s). The first two concurrent
rounds raise a possible regression: Qwen B2 aggregate request rate is median
71.35 at default versus 64.86 at zero, and older DeepSeek B2 is 68.97 versus 56.44.
Fresh five-round checks are queued before treating these as repeatable effects.
GLM B2 and Nemotron B3 instead improve slightly at zero in these samples.
Full medians, ranges, sample SD/variance, prefill and concurrency rates are
regenerated incrementally in
[`omp-policy/measured-tables.md`](../native-results/deepseek-v41/omp-policy/measured-tables.md)
from the preserved JSON, using `util/analyze_omp_policy.py` and
`util/summarize_omp_release.py`.

Devstral's first concurrent warmup exposed a harness assumption, not a model
failure: its second job reused two prompt tokens shared with the first job in
the same batch. The first job had zero cached tokens and outputs were exact.
The assertion now requires zero previous-request cache reuse for the first job
and identical per-job cache occupancy across matched policies; within-batch
sharing is recorded rather than rejected. Completed Devstral short/4K cohorts
remain in `baseline-next-devstral`; only unfinished concurrency, masked, parity
and telemetry cases resume in `baseline-devstral-tail`. The failed queue and
its source snapshot remain available. This does not change a product guard.

Gemma and Devstral provide useful negative controls. Gemma's short unmasked
medians are 37.075/36.592/36.703/36.251 tok/s at 200/0/1/50 ms: the ranges
overlap, with no established throughput benefit. Devstral stays near 12.6 tok/s
and its 4K prefill near 9.6 s across policies. In Devstral's masked case, default
CPU consumption is about 55.8 cores versus 1.0 at zero, but both decode near
12.52 tok/s. Removing worker spinning therefore saves CPU without necessarily
removing the limiting factor for these dense-model workloads. No CPU-power or
energy saving is inferred from core-equivalent counts alone.

GLM TP's completed short unmasked medians are 80.079/82.833/82.586/82.131 tok/s
at 200/0/1/50 ms; zero improves median decode by 3.44%, with ranges
79.826–80.216 versus 82.591–82.893. The serving-style EOS-mask case is more
sensitive: 58.591 versus 80.201 tok/s at 200/0 (two runs each; default range
54.210–62.973). TP B2 aggregate rates remain essentially unchanged near
20.8 tok/s. All outputs/MTP counts match. Runtime readback confirms each value
in the main process, GPU worker and CPU coordinator. Main-process CPU columns
exclude TP children; synchronized telemetry separately sums the process tree.
One 1-ms long-context decode sample reached only 26.668 tok/s and is retained;
it does not justify attributing a general regression to 1 ms without repetition.

TP graph diagnostics confirm 45 native blocks per GPU rank, 270 captures and
393,390 replays per rank in the baseline process. Separate unmasked 512-token
telemetry gives sustained parent decode CPU near 0.98–0.99 cores at every value
and total process-tree CPU near 2.33–2.93. Thus the larger whole-request CPU
figures in the short table include prefill and its worker-wait tail; they should
not be described as sustained unmasked decode spinning. Monitored unmasked TP
throughput is also closer (82.52 default versus 83.07 at zero) than the shorter
primary cohort. The masked workload remains a distinct, more sensitive case.

The staged implementation now reproduces the manual-setter result without
using that experimental setter. `EXL3_OMP_BLOCKTIME_MS=0` is read at import;
server status confirms actual 0, successful initialization on the serving
thread, previous 200, and 56 Torch/inter-op threads. Five alternating warmed
177/512 pairs give native median **33.184 tok/s** (32.994–33.218; SD 0.090;
variance 0.0081), stock **29.985** (29.724–30.166; SD 0.167; variance 0.0278).
Every run preserves the full sequence and 288 accepted / 77 rejected drafts.
Warmed 8K prefill is 3.761 s and decode 31.897 tok/s. Evidence:
`omp-policy/candidate-deepseek/`. Long-context and staged cross-model gates
continue before committing or changing the installed release.

The staged near-65K result is 27.458 s prefill and 27.998 tok/s decode, with
64511 actual prompt tokens, zero prefix-cache reuse, 512 outputs and exact
288/81 MTP counts. There is no material prefill regression relative to the
manual zero-block-time sweep. The CPU parallel-work controls also pass through
the implemented entry point, retaining 56 threads and exact numerical outputs.
Qwen's staged short, 4K, B2, masked, stock and ordinary-generation cases have
completed with exact within-cohort token/MTP parity; the remaining staged
models continue sequentially.

All seven staged cross-model cases have now passed their functional checks,
including both TP GPU ranks and the CPU coordinator reading actual 0 ms through
the implementation. A performance question remains for GLM TP: its staged
short median was about 76.75 versus 82.83 tok/s in the earlier manual-setter
process. The complete TP shard plan and placement are identical. Separate
512-token telemetry also differs (78.02 versus 83.07), with nearly identical
GPU SM clocks (~2686 MHz on the 4090D and ~2166 MHz on Blackwell); this does not
establish a clock explanation. The initial staged cohort had fewer warmup
requests than the four-setting baseline. A targeted fresh-process staged /
old-wheel-manual / staged comparison now uses four identical warmups and five
short measured runs per process. The earlier slower process is retained, and
no universal TP speedup is claimed while this check is pending.

The fresh five-round concurrency rechecks do **not** reproduce the initially
suspected zero-block-time regression. Qwen B2 aggregate medians are
71.951/74.336/74.075 tok/s at 200/0/1 ms; older DeepSeek V4 B2 gives
65.070/71.150/70.387. The old DeepSeek default still has a wide range, including
one slow sample. Zero uses about 1.05 Qwen and 1.32 DeepSeek main-process cores,
versus 9.20 and 55.14 at default. The original two-round lower results remain
in the tables, but do not establish a general concurrency regression. No
deadlocks or token/MTP differences occurred. These checks used five warmed,
alternating/rotated rounds in independent fresh processes; artifact directories
are `baseline-concurrent-recheck-qwen` and `baseline-concurrent-recheck-deepseek4`.

The limited nearby old-wheel default control is complete: three warmed pairs,
unchanged installed `ds41.1` wheel, actual 200 ms and 56 Torch threads. Native
median is **26.828 tok/s** (26.604–26.958), stock **21.407** (21.072–22.446),
with exact 177/512 tokens and 288/77 MTP counts. Relative to that fresh control,
the staged 0-ms native median of 33.184 is about **23.69% higher**. This is a
between-process comparison supported by the earlier within-process block-time
interventions, not a claim that all historical absolute-rate variation has been
explained. Final installed-wheel validation will provide the release figure.

The matched-warmup GLM TP recheck is complete. Staged / old-wheel-manual /
staged short medians are **86.781 / 78.282 / 85.877 tok/s**, each from five
measurements after four identical warmups. Ranges are 86.643–87.078,
77.137–85.664, and 85.802–86.527. The old-wheel process itself moved from
85.664/85.189 to 77–78 without a policy or workload change. Full output and
162/120 MTP counts remain identical. Thus the slow state is not specific to
the new initialization code; its precise cause remains unresolved. Do not
interpret these between-process rates as a new TP algorithm speedup.

Release gates passed: 234 native/policy tests, real-runtime unset/0/1/50 checks,
all seven staged cross-model configurations, exact outputs across 50 distinct
workload cohorts, and V4.1's unchanged 26/14 placement, MTP3, 65536 capacity,
`fp16_disk` Engram mode and graph replay. Implementation commit:
`f30d73e37e9396fa71b28297a030df2ba9898416` on the existing feature branch.

The separately versioned Windows wheel is now packaged as
`1.6.6+nativeplan.v2.tp.moe1.ds41.1.omp1` under `omp-policy/package/wheels/`.
SHA-256: `ab08485c17c5be11fb4db03f3bb2730f8e142aa1b044bbb63bb52a3220cbdc59`.
The established `build_py` / `bdist_wheel --skip-build` procedure reused the
byte-identical validated SM89/SM120 extension; no native recompilation or
kernel change was needed. The original release wheel/hash remains preserved.
Extracted-wheel runtime checks and installation validation follow.

## OpenMP release completion and activation

The extracted wheel and the installed Tabby wheel both pass real-runtime
unset/0/1/50 checks. The installed 0-ms run completes five warmed stock/native
pairs, 8K and 64511-token prefill/decode, and a separate telemetry pair. A second
fresh installed process confirms unset behavior and exact outputs. The final
saved-output audit covers **720 requests / 50 cohorts**, with no mismatches.
All 645 installed code payloads and the extension hash match the wheel proof.
The old wheel, unrelated package versions, Tabby configuration/authentication
hashes and the unrelated `convert.py` change are preserved. Temporary servers
and monitors have exited, and port 5008 is free. Consolidated evidence:
[`final-release-summary.json`](../native-results/deepseek-v41/omp-policy/final-release-summary.json).

To select the recommended value, set it in the process environment **before
importing ExLlamaV3 / launching Tabby**, using the existing launch arguments:

```powershell
$env:EXL3_OMP_BLOCKTIME_MS = '0'
```

This command affects the current shell and its child processes. No `setx`,
global CPU policy, persistent Tabby config or authentication change was made.
The scoped launch example in [the policy documentation](openmp_wait_policy.md)
restores the previous shell value afterward. Use `1` for frequent short CPU
parallel regions, or leave the option unset to preserve existing runtime behavior.

Rollback, if needed, uses the preserved original wheel without changing dependencies:

```powershell
D:/tabbyAPI/venv/Scripts/python.exe -I -m pip install --no-index --no-deps --force-reinstall D:/exllamav3/native-results/deepseek-v41/package/wheels/exllamav3-1.6.6+nativeplan.v2.tp.moe1.ds41.1-cp313-cp313-win_amd64.whl
```

The delivered optimization is worthwhile for this V4.1 workload: the installed
native median is **33.162 tok/s**, versus **26.828** on the nearby prior-wheel
control (+23.61%) and **27.086** with the new wheel unset (+22.43%). Separate
native telemetry measures **1.16 CPU cores**, while normal 56-thread CPU
computation remains available. This supports the opt-in worker-wait policy;
it does not justify forcing zero for all CPU-heavy workloads or claiming that
the remaining TP/process variance has been fully explained.
