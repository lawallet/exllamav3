# R4 experiment: upstream HC fusion inside GLM native stage graphs

Date: 2026-10-27. Experimental branch: `experiment/v165-r4-hc-fusion`.
Base: R1/R7-fixed commit `d62891dee3e4cf233207c0a7ec765adba4b7bc6f`.
Scope: R4 from [the architectural review](native_plan_v165_ir_review.md).

**Recommendation: retain the existing GLM stage graphs.** The default-off
prototype passes the exact kernel/local-stage safety checks, but every matched
TP2 pair was slower than existing graphs. TP3 throughput observations have
different output trajectories and do not establish a controlled gain. Keep
this implementation on the experimental branch; do not promote or enable
fusion by default.

## Implementation

The experiment reuses upstream v1.6.5's existing `hc_apply_partials_kernel`
and norm-enabled `hc_mix_finalize_kernel`. Their device arithmetic is unchanged.
An internal `hc_mix_fused_gr()` adapter supplies the capture stream and records
the dynamic residual and boundary-output arguments for graph replay. The existing
Python `hc_mix_fused()` signature and non-graph behavior are preserved.

| Stage | Existing graph | Experimental fused graph |
| --- | --- | --- |
| Begin | mix partials, mix finalize, RMSNorm: 3 kernels | mix partials, finalize with RMSNorm: 2 kernels |
| Middle | HC apply, mix partials, mix finalize, RMSNorm: 4 kernels | apply with mix partials, finalize with RMSNorm: 2 kernels |
| Finish | HC apply, eager | Unchanged |

Fusion stays entirely inside existing stages. Attention, MLP, collectives,
cross-block residual handling, and the attached MTP draft remain unchanged.
There is no deferred HC update across a TP boundary. Collapsed outputs and
per-plan scratch still exist; this is launch fusion, not scratch elimination.

The graph adapter admits only the existing native HC specialization: four
streams, hidden width 4096, FP32 residuals, FP16 projection weights, at most
12 rows, FP16/BF16 norm weights, and FP16/FP32 boundary outputs. Python retains
the existing B1–4, Q1–6, B×Q≤12 guards. The adapter rejects an incompatible
chunk layout before launching rather than silently recording an extra apply
node with an incorrect pointer-patch list.

## Activation and safeguards

Fusion is **off by default**. In the experimental wheel, opt in before process
startup with `EXL3_NATIVE_TP_HC_FUSION=1`. This only affects GLM native TP plans
when native stage graphs are enabled. `EXL3_NATIVE_TP_GRAPHS=0` or
`EXL3_GRAPHS=0` executes the existing unfused eager implementation. DeepSeek's
native HC graph policy remains disabled, even with the fusion switch set.

Each plan owns separate ordinary and fused begin/middle graph slots, including
both middle boundary dtypes. This permits warmed A/B switching without
recapture. Both variants share the existing plan-owned weights and scratch;
their captured pointers cannot outlive those objects. Dynamic residual and
middle-boundary pointers are patched on every replay. Existing binding checks,
R7 norm validation, scoped `GraphCapture` cleanup, current-stream launch, and
device-aware graph destruction remain in force.

Existing `graph_captures`/`graph_replays` counters count both variants. New
`fused_graph_captures`/`fused_graph_replays` counters identify actual fused work.
The validation/benchmark driver checks these counters rather than assuming
that setting an environment variable selected the path.

Implementation: [block plan](../exllamav3/exllamav3_ext/libtorch/block_plan.cpp),
[HC launch adapter](../exllamav3/exllamav3_ext/hc_mix.cu), and
[GLM-only activation](../exllamav3/modules/native_tp.py).

## Kernel and graph validation

The isolated experimental binary passed **412 tests plus 377 subtests**, with
no failures or skips, followed by **171 HC tests on Ada**. The primary process
exposes physical devices `2,0,1`: RTX PRO 6000 Blackwell and two RTX 4090 D Ada
cards. Native stage tests iterate all three devices; shared HC tests use
Blackwell and are repeated explicitly on Ada.

- All supported B/Q combinations, both norm dtypes, both boundary dtypes,
  and repeated calls with retained, different boundary storage match eager
  stages exactly, including residual updates and finish.
- Switching between ordinary and fused graphs preserves exact results and
  both capture sets. Native graph disabling stops replay.
- Malformed dynamic operands, rebound fixed storage/layout, exposed-scratch
  changes, and invalid rebound norm weights reject before residual mutation.
- Non-default streams, noncontiguous boundary outputs, in-place weight value
  changes, and destruction with a different current device pass.
- A fresh-process matrix covers all 27 unset/0/1 combinations of the global,
  native-graph, and fusion switches. It checks exact outputs and capture/replay
  counts for direct plans, GLM activation, and unchanged eager DeepSeek policy.
- Upstream primitive tests compare streams, post gates, combine matrices,
  collapsed outputs, and normalized outputs with exact equality.

See [fusion safety tests](../tests/test_native_plan_fusion.py) and
[flag matrix](../tests/test_native_graph_policy_process.py).

## Real-model validation and measurements

The campaign uses `GLM-5.3-Flash-Uncensored-EXL3-3.05bpw`, 65,536-token caches,
the existing native TP backend, and unchanged row limits. TP2 uses physical
devices `2,0`; TP3 uses `2,0,1`. The checkpoint occupies 126,265,545,428 bytes,
so an Ada-only TP2 load does not fit the two 48 GiB cards. Both GPU families
are exercised independently by the kernel/stage tests and together by TP.

TP2 passed both load cycles: eight exact logit/prefill/trace/state cases and
eight batch/MTP generator comparisons, including prompt-cache reuse. Final
unload returned parent allocations to 138,789,888 bytes after collection.

The first TP3 attempt failed the existing whole-model noise envelope at B2.
Its local stages were all exact and ranks agreed. Native-vs-stock maximum
logit error was 2.38671875, while one stock repeat measured 1.060546875; the
existing bound was therefore 2.22109375. RMSE was 0.15566084 versus stock's
0.15137322 and passed its separate bound. This failed run is retained under
`models/tp3`; it is not counted as a validation pass.

To characterize this variation, the validator now accepts an optional
`--stock-noise-repeats` count (default remains one). The follow-up preselects
five stock repeats for every case on the same forced tokens and records all
five errors, using their maximum for the existing noise envelope. The original
coefficients and exact local-stage requirements are unchanged. This is a
measurement of stock reduction-order variability, not a relaxation of fusion
parity or a change to TP reductions.

The five-repeat TP3 follow-up passed both load cycles: eight cases with exact
local stages/rank agreement and all noise-envelope checks, plus eight B1–4
MTP generator cases with exact local-stage checks. For example, B2's first
follow-up cycle measured stock maximum error 2.2998046875 and fused-vs-stock
error 2.431640625. Whole-model TP3 logits/completions are **not** certified
bit-identical across independent executions. Final unload left 148,224,000
bytes of parent allocations after collection; worker processes were unloaded.

| Model gate | TP2 Blackwell + Ada | TP3 Blackwell + two Ada |
| --- | --- | --- |
| B1–4 decode, two load cycles | 8 exact global cases | 8 exact local-stage cases; measured stock-noise envelope passed |
| B1–4 attached-MTP generation, two load cycles | 8 exact comparisons | 8 cases passed exact local-stage checks |
| Unload/reload | Passed | Passed |
| Native/fused graph accounting | Two fused stage replays per native block execution | Same, on all three ranks |

B4/Q4 verification still falls back to stock because 16 rows exceed the
12-row native limit; eligible B4 calls remain native. A passing B4 MTP case
does not mean that every verification call uses fusion.

## Benchmark method

Each configuration compares `native` (eager IR), `graphs` (existing stage
graphs), and `fused` (experimental fused graphs) in the same loaded model and
binary. Generation uses batch one, a fixed chat prompt, greedy sampling,
fixed-length requests with no early stop, the same cache capacity/placement,
and unchanged draft settings. Standard decode and attached MTP are separate
configurations, each on TP2 and TP3.

Every mode is warmed for 128 tokens and through the full 1,024-token request
before timing. There are three balanced rounds per length, rotating mode order
so each mode occupies every ordinal position once. Each request gets a fresh
generator/page table. The timed runs have no correctness tracing or profiling.
Assertions require zero new graph captures and the correct fused replay
delta. Outputs and draft acceptance are recorded to identify changed workloads.

`decode_tps` uses the generator's generation interval; wall throughput,
per-request times, acceptance counts, completion hashes, graph counters, and
per-rank allocated/reserved/free memory are retained in each raw sample.
The comparison of interest is fused graphs **against existing graphs**;
eager IR is an additional reference. Performance at batches 2–4 is outside
this timing campaign; those batch sizes are covered by correctness checks.

## Throughput results

Values are median decode tok/s across three rounds. The percentage is the
median of the three **paired** fused/existing-graph changes, not the ratio of
the two column medians. All TP2 completion hashes and acceptance counts match
across modes and rounds at each length.

| Placement / workload | Tokens | Eager IR | Existing graphs | Fused graphs | Fused vs existing, paired median | Paired range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP2 standard | 128 | 50.20 | 57.39 | 56.81 | -0.38% | -2.48% to -0.12% |
| TP2 standard | 1,024 | 50.23 | 57.11 | 56.40 | -1.25% | -2.27% to -0.79% |
| TP2 MTP | 128 | 80.32 | 87.71 | 82.93 | -5.45% | -5.68% to -0.81% |
| TP2 MTP | 1,024 | 79.24 | 86.89 | 86.68 | -0.15% | -0.48% to -0.10% |

TP3 observations are below. **None of these four configurations is output-
matched across the timed samples**, so these rates must not be interpreted as
controlled fusion gains. Each row still uses the same model placement,
generation settings, length, warmup, and balanced order.

| TP3 workload | Tokens | Eager IR | Existing graphs | Fused graphs |
| --- | ---: | ---: | ---: | ---: |
| Standard | 128 | 31.17 | 34.31 | 35.04 |
| Standard | 1,024 | 31.06 | 34.53 | 35.00 |
| MTP | 128 | 56.34 | 64.74 | 64.60 |
| MTP | 1,024 | 60.76 | 66.65 | 64.90 |

TP3 MTP illustrates the confound directly: in the second 1,024-token round,
existing graphs accepted 684 draft tokens and rejected 336, while fused graphs
accepted 656 and rejected 448. Their target workloads differ. Standard TP3
also produces different completions, potentially changing expert routing.
The modest observed standard-decoding advantage is therefore insufficient
evidence to promote fusion. The matched TP2 measurements support no benefit.

## Graph accounting and VRAM

All **72 timed samples** completed with **zero new ordinary or fused captures**.
Every graph-mode sample recorded exactly two stage replays per native block
execution. Eager samples recorded zero replays. Existing-graph samples recorded
zero fused replays, and fused samples counted every replay as fused. Counts
below are per rank, identical across ranks within a sample.

| Workload | Tokens | Existing graph replays | Fused graph replays |
| --- | ---: | ---: | ---: |
| TP2/TP3 standard | 128 | 11,520 | 11,520 |
| TP2/TP3 standard | 1,024 | 92,160 | 92,160 |
| TP2 MTP | 128 | 3,960 | 3,960 |
| TP2 MTP | 1,024 | 32,220 | 32,220 |
| TP3 MTP | 128 | 4,050–4,230 | 3,870–4,320 |
| TP3 MTP | 1,024 | 30,600–32,310 | 31,320–33,120 |

The TP3 MTP ranges reflect different acceptance/work, not missing replay.
Each configuration captured 90 graphs per rank when first warming the existing
variant and another 90 when warming the fused variant. The subsequent long
warmups and timed samples required no new captures.

Warmed memory snapshots, GiB, in reported rank order:

| Configuration | GPU order | Torch allocated | Torch reserved | CUDA free |
| --- | --- | --- | --- | --- |
| TP2 standard | Ada, Blackwell | 37.358 / 77.912 | 41.078 / 86.180 | 5.114 / 7.341 |
| TP2 MTP | Ada, Blackwell | 38.444 / 80.643 | 42.330 / 88.764 | 3.859 / 4.751 |
| TP3 standard | Ada, Ada, Blackwell | 28.250 / 28.251 / 59.451 | 31.047 / 31.051 / 65.953 | 15.260 / 15.256 / 27.793 |
| TP3 MTP | Ada, Ada, Blackwell | 28.861 / 28.860 / 62.085 | 31.750 / 31.764 / 68.309 | 14.512 / 14.498 / 25.432 |

Within each configuration, reserved and free memory were identical across all
timed snapshots; allocated memory varied by at most 17,920 bytes across
lengths/modes. There is no demonstrated steady-state VRAM saving. Both graph
variants are deliberately resident for A/B switching, so these figures describe
the shared experiment process, **not exclusive-mode graph allocation costs**.
`cudaMemGetInfo` also includes allocations outside Torch and is not a precise
graph-memory accounting API. Raw before/after warmup and timing snapshots are
retained. After the campaign, `nvidia-smi` reported 15 / 1,423 / 18 MiB used on
physical GPUs 0 / 1 / 2, with the model processes unloaded.

## Interpretation and limits

Fewer graph nodes do not guarantee faster execution. Upstream's folded
finalize handles a whole row in one block and adds shared-memory/reduction
work; the apply/partials fold also changes block parallelism and synchronization.
These are plausible costs visible in the implementation. This campaign does
not contain kernel-level profiling and does not attribute the observed
throughput changes to a specific hardware bottleneck.

The tested combination folds both apply and norm within the existing stages.
It does not establish that every possible partial-fusion or shape-specific
variant would behave the same. The evidence supports retaining the current
implementation, rather than adding selection heuristics based on these runs.

Validation covers this Windows CUDA build on Ada/Blackwell and the local GLM
checkpoint. It does not certify Linux/ROCm, other checkpoints, full-64K prompts,
TabbyAPI HTTP serving, or independent simultaneous calls on one plan. The
65,536-token figure is cache capacity. No architecture support, DeepSeek graph
policy, collective algorithm, or default fusion setting changed.

## Build and evidence

The build uses Python 3.13.12, Torch 2.9.0+cu128, CUDA 12.8, and MSVC 2022.
`cuobjdump` confirms 169 images each for sm_89 and sm_120. It reuses the local
compiler object cache and writes a separate build library and wheel; prior
wheels, extracted packages, branches, and checkpoint tags remain intact.

Wheel: `exllamav3-1.6.5+nativeplan.v2.tp1-3r4hcfusion-cp313-cp313-win_amd64.whl`.
The separate `3r4hcfusion` wheel build tag identifies this experiment; package
version remains `1.6.5+nativeplan.v2.tp1`. ZIP integrity, every packaged
production source file, and isolated package/extension imports were verified.

Wheel size: **32,438,144 bytes**. SHA-256:
`5a23c655dfe85f9289ce20ee05e2d09ad56352931415f38b61870d81e0840240`.
Extension SHA-256:
`3217e605f4d14b818f5af7a5cbf46b34621f04c8c40c06162aea66321cb449b0`.

The separately tagged experimental bundle is
`dist/native-ir-tp-1.6.5-r4-hc-fusion`, with the wheel, this report, summary,
package proof, final audit, checkpoint metadata, and `SHA256SUMS.txt`.
Git checkpoint: `native-plan-v1.6.5-r4-hc-fusion-experiment-v1`.
This artifact preserves the experiment for reproduction; it is not a
recommendation to enable the fused path.

Evidence directory: `native-results/v1.6.5-r4-hc-fusion`, including the build
script/logs, extracted-wheel proof, sequential campaign driver, full test logs,
source/binary hashes, model parity artifacts, and raw benchmark samples.
These are ignored local artifacts and are not guaranteed to accompany a clone.
The final audit confirms every production source still matches the verified
wheel and all 22 pre-existing untracked development artifacts retain their
hashes. Earlier branches and checkpoints are preserved. No package was
installed and nothing was pushed.
