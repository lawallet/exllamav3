# v1.6.5 R3: heterogeneous Windows TP scheduling investigation

Completed October 27, 2026 on `investigation/v165-r3-tp-scheduling`, based directly
on the validated R1/R7 commit `d62891dee3e4cf233207c0a7ec765adba4b7bc6f`.
Checkpoint tag: `native-plan-v1.6.5-r3-tp-diagnostics-v1`.

**Recommendation: retain the R1/R7 runtime implementation.** Existing graphs
provide a repeatable gain. CPU reduction arithmetic is a small part of the
measured interval; rank waiting, host submission, and process-dependent CPU
placement deserve more attention. The selected-core prototype has mixed
results and is not promoted into a runtime policy. The delivered wheel adds
explicitly activated diagnostics only; no new performance path is enabled.

## Scope and controls

The baseline is the R1/R7 wheel, not the R4 fusion experiment. No HC fusion,
model support, DeepSeek graph policy, collective algorithm, rank dispatch order,
shared-memory lifetime, pointer binding, or graph boundary changes are included.
All existing branches, checkpoints, wheels, and the 22 pre-existing untracked
development artifacts are preserved. Nothing is installed or pushed.

The machine has an AMD EPYC 7663 CPU (56 cores / 112 logical processors), a
96 GiB RTX PRO 6000 Blackwell Max-Q, and two 48 GiB RTX 4090 D cards. CUDA
12.8 / Torch 2.9.0+cu128 / Python 3.13.12 / MSVC 2022 are used under Windows
WDDM on Windows 11 build 26200. Windows exposes processor groups of 64 and
48 logical processors.

| Configuration | CUDA_VISIBLE_DEVICES | Worker rank mapping |
|---|---|---|
| TP2 | `2,0` | rank 0: logical 1 / physical 0 Ada; rank 1: logical 0 / physical 2 Blackwell |
| TP3 | `2,0,1` | rank 0: logical 1 / physical 0 Ada; rank 1: logical 2 / physical 1 Ada; rank 2: logical 0 / physical 2 Blackwell |

Blackwell is the inline output worker in the parent process. Child GPU workers
are dispatched first, followed by that inline worker. The CPU reduction helper
is a separate process. Device UUIDs and actual shard allocation are recorded
for every model load; this is real tensor parallelism, not layer splitting.

Model: `D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw`.
Generation comparisons use batch 1, a fixed 23-token prompt, greedy sampling,
65,536-token target/draft caches, warmed shapes and longest requests, three
repetitions, alternating eager/graph order, and 128 / 1,024 generated tokens.
MTP uses the existing draft on logical CUDA 0 with three draft tokens.
"Eager" here disables the native HC stage graphs only. Upstream attention and
other existing graphs remain enabled in both arms (`EXL3_GRAPHS=1`). Graph
counters in the benchmark JSON refer to native block plans, not every upstream
graph object.
All GPU workloads in this investigation run sequentially. No machine-wide
clock, power, desktop, driver, or installed-package settings are changed.
Baselines use the original CPU placement. A separate opt-in CPU-set experiment
changes only this test's calling threads and restores their original selection.

The baseline SHA-256 is
`a4cc4e2b1fc28059e3436420586fe0707153d49b10e4a341251ab3a8a09d1c88`.
The initial campaign verifies the extracted extension and production source
payload before and after the run. Raw inputs, hashes, placement, counters,
VRAM, timings, outputs, and acceptance statistics are under
`native-results/v1.6.5-r3-tp/baseline/`.

## Initial unprofiled baseline

Values are medians of three runs, in generated tokens/s. Decode throughput
excludes prefill; request throughput includes prefill and request setup.

| Placement | MTP | Tokens | Eager decode | Graph decode | Eager request | Graph request |
|---|---|---:|---:|---:|---:|---:|
| TP2 | off | 128 | 52.04 | 58.44 | 34.77 | 37.75 |
| TP2 | off | 1,024 | 50.95 | 57.76 | 48.00 | 53.95 |
| TP2 | on | 128 | 81.91 | 89.64 | 45.27 | 47.51 |
| TP2 | on | 1,024 | 80.82 | 88.12 | 73.34 | 79.57 |
| TP3 | off | 128 | 50.54 | 58.08 | 33.05 | 36.10 |
| TP3 | off | 1,024 | 50.42 | 57.90 | 47.46 | 54.08 |
| TP3 | on | 128 | 76.87 | 85.19 | 44.04 | 46.83 |
| TP3 | on | 1,024 | 84.23 | 92.85 | 76.72 | 83.18 |

All 48 timed samples have zero native graph captures. Eager has zero replays;
graphs have exactly two replays per native block execution, on every rank.
TP2 completion hashes and MTP acceptance agree across modes and repetitions.
TP3 output trajectories differ because the existing three-contributor reducer
accumulates in arrival order. Its MTP timings therefore include different
acceptance/routing work and are not an exact workload-controlled speedup claim.
Fixed-token target replays supplement those ordinary-generation observations.

For the matched TP2 cases, median paired graph improvements are **12.31% /
12.63%** without MTP and **9.44% / 9.04%** with MTP, at 128 / 1,024 tokens.
These compare existing stage graphs directly to eager native IR.

After the 1,024-token samples, allocated / reserved CUDA memory is:

| Placement | MTP | Ada rank(s), GiB each | Blackwell, GiB |
|---|---|---:|---:|
| TP2 | off | 37.36 / 41.08 | 77.91 / 86.18 |
| TP2 | on | 38.44 / 42.33 | 80.64 / 88.76 |
| TP3 | off | 28.25 / 31.05 | 59.45 / 65.95 |
| TP3 | on | 28.86 / 31.75–31.76 | 62.09 / 68.31 |

Eager and graphs agree to the displayed precision. These are PyTorch allocated
and reserved bytes, not total WDDM/desktop residency. Full per-sample memory
and graph counters are preserved in `baseline/*/results.json`.

## Measurement design

Unprofiled throughput and profiling are separate processes/campaigns. The
profiling window begins after at least 128 generated tokens. MTP can cross
that boundary by a few tokens; actual target and draft shapes/counts are saved.

The new `util/profile_tp_scheduling.py` can collect:

- Reducer-only timestamps without per-layer Python wrapping or CUPTI.
- Per-rank host scopes around selection, native stages, attention/MLP,
  collectives, and parent submission/acknowledgment handling.
- PyTorch/CUPTI kernel, CUDA API, and explicit memcpy traces, with clock
  anchors for correlation to the CPU helper.
- Optional CUDA-event or Nsight/NVTX diagnostics.

The C++ probe is **off by default** and only enabled explicitly by
`tp_profile_begin(capacity)` in the helper thread. It preallocates bounded
thread-local records outside the measured loop, counts dropped records, and
clears the active record through RAII even on an exception. It records queue
waiting, actual sleeps, job sizes, first-chunk arrival observations, arithmetic
brackets, publication, completion, chunk count, and ISA selection. It changes
no shared protocol structures, arithmetic, flags, waits, or CUDA kernels.

Interpretation limits:

- CPU timings are wall-clock brackets, including possible descheduling; they
  are not CPU-cycle counts. Queue waiting can simply mean no job is needed yet.
- Arrival observations are when the helper sees each ready flag, not exact GPU
  write timestamps. The last observed contributor can be affected by polling
  order and CPU scheduling, especially for near-simultaneous arrivals.
- GPU `send` duration includes packing, mapped-host writes, and flag publication.
  GPU `recv` duration includes waiting for the CPU/other ranks and readback.
  Neither is a pure PCIe transfer measurement. Mapped-host loads/stores do not
  appear as ordinary CUDA memcpy events.
- CPU and GPU spans overlap. Nested host scopes, per-rank waits, and reducer
  activity must not be added to produce a fictitious end-to-end total.
- CUPTI plus Python annotations noticeably perturb host submission. Kernel
  category and gap totals describe the traced run, not the unprofiled baseline.
  Reducer-only measurements provide a lower-overhead cross-check.
- No OS scheduling/ETW trace is available in this non-administrator session.
  CUDA API-to-kernel gaps cannot by themselves distinguish Windows scheduler,
  driver batching, dependencies, and device scheduling.

## Source audit

The v1.6.5 reducer already fuses the first two CPU contributions into one
addition pass and parallelizes larger, multi-chunk jobs. Decode-sized jobs
remain single-threaded; changing `EXL3_TP_REDUCE_THREADS` therefore does not
accelerate these small reductions. The single-/multi-chunk threshold is
131,072 wire bytes, as defined by `CPUREDUCE_CHUNK_SIZE`.
The relevant upstream changes identified in the original review are
`8114c493` and `a530c6c0`. The implementation inspected here is the actual
merged R1/R7 baseline, not an assumption about a newer upstream checkout.

Small reductions launch separate send and receive kernels. Each rank publishes
its contribution before the helper adds it, and every rank consumes the
published result. Three-rank arithmetic remains arrival-ordered. Multi-chunk
ring throttling protects accumulator lifetime and cannot be inferred redundant
from an inclusive timing profile.

The baseline already defaults `EXL3_TP_NO_FWD_BARRIER` to `1` in
`model_tp_fn.py`; the absence of a forward-start barrier in these traces is
pre-existing behavior. R3 does not change that default. Deferred acknowledgments
and strong references to shared input/recurrent-state exports are also retained.

## CPU reduction phases and rank imbalance

The final reducer-only probe recorded 79 reductions per target forward. Every
job was single-chunk: 8,192 wire bytes for Q1 and 32,768 for Q4 MTP verification.
The source tensors are FP32 and use the existing BF16 wire format. All three
ranks contribute in the traced TP3 decode case. The measured ISA is AVX2.
No records were dropped.

The table reports medians per reduction and summed arithmetic per target
forward. Q4 target forwards are not individual accepted/generated tokens.

| Configuration | Mode | Target forwards | Arithmetic, us/reduce | Arithmetic, ms/target | Arrival spread, us/reduce | Last observed contributor |
|---|---|---:|---:|---:|---:|---|
| TP2 Q1 | eager | 256 | 1.3 | 0.107 | 67.9 | Blackwell 89.6% |
| TP2 Q1 | graphs | 256 | 1.3 | 0.106 | 59.8 | Blackwell 95.2% |
| TP2 MTP Q4 | eager | 89 | 3.9 | 0.320 | 81.9 | Blackwell 96.1% |
| TP2 MTP Q4 | graphs | 89 | 3.9 | 0.320 | 66.6 | Blackwell 96.9% |
| TP3 Q1, slow process | eager | 256 | 2.5 | 0.206 | 286.8 | first Ada about 95% |
| TP3 Q1, slow process | graphs | 256 | 2.5 | 0.208 | 235.6 | first Ada about 95% |
| TP3 MTP Q4 | eager | 86 | 7.9 | 0.634 | 122.4 | Blackwell 86.5% |
| TP3 MTP Q4 | graphs | 96 | 7.9 | 0.644 | 102.0 | Blackwell 90.2% |

CPU arithmetic accounts for under 3% of the helper-covered target interval in
all these cases. Even that is a generous optimization ceiling: some TP3
arithmetic overlaps waiting for the third contributor, and removing arithmetic
entirely is not a feasible optimization. The large gap is between rank progress,
not between alternative CPU addition implementations.

There were no queue sleeps in seven of these eight windows. The slow TP3 eager
window had three sleeps totaling about 19.4 ms across 256 target forwards
(0.076 ms/forward). That cannot explain its roughly doubled decode latency.
There is no measured basis here for lengthening the spin budget, adding decode
reduction threads, or removing collective waits.

Queue wait / active-job wall time per target, in ms, were respectively:
TP2 Q1 eager 5.25 / 13.20 and graphs 2.15 / 14.33; TP2 Q4 eager 1.49 / 25.13
and graphs 1.34 / 22.16; slow TP3 Q1 eager 2.33 / 33.22 and graphs 1.16 / 28.31;
TP3 Q4 eager 6.59 / 21.10 and graphs 8.48 / 16.33. Active-job time includes
polling for rank flags and possible CPU descheduling; it is not CPU arithmetic.
These intervals overlap GPU execution and cannot be added to GPU timings.

### Intermittent slow TP3 state

One reducer-only TP3 Q1 process ran much slower than the initial baseline,
without any change in shard placement. This also appeared outside the probe:
its post-profile uninstrumented warm samples were 27.11 tokens/s eager and
31.25 with graphs. The next TP3 trace process, using the **same diagnostic
binary and allocation**, warmed at 49.74 / 58.14 tokens/s before tracing began.

The slow case reverses the usual rank ordering: physical GPU 0, the first Ada,
is last for about 95% of reductions. CPU arithmetic remains about 2.5 us, and
rare helper sleeps are too small to explain the change. The source, binary,
allocation, and algorithm are therefore insufficient to predict throughput
across fresh processes on this setup. Do not pool these two operating states
into one speedup number, or attribute the difference to an optimization.

The slow run has no simultaneous OS scheduling trace or per-kernel GPU trace,
so its underlying cause remains unresolved. The later fast TP3 trace records
normal active GPU clocks and PCIe Gen4 x16 links; this does **not** establish
the clocks or link state during the earlier slow run. Primary CPU affinity and
processor snapshots are saved in subsequent diagnostics. They are not a
substitute for a scheduling timeline.

## GPU computation, waiting, submission, and transfers

The following are **instrumented trace times**, in ms per target forward.
They must not be substituted for the unprofiled throughput table. Q1 has one
target row; MTP has four verification rows. The Blackwell MTP computation
column also includes draft/sampling kernels on that process/device.

| Placement / mode | Device | Compute kernels | Reduce send | Reduce recv, inclusive | Other collectives | Uncovered GPU timeline |
|---|---|---:|---:|---:|---:|---:|
| TP2 Q1 eager | Blackwell | 12.20 | 0.24 | 6.46 | 0.16 | 31.91 |
| TP2 Q1 eager | Ada | 6.18 | 0.35 | 32.82 | 6.97 | 4.87 |
| TP2 Q1 graphs | Blackwell | 12.19 | 0.24 | 10.55 | 0.16 | 28.17 |
| TP2 Q1 graphs | Ada | 6.15 | 0.34 | 34.61 | 6.72 | 3.46 |
| TP3 Q1 eager | Blackwell | 10.61 | 0.24 | 14.58 | 0.17 | 36.36 |
| TP3 Q1 eager | first Ada | 5.52 | 0.34 | 43.86 | 7.65 | 4.85 |
| TP3 Q1 eager | second Ada | 5.40 | 0.34 | 44.15 | 7.71 | 4.63 |
| TP3 Q1 graphs | Blackwell | 10.55 | 0.24 | 9.04 | 0.84 | 28.06 |
| TP3 Q1 graphs | first Ada | 5.50 | 0.35 | 25.46 | 5.78 | 11.62 |
| TP3 Q1 graphs | second Ada | 5.39 | 0.34 | 31.66 | 7.07 | 4.26 |
| TP2 MTP eager | Blackwell | 20.95 | 0.61 | 11.80 | 3.92 | 35.80 |
| TP2 MTP eager | Ada | 10.00 | 0.74 | 26.17 | 4.47 | 25.15 |
| TP2 MTP graphs | Blackwell | 20.89 | 0.68 | 2.56 | 0.87 | 36.99 |
| TP2 MTP graphs | Ada | 9.96 | 0.76 | 28.72 | 5.75 | 15.73 |
| TP3 MTP eager | Blackwell | 18.07 | 0.60 | 10.24 | 3.07 | 42.44 |
| TP3 MTP eager | first Ada | 8.53 | 0.74 | 37.60 | 7.23 | 17.91 |
| TP3 MTP eager | second Ada | 8.55 | 0.78 | 34.94 | 6.38 | 21.44 |
| TP3 MTP graphs | Blackwell | 18.64 | 0.61 | 10.80 | 0.99 | 42.12 |
| TP3 MTP graphs | first Ada | 8.71 | 0.75 | 39.86 | 7.17 | 16.09 |
| TP3 MTP graphs | second Ada | 8.62 | 0.77 | 39.49 | 6.69 | 17.04 |

Intervals are unioned rather than blindly summing overlapping kernel durations.
Explicit memcpy intervals are separately recorded in `profile-summary.json`.
The small memcpy category is not displayed in this table. Each rank has its
own observed GPU window; do not add rows across ranks. MTP trace windows have
only 4–7 target forwards and different acceptance trajectories, so their
totals are descriptive, not paired performance estimates.

The trace exposes a genuine imbalance in work: Blackwell runs about twice the
non-collective kernel work of either Ada in TP3 Q1, consistent with its larger
shards. However, this alone does not explain the whole throughput gap. The
Blackwell timeline also has substantial gaps between kernels. In the TP3 Q1
graph trace, 28.06 ms/target is uncovered: the main-stream breakdown is about
18.83 ms before the next CUDA API entry, 7.82 ms inside that API, and 1.45 ms
after its return. Those are **trace-perturbed** wall spans, not an unprofiled
CPU/driver cost decomposition. They identify submission as a profiling target,
not 28 ms of safely removable overhead.

On Blackwell, Q1 ordinary kernel launches fall from 787 to 472 per TP2 target
(788 to 473 at TP3), while graph launches rise from 48 to 138. That is **225
fewer launch API calls per target forward**, with essentially unchanged kernel
computation time. The matched unprofiled graph gain supports the conclusion
that host submission overhead matters on this machine.

Parent command preparation, pipe transmission, and acknowledgment handling
are much smaller than the full inline dispatch span. For example, the TP3
graph trace has about 0.167 ms/target preparing inputs, 0.055 ms in helper pipe
send, 0.039 ms sending pre-pickled GPU commands, and 0.202 ms in acknowledgment
draining. These scopes are nested in larger calls and are not additive.
`tp_worker_dispatch` includes the entire inline output-worker forward; calling
that whole interval "IPC overhead" would be incorrect. Native plan selection
is around 1.5–2.1 ms/target in the CUPTI traces; the lighter host pilot measures
roughly 1.4–1.5 ms on Blackwell for TP2 Q1. Its full pointer/layout guard must
be preserved if this becomes an optimization target.

The earlier host-only TP2 Q1 pilot provides a less intrusive cross-check
(Blackwell, ms/target):

| Host scope | Eager | Graphs |
|---|---:|---:|
| Native plan selection | 1.49 | 1.40 |
| Native begin | 2.51 | 1.30 |
| Attention forward | 4.98 | 4.68 |
| Native middle | 2.26 | 1.03 |
| MLP forward | 7.16 | 7.18 |
| Native finish | 1.02 | 1.04 |
| Complete parent target forward | 23.07 | 20.00 |

These are host wall spans, including CUDA API work and nested waits, not GPU
kernel time or exclusive CPU cycles. Observer/control work outside the shown
scopes is included in the complete forward. The reduction in begin/middle
submission cost agrees with the launch-count change and unprofiled graph gain.
The pilot used the first diagnostic build; its unpopulated ISA metadata field
is disregarded. The final reducer-only and CUPTI results use `4r3profile2` with
validated ISA recording.

### What the transfer measurements actually establish

Every send/receive pair in the final traces matches one recorded helper job.
Median Q1 send kernels are about **2.9 us on Blackwell and 4.4 us on Ada**;
Q4 send kernels are roughly **8–10 us**. This includes packing and flag
publication as well as the mapped-host write.

Correlating CPU publication with GPU receive spans places about **98–99% of
the long Ada receive time before publication** in the Q1 graph traces.
For example, TP3's first Ada has 25.46 ms/target of receive time, approximately
25.12 ms before publication and 0.35 ms afterward. The second Ada has
31.66 / 31.15 / 0.51 ms respectively. Most of that inclusive kernel duration
is therefore waiting for rank/helper progress, not transferring the sum back.

Median publication-to-receive-end intervals are only a few microseconds, but
CPU/GPU clock mapping has an approximately 5–7 us-wide uncertainty interval
(plus a sub-microsecond helper calibration bracket). Do not treat a reported
2–7 us cross-clock interval as a precise readback benchmark. The coarse
waiting-versus-readback distinction is much larger than that uncertainty.

The Q1 reduction payload is 79 x 8 KiB per rank in each direction: about
2.47 MiB/target across TP2 or 3.70 MiB across TP3, excluding flags, transaction
overhead, other collectives, and explicit copies. Payload volume is measurable;
actual PCIe bus utilization is not established by these traces. Neither low
aggregate payload nor long receive kernels proves a link-bandwidth bottleneck.

### Synchronization is a dependency, not automatically wasted work

The generator synchronizes before consuming sampled tokens and using them as
the next forward's input. In the TP3 trace, device synchronization rises from
about 0.026 ms/target eager to 0.605 ms with graphs even though unprofiled graphs
are faster. Faster submission can simply leave more work outstanding at that
dependency. Removing the wait based on its inclusive duration would be wrong.
No synchronization is removed or narrowed in this investigation.

## Matched unprofiled TP3 target replay

`util/benchmark_tp_replay.py` uses the R1/R7 binary and the existing validation
oracles. Both modes execute native IR; only stage graph activation differs.
Every comparison below has three alternating pairs, one identical supplied
target-input hash across all six runs, zero native-stage captures inside the timed region,
and zero eager replays / exactly two graph replays per native execution.
Input hashes and counter deltas are computed outside the timed region.

| Target rows per forward | Timed forwards | Eager rows/s, median | Graph rows/s, median | Median paired graph gain |
|---:|---:|---:|---:|---:|
| 1 | 128 | 29.04 | 32.51 | 12.32% |
| 1 | 1,024 | 48.76 | 54.85 | 12.62% |
| 4 | 128 | 152.62 | 173.28 | 13.86% |
| 4 | 1,024 | 101.62 | 113.44 | 11.63% |

These are target-only replay measurements with a fixed 128-token context,
four untimed warm forwards, explicit logits/argmax handling, and forced input
continuations. Q4 repeats the selected token across four input positions, as
the established validation harness does. It measures four-row target work,
not draft-model execution or MTP acceptance. Each row is a separate fresh
process and workload; do not compare absolute rates across rows as an isolated
context-length or scheduling effect. The ordinary-generation table separately
covers 128 / 1,024 generated tokens with real MTP.
Unlike that generation matrix, these supplemental replays warm four forwards
per sample rather than an entire longest request; they do not claim a global
count of upstream graph captures.

Fixed tokens control the supplied inputs, but TP3's existing arrival-ordered
arithmetic can still vary intermediate states/routing. Exact local-stage
oracles and five fixed-token reference repeats are used before timing; the
noise-envelope coefficients are unchanged. No TP algorithm is altered to
force determinism.

## Optional CPU placement experiment

An explicit `EXL3_R3_PIN_THREADS=1` enables a test-only comparison of existing
stage graphs with original placement versus separate physical cores for the
CPU helper and GPU submission threads. The topology API identifies group-zero
logical CPUs 16, 18, 20, and 22 as four distinct physical cores. The helper uses
16; logical CUDA devices 0, 1, and 2 use 18, 20, and 22 respectively. CUDA driver
threads and other applications are not assigned or reserved cores.

The experiment uses Windows selected CPU sets rather than hard group affinity.
Windows 11 can schedule default threads across processor groups; a hard
`SetThreadGroupAffinity` call would narrow that behavior. Selected CPU sets can
instead be saved and restored, including clearing an originally empty
selection. The isolated API check verifies assignment and exact restoration.
These are soft scheduling assignments, not exclusive reservations. See
[Microsoft's processor-group rules](https://learn.microsoft.com/en-us/windows/win32/api/processtopologyapi/nf-processtopologyapi-setthreadgroupaffinity)
and [selected CPU-set semantics](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadselectedcpusetmasks).

The experiment is confined to `native-results/v1.6.5-r3-tp/affinity.py`; no
production affinity policy is added. Both comparison arms run existing graphs.
For compatibility with the original benchmark driver, raw mode `native` means
original CPU placement and raw mode `graphs` means selected CPU sets; each
output includes an explicit label mapping. The run asserts zero timed captures
and two graph replays per native execution in **both** arms.

Both arms are warmed before timing, including a full longest request. Clearing
the selected CPU sets restores eligibility but does not return a thread to its
previous physical CPU. This alternating steady-state comparison can therefore
have placement carryover. It does not test whether pinning from process start
prevents the intermittent slow TP3 state, and small differences from three pairs
are not a statistical guarantee for other cores, processes, or desktop loads.

The completed generation comparison is below. Both columns execute the
**original R1/R7 wheel with existing stage graphs**. Rates are generated
tokens/s, medians of three runs. The last column is the median of the three
paired percentage changes, not the ratio of the two median rates.

| Placement | MTP | Tokens | Original CPU sets | Selected cores | Paired change |
|---|---|---:|---:|---:|---:|
| TP2 | off | 128 | 59.06 | 60.03 | +1.65% |
| TP2 | off | 1,024 | 59.14 | 59.80 | +1.12% |
| TP2 | on | 128 | 90.38 | 92.31 | +2.41% |
| TP2 | on | 1,024 | 89.55 | 91.03 | +1.71% |
| TP3 | off | 128 | 57.96 | 48.37 | -16.68% |
| TP3 | off | 1,024 | 58.38 | 48.40 | -17.09% |
| TP3 | on | 128 | 54.80 | 83.52 | +55.14% |
| TP3 | on | 1,024 | 61.00 | 83.91 | +40.14% |

TP2 output hashes and MTP acceptance match across both arms and the earlier
campaigns. Its gain is modest. TP3 trajectories differ, so its percentage
changes are **descriptive observations, not matched-workload causal speedups**.
The sign reversal between TP3 decode and MTP argues strongly against adopting
this one core map as a general policy. The selected-core TP3 MTP median is also
below the initial graph baseline's 92.85 tokens/s at 1,024 tokens, under another
trajectory/process; the large within-process improvement is not a new universal
throughput record. CPU placement is worth diagnosing, but these data do not
identify a generally beneficial assignment or establish the cause of the slow
TP3 state.

A separate TP3 Q1 replay uses the original wheel, identical supplied target
inputs across all six samples, and existing graphs in both arms. It runs
1,024 timed forwards after a 128-token context and four untimed warm forwards,
with the same target-replay limitations described earlier.

| Pair | Original CPU sets, rows/s | Selected cores, rows/s | Change |
|---:|---:|---:|---:|
| 0 | 55.76 | 46.12 | -17.28% |
| 1 | 55.86 | 46.73 | -16.35% |
| 2 | 47.62 | 46.70 | -1.94% |
| Median | 55.76 | 46.70 | -16.35% paired |

All samples have zero native-stage captures and two graph replays per native
execution. The exact local-stage checks and unchanged TP3 noise-envelope
checks against five fixed-token repeats pass. This supports rejecting this
particular selected-core map for TP3 Q1 even after controlling supplied tokens.
The third original-placement sample also slows markedly, illustrating why
process/placement variability must remain visible. Internal TP3 arithmetic and
routing can still vary; no whole-model bitwise-determinism claim is made.

All 170 GPU-worker placement snapshots in the selected arms of the generation
matrix were on their assigned CPUs. CPU-set API round trips also verify the
requested masks and restoration. These are point observations outside timed
regions, not proof of continuous CPU residency. Details are retained in
`affinity-replay-summary.json`, the raw placement logs, and `audit.json`.

## Correctness and diagnostic-wheel validation

The final diagnostic build passes **673 tests and 180 subtests** in the native
IR and TP regression suite. The five focused profiling-contract tests also
pass separately. Those check explicit activation, thread isolation, bounded
storage/drop accounting, timestamp structure, and exact TP2/TP3 reductions at
small, threshold, multi-chunk, and ring-wrap sizes. The arithmetic inputs for
the TP3 collective-contract test are exactly representable integers so its
reference does not depend on arrival order.
The suite's intentionally skipped long-round test was then run separately with
`--slow` and passed. It verifies that a reduction round lasting longer than the
ordinary in-pass wait timeout is still valid; the per-job timeout contract is
unchanged.

Real-model validation passes on both TP2 and TP3, with two load cycles, Q1/Q4
target checks, MTP generation, and prompt-cache reuse. TP2 logits, prefill,
rank traces, and cache/recurrent state are exact. TP3 passes the existing exact
local-stage oracle and the unchanged noise-envelope coefficients against five
fixed-token reference repeats per case; it is not claimed globally bitwise
deterministic. Both Ada cards and Blackwell participate in those TP3 checks.
The real-model matrix here is batch 1 with Q1/Q4, not a new B2–4 performance
qualification.

The experimental wheel is a **diagnostic wheel**, with CPU timing disabled
unless explicitly activated through its API:

```text
exllamav3-1.6.5+nativeplan.v2.tp1-4r3profile2-cp313-cp313-win_amd64.whl
wheel SHA-256: bc4e1613573ae8b0c35f18eb16419ab569f273b27e74f64e259ff3d3f4cbbb85
extension SHA-256: 8ddd3c2527eabeb94f0d7ce9ad958acd9ba758345eaf3cbf3c3413952c709e5d
```

The wheel was extracted rather than installed. Its complete source payload
matches the working production sources; its binary was used for the final
regressions, model validation, reducer-only measurements, and GPU traces.
`cuobjdump` verifies Ada `sm_89` and Blackwell `sm_120` images. The EPYC executes
the AVX2 reducer; the AVX-512 instrumentation is compiled but not exercised on
this CPU. The original R1/R7 and R4 artifacts remain intact.

The only existing production files changed are the extension bindings and
three CPU-reducer implementation files, plus a new bounded-record header.
The Python model/TP implementation, native plan guards, HC kernels, graph
implementation, graph-disable policies, and pointer rebinding code are unchanged.

## Diagnostic wheel with profiling disabled

A second, fully unprofiled generation campaign repeats the initial settings
with the final diagnostic binary. Each entry below is a three-run median in
generated tokens/s; these are separate processes from the original campaign.

| Placement | MTP | Tokens | Original eager | Diagnostic eager | Original graphs | Diagnostic graphs |
|---|---|---:|---:|---:|---:|---:|
| TP2 | off | 128 | 52.04 | 51.24 | 58.44 | 58.45 |
| TP2 | off | 1,024 | 50.95 | 51.29 | 57.76 | 57.97 |
| TP2 | on | 128 | 81.91 | 82.96 | 89.64 | 91.00 |
| TP2 | on | 1,024 | 80.82 | 81.60 | 88.12 | 89.50 |
| TP3 | off | 128 | 50.54 | 51.09 | 58.08 | 58.94 |
| TP3 | off | 1,024 | 50.42 | 51.12 | 57.90 | 58.95 |
| TP3 | on | 128 | 76.87 | 69.36 | 85.19 | 75.28 |
| TP3 | on | 1,024 | 84.23 | 78.25 | 92.85 | 79.82 |

All 48 diagnostic samples have zero native-stage captures and the expected
replay counts. The TP2 output hashes and MTP acceptance agree with the original
campaign. Its measured differences are small (about -1.5% to +1.7%) and do not
establish a benefit from the inactive probe. TP3 MTP is lower in this campaign,
but output/acceptance trajectories differ; this is not an exact-workload
regression test. The previously observed fast and slow TP3 processes using the
same binary also prevent attributing all cross-process variation to code.
No claim of universally zero inactive-probe overhead is made.

## Final audit and delivery

The preservation audit verifies all 22 original untracked development artifacts
byte-for-byte and all 110 other Git refs in the initial snapshot. The
R1/R7 and R4 branch heads are unchanged. The R3 branch is the only branch advanced
by this work. Wheel CRC, complete payload hashes, production-source agreement,
and both CUDA architecture images are verified.

The audit also checks 144 unprofiled generation samples and 30 target-replay
samples, native graph counters, TP2 output/acceptance parity across campaigns,
TP3 local correctness, both model load cycles, and zero dropped records in the
final helper/CUPTI observations. The full suite, separate profiling-contract
tests, and long-round test pass.

The delivery directory is `dist/native-ir-tp-1.6.5-r3-tp-scheduling/`. It contains
the separately tagged diagnostic wheel, this report, summaries, validation
evidence, reproduction scripts, a checkpoint record, and `SHA256SUMS.txt`.
Large raw traces stay under `native-results/v1.6.5-r3-tp/`. Nothing is installed
or pushed, and no R3 changes are merged into R1/R7 or R4.

## Optimization opportunities and limits

The recommendation is to **retain the validated R1/R7 implementation and its
existing graph/scheduling policies**. Keep R3 as a diagnostic branch and keep
the selected-core experiment opt-in. There is no demonstrated general-purpose
scheduling improvement here that warrants a runtime default or a collective
change. The useful result is narrower attribution: small CPU addition cost,
substantial rank waiting, measurable submission savings from the existing
graphs, and significant process/placement sensitivity that still needs an OS
scheduling timeline to explain.

| Candidate | Evidence | Appropriate next step |
|---|---|---|
| Reduce Python work in native-plan selection | About 1.4–1.5 ms/target in the lighter TP2 host pilot | Profile the individual guard/key-building operations with a bounded observer. Any later optimization must retain the complete pointer, layout, version, and lifetime checks. This investigation does not bypass or cache away those checks. |
| Improve heterogeneous work balance | TP3 Blackwell runs about twice the non-collective kernel time of either Ada in Q1; it is usually the last contributor outside the anomalous slow process | A separate fixed-input study of supported shard allocations could test the tradeoff between compute balance and VRAM. No sharding algorithm or allocation policy is changed here. |
| Understand process-dependent rank starvation | One TP3 process falls to 27/31 tokens/s while the next identical binary warms at 50/58; the slow rank changes to the first Ada | Capture CPU scheduling and GPU timelines together when the slow state occurs, with a matched unprofiled control. A later administrative ETW/Nsight session would distinguish descheduling from driver/device delays better than inclusive timers. |
| Further batch host submission | Existing HC graphs remove 225 launch API calls/target and improve matched TP2 generation and TP3 target replay | Preserve the existing stage boundaries and guards. This evidence motivates investigation, but does not authorize broad scheduling changes or another fusion experiment. |
| Faster small CPU reductions or more reduction threads | Arithmetic is only 0.1–0.6 ms/target; all measured decode jobs are single-chunk | Low priority. The upstream two-source optimization is already present, and the multithreaded path does not apply to these payloads. |
| Remove waits or tune PCIe based on long receive kernels | Most long Ada receive spans occur before helper publication; transfer time is not isolated precisely | No change justified. Keep flag ordering, ring lifetime protection, acknowledgments, and token dependencies intact. |

These are measured leads, not implemented performance promises. R3 adds
diagnostic visibility and a test-only CPU-placement experiment. It does not
introduce a collective redesign, a new default, or a new model architecture.

## Reproduction and evidence

The full raw campaign is in `native-results/v1.6.5-r3-tp/`. Its
`REPRODUCE.md` gives commands using the extracted wheels without installation.
`baseline.py` and `campaign.py` retain the exact model arguments, visible-device
mapping, run order, and phase logs. Each generation result includes completion
hashes, acceptance counts, graph deltas, and per-rank memory.

`baseline-summary.json`, `campaign-summary.json`, `replay-summary.json`, and
`profile-summary.json` retain individual timing samples and detailed phase
statistics. CUPTI traces and clock anchors are in `torch/`; bounded helper
records are in `helper/`. `models/` contains full correctness/state evidence
for both load cycles, and `gates-v2/` contains the test logs and JUnit results.
The initial pilot is retained separately and is not the delivered wheel.

Useful source entry points are
[CPU job dispatch](../exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu#L107),
[CPU reduction phases](../exllamav3/exllamav3_ext/parallel/all_reduce_cpu_avx2.cpp#L536),
[bounded probe](../exllamav3/exllamav3_ext/parallel/tp_profile.h),
[observer](../util/profile_tp_scheduling.py),
[fixed-input replay](../util/benchmark_tp_replay.py), and the unchanged
[TP dispatch/lifetime logic](../exllamav3/model/model_tp.py#L770).
