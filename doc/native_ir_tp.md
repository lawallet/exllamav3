# Native IR with existing tensor parallelism

## Workspace and runtime provenance

Development starts from `native-plan-v1.5.4-glm-regression`, commit
`311e156fae3bf2bd7beecb17fbe93e90274e74ef`, in `D:\exllamav3`.
The original checkout has an unstaged three-line `convert.py` change, no staged
changes, and untracked task prompts, `doc/glm_v154_performance_investigation.md`
and `util/diagnose_glm_v154.py`. None are moved, stashed or modified.
Implementation was developed in the sibling worktree `D:\exllamav3-native-ir-tp`
on `feature/native-ir-tensor-parallel`. Following validation, the user explicitly
requested a compiled wheel, a commit, and checkout of that branch in
`D:\exllamav3`. The original branch is retained at its starting commit.
The user subsequently requested installation into TabbyAPI for a two-GPU
backend check. No upstream merge is included.

At the start, TabbyAPI imported the installed `1.5.1+nativeqwen.concurrent1` wheel, not this
checkout. Its extension has no `BC_BlockPlan` API. The existing development
branch is already based on 1.5.4; tests use its previously built extension at
`D:\exllamav3\native-results\v1.5.4-integration\extension`, through an isolated
process import path. The initial validation reused that binary. The subsequent
user-requested release build compiles a separate extension and packages it with
the TP-aware Python code. After extracted-wheel validation, the user authorized
installing it into TabbyAPI; the previous wheel and exact installed files were
backed up first. Earlier build outputs remain unchanged. Verified dependencies:
Python 3.13.12, Torch 2.9.0+cu128,
CUDA runtime 12.8, triton-windows 3.8.0.post28.
The installed Python package also has Qwen4Exp/GLM/Gemma TP capability flags,
DSV4 and GDN TP export/import, and the inherited TP capability for Devstral.
The integration uses only those pre-existing architecture capabilities, with
the source branch's existing implementation; no upstream code was fetched.

Physical devices are 0/1: RTX 4090 D (49140 MiB each), 2: RTX PRO 6000
Blackwell Max-Q (97887 MiB). Test processes select physical devices using
CUDA_VISIBLE_DEVICES; implementation uses only upstream rank-local devices.

## Audit and design

`TransformerBlock.forward` dispatches plans emitted by `modules/native_plan.py`.
The extension resolves instructions to native function pointers in
`block_plan.cpp` and `block_plan_v2.cpp`; CUDA operations use a device guard and
the current stream. Plan scratch belongs to each block/shape. Original guards
reject TP, and TP export/import previously omitted registration entirely.

Upstream TP allocates shards through TPAllocator, exports modules, and rebuilds
them in spawned workers plus the output-device pseudo-worker. Attention owns
head slicing, its KV/recurrent cache and output reduction/broadcast. MLP/MoE
owns channel/expert slicing, routing, EXL3 launches and output reduction.
Norm and residual stream tensors are replicated. Native plans must never reduce
these again. QSA and other single-owner attention use upstream broadcasts;
empty local shards must still participate. Worker exceptions and acknowledgments
remain governed by model_tp_fn/model_tp. Native and NCCL backends remain intact.

The integration carries only an existing native registration identifier across
TP import. Plans bind imported tensors, never parent pointers. Qwen's fused
ROUTE/MOE instructions cannot be used around distributed experts: emit its
existing GR/APPLY instructions with two Python boundaries instead, so upstream
routing and reduction still execute. Other families retain their existing plan
instruction sequences. No kernel, quantization or allocation strategy changes.

## Architecture intersection at the starting commit

| Internal architecture | Native IR | Upstream TP | Scope |
| --- | --- | --- | --- |
| Qwen4ExpForConditionalGeneration (Qwen3.8) | Yes | Yes | GR boundaries |
| Glm5NextForConditionalGeneration | Yes | Yes | HC boundaries |
| DeepseekV4ForCausalLM | Yes | Yes (inherited capability; DSV4 import exists) | HC boundaries |
| Gemma4ForConditionalGeneration text | Yes | Yes | Four RMS norms |
| Ministral3ForCausalLM (Devstral 2) | Yes | Yes (inherited) | Pre-norm residual |
| LagunaForCausalLM | Yes | Yes (inherited) | Excluded at user request: checkpoint removed |
| Qwen3NextForCausalLM | No matching native registration | No | Excluded |
| Other upstream TP architectures | No native registration | Various | Excluded |

The earlier [Gemma v1 assessment](gemma4_ir_assessment.md) was negative, but
the later [v2 report](native_ir_v2_report.md) confirms its implemented and
registered native path. The [Devstral report](devstral2_ir_report.md) also
retains its generic IR implementation while explicitly reporting near-neutral
throughput. Both implementations already existed at the starting commit.
Including them here establishes TP compatibility; it does not imply IR is
necessary or faster for every model. No new native model family is introduced.

MTP draft components do not themselves support TP. Target TP with a separately
loaded draft remains an upstream protocol; this does not add draft TP support.
Large prefill shapes and instrumentation/override modes retain explicit stock
fallback, matching the existing small-row IR domain.

The user requires a maximum context of **65536** for testing. The validation
driver allocates a 65536-token cache (also for the MTP draft). Concurrent
sequences share this capacity; test prompt length is reported separately.

## Use and diagnostics

Enable `EXL3_NATIVE_PLAN=1` and load with the existing `tensor_p=True` option.
No GPU IDs, allocation weights or backend settings are added by this feature.
Single-device non-TP execution retains its existing architecture switches.
TP-off and IR-off execution remain available through the original controls.

Call `model.native_plan_tp_diagnostics()` outside a timed region to synchronize
and inspect each worker's rank, GPU UUID, backend, local shard dimensions,
constructed plan types, completed plan/operation counts, and fallback reasons.
An operation count refers to an emitted IR instruction, not a CUDA kernel launch.
Counters advance only after begin/middle/finish and both upstream sublayers
return successfully. Warmup does not count as native TP execution. Unsupported
shapes/modes report a stock fallback once per reason per worker; counters retain
every fallback. Exceptions after execution starts propagate rather than retrying
a partly mutated residual through stock execution.

Bindings and scratch are local to each imported block and shape. Cache keys
include device, shape, dtype, operand identity/storage/layout/version and norm
configuration; load/unload clears them. Generator batching remains supported;
independent simultaneous calls into the same block's shared scratch retain the
upstream runtime's serialization requirement.

Block-local norm-state export requests use a visible stock fallback, preserving
the side effect in `RMSNorm.forward`. MTP's final norm/mixer exports remain
eligible. Native decode domains retain the existing small-row limits; batched
speculative verification can exceed them and fall back even when ordinary
batched decode uses native plans.

Example from the feature worktree, using the existing interpreter and binary:

```powershell
$env:CUDA_VISIBLE_DEVICES = '2,0'
$env:PYTHONPATH = 'D:\exllamav3-native-ir-tp;D:\exllamav3\native-results\v1.5.4-integration\extension'
& D:\tabbyAPI\venv\Scripts\python.exe util/validate_native_tp.py `
  --model <existing-model-directory> `
  --extension-dir D:\exllamav3\native-results\v1.5.4-integration\extension `
  --output native-results/tp/<new-run> --batches 1 2 3 4 --generator
```

Select `2,0,1` for three ranks. Select `2` with `--single` for the TP-disabled
baseline. The driver rejects single-rank TP. `--mtp` attaches the existing
single-device draft, using its upstream default draft size. `--reload` repeats
the correctness checks across unload/reload. `--context` chooses actual prompt
length within the fixed cache capacity. `--profile` instruments timing and must
not be used for final throughput claims.

The driver uses upstream null-backend warmup before real collectives. For cold
Triton compilation only, the harness temporarily extends the host RPC wait to
600 seconds. It dispatches these independent null-backend compilation passes
serially across ranks, avoiding a Windows Triton cache replacement race when
the two 4090s compile the same key. It then restores normal dispatch and drains
acknowledgments. All measured and correctness inference uses the original
multi-rank backend, collective timeouts, scheduling and reductions.

## Validation

The final combined native/dispatch/attention-export unittest suite passed all
176 tests, plus the standalone TP worker terminal-signal regression. This
includes the batched-Qwen-verification case. No dependency was installed.
The checks cover metadata-only export/import, capability exclusions, ordered
upstream sublayer calls, failed-stage accounting, cache invalidation, backend
identity, unsupported shapes/extension APIs, norm export side effects and
diagnostic snapshots/instruction totals.

CPU profiling of the initial Gemma integration identified repeated guard and
import work. Plan-selection imports are now resolved outside the hot path, TP-bound blocks skip
the old single-device owner guards, and invariant operand validation runs only
when a cached signature changes. Weight identity/storage/layout/version checks
remain active. The integration does not change collective count or wire traffic;
measured improvements come from local execution overhead, and vary by model.

Two-rank correctness requires exact prefill logits, decode logits, every traced
block residual on every rank, and live KV/recurrent state. Generator checks
compare completions, target-logit hashes and cache/draft statistics, including a
second request on the same generator. DeepSeek state probes exclude unused
capacity: notably its compressor saves only the final completed overlap window
of each forward call. The probe tracks those actually written indices.

The existing CPU reduction backend sums contributions in arrival order, rounding
to its wire dtype after each addition (see `all_reduce_cpu_avx2.cpp` and the
AVX512 implementation). With three ranks, stock TP itself is not bitwise
repeatable. Changing that backend is outside this integration. Three-rank
checks therefore use two complementary gates:

* Every executed native begin/middle/finish stage is checked **exactly** against
  stock norm/residual arithmetic using the same inputs and upstream reduced
  attention/MLP outputs. The oracle does not replay attention, mutate KV state
  twice or insert extra collectives. It also checks identical residuals across
  ranks. This instrumentation is disabled during timing.
* End-to-end logits use fixed reference tokens and a repeated stock run to
  measure reduction-order noise. Candidate aggregate RMSE must be at most
  twice stock-repeat RMSE plus 0.01, and maximum absolute error at most twice
  stock-repeat error plus 0.1. Reports retain the actual measurements and exact
  flags. These empirical envelopes are secondary checks, not universal model
  tolerances. Three-rank generator outputs may differ at near-tied tokens;
  local native-stage comparisons remain exact during those requests.

Timings use three alternating stock/native pairs of 128 decode steps after
four untimed steps, batch 1, with synchronized output sampling. Three-rank
timings replay the same token sequence to avoid comparing divergent workloads.
MTP is exercised separately through the generator; direct decode timings do
not use speculative decoding even when a draft is resident. Large prefill
retains the existing stock path, so no prefill IR speedup is claimed.

Per-run artifacts include configuration and model file metadata, source hashes,
extension SHA256, environment/GPU UUIDs, actual upstream shard allocation,
plan execution counters, parity reports, timings and saved logits. These are
under `native-results/tp` in the feature worktree. A `passed.json` marker is
written only after all requested checks finish and product source hashes remain
unchanged. Failed exploratory runs are retained as evidence.

Strict fully VRAM-resident single-GPU baselines are limited to Gemma and Devstral. GLM and DeepSeek
target weights alone exceed the Pro 6000's 95.6 GiB capacity. Qwen's complete
checkpoint includes a roughly 39 GB host-resident ngram table; that existing
architecture mechanism is retained in TP, but does not qualify as a fully
VRAM-resident single-GPU baseline. That strict baseline is explicitly skipped.
The user subsequently requested the practical Qwen comparison against their
existing Pro-6000-only setup. `--allow-host-ngram-baseline` measures that
unchanged single-GPU path, including its normal host table; it is labeled
separately from fully VRAM-resident baselines. No CPU
expert offload or altered allocation strategy is introduced to force a fit.

The requested GLM/DeepSeek single-card follow-up was checked directly against
safetensors headers: GLM base weights occupy 113.52 GiB (excluding vision and
MTP layer 45); DeepSeek base weights occupy 102.33 GiB (excluding MTP tensors).
Both exceed physical GPU 2's 95.59 GiB before caches and runtime allocations.
Their single-card entries are therefore **N/A: exceeds VRAM**, not unmeasured
throughput estimates. No oversized load was attempted. The capacity evidence
is saved in `native-results/tp/single-capacity-preflight.json`. Smaller
quantizations would be different checkpoints and require a separate comparison.

NCCL is not runtime-tested on this Windows environment; the measured backend
is the existing native CPU-assisted TP backend.

The user normally serves with MTP enabled. `--mtp --generator-bench --runs 0`
adds actual speculative-generation timing: one warm request per execution mode,
then three alternating stock/native pairs of 128 output tokens, greedy sampling,
the model's default fixed draft size, and ngram speculation disabled. Each
sample uses a fresh generator/page table and the same natural-language prompt;
its exact token count and accepted/rejected draft counts are saved. Stage
shadowing, logits hashing and block tracing are disabled during these timings.
Both generator decode time and wall time are reported. Single-GPU MTP baselines
measure only the existing IR-enabled path, with repeated native correctness
checks. The target and draft each retain a 65536-token cache.

## Completed validation (2026-10-24)

| Coverage | Result / local artifact directories under `native-results/tp/matrix` |
| --- | --- |
| Two-rank exact parity, B1/B2/B3/B4 Q1 and B1 Q5 | All five families passed: `qwen-two`, `glm-two`, `deepseek-two-r3`, `gemma-two`, `devstral-two`. `devstral-two-final` repeats B1–4 and timing on the final source. |
| Three-rank local-stage exactness, stock-repeat noise envelope, rank agreement, B1–4 | All five passed: `qwen-three-ar`, `glm-three-ar`, `deepseek-three-ar`, `gemma-three-r2`, `gemma-three-repeat`, `devstral-three`. |
| Concurrent generation and repeated prompt/cache reuse | All five passed on two ranks. Three-rank generation passed local-stage checks; outputs/logit hashes can vary with stock reduction order. |
| MTP and repeated requests, B1–4 | `qwen-three-mtp-r3`, `glm-three-r2`, `deepseek-three`; also early exact two-rank GLM MTP in `../glm-2-r3`. Qwen B2–4 speculative calls are explicitly **stock fallback only**; B1 executes native plans. |
| Actual timed MTP, 128 output tokens | All six two-/three-rank runs passed: `{qwen,glm,deepseek}-mtp-speed-{two,three}`. Two-rank samples have identical completions and acceptance counts across modes. |
| 8192-token prompts with exact live KV/recurrent state | All five passed in `{model}-long`. |
| Near-capacity prompt | Gemma passed exact prefill, decode, block outputs and live state at 65024 prompt tokens (`gemma-near-limit`). |
| Unload/reload and plan reconstruction | `qwen-reload` and `gemma-reload` passed both cycles. Gemma's parent allocations returned to exactly 46,861,824 bytes after GC on both unloads; loaded memory stayed flat. |
| TP-disabled baselines | Gemma and Devstral passed B1/B2 native generation; practical Qwen passed B1 ordinary and MTP on physical GPU 2. No one-rank TP was run. |

Initial validation finished with no whitespace errors, the original branch at
`311e156fae3bf2bd7beecb17fbe93e90274e74ef`, and all pre-existing checkout files
preserved. All owned test processes exited; GPU allocations returned to
desktop/idle levels. The later branch switch preserves those files in the main
checkout and retains the original branch pointer.

### Ordinary decode, no resident draft

Median output tokens/s, B1, 256-token prompt, 128 timed decode steps, cache
capacity 65536. Three alternating pairs per topology; Gemma three-rank includes
all six pairs from the original and repeat campaigns. Single-GPU columns use
physical GPU 2 with TP disabled. Qwen includes its existing host ngram table.

| Model | Single GPU IR | 2 GPUs stock | 2 GPUs IR | IR change | 3 GPUs stock | 3 GPUs IR | IR change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 71.79 | 31.56 | 31.42 | -0.5% | 31.28 | 31.51 | +0.8% |
| GLM-5.3 | -- | 42.64 | 51.64 | +21.1% | 26.18 | 35.57 | +35.8% |
| DeepSeek V4 | -- | 21.16 | 24.14 | +14.1% | 20.91 | 23.96 | +14.6% |
| Gemma 4 | 41.52 | 45.18 | 46.08 | +2.0% | 40.69 | 43.30 | +6.4% |
| Devstral 2 | 15.48 | 19.17 | 19.27 | +0.5% | 24.55 | 24.63 | +0.3% |

These gains compare IR with stock **within the same topology**. Qwen TP is
roughly 56% slower than its practical single-GPU IR baseline. Devstral benefits
from the third GPU (about 28% over two-rank IR); its IR-versus-stock change is
near-neutral. Gemma's three-rank native samples span 19.98–44.67 tokens/s,
including one low outlier. Its stock range is 39.65–41.91; no sample is discarded.
Gemma's initial 8192-context native sample was 36.16 versus stock 43.26 tokens/s;
three repeat pairs were native 44.39–44.72 versus stock 43.83–44.14. This does
not establish a repeatable long-context slowdown.

Median two-rank decode latency falls from 23.45 to 19.36 ms/token for GLM and
47.25 to 41.42 for DeepSeek. Qwen stays around 31.7–31.8 ms/token, compared with
13.93 ms/token on the Pro 6000 alone. Prefill uses stock execution: measured
two-rank stock/IR prompt rates were Qwen 330/328, GLM 376/379, DeepSeek 364/367,
Gemma 1172/1159, and Devstral 367/365 tokens/s. These are workload-specific
observations, not evidence of a prefill IR optimization.

Earlier three-rank direct timings with a resident MTP draft are retained but
excluded from this table. A resident draft changes free VRAM and hence the
upstream allocation/shard geometry. It is not a controlled target-only scaling
comparison. Allocation files and GPU UUIDs accompany every final run.

### MTP generation, matching the user's serving mode

Median decode tokens/s using the generator's `new_tokens / time_generate`.
Parentheses contain full-request wall throughput, including prefill and request
processing. Each row uses three samples per mode, B1, 128 output tokens, a fresh
generator, and 65536-token target/draft caches. Prompts contain 25 tokens for
Qwen/DeepSeek and 23 for GLM; draft sizes are respectively 4, 5 and 3.

| Model | Single GPU IR | 2 GPUs stock | 2 GPUs IR | 3 GPUs stock | 3 GPUs IR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 103.62 (93.23) | 52.30 (46.01) | 53.34 (46.85) | 52.31 (39.37) | 56.22 (41.27) |
| GLM-5.3 | -- | 75.98 (65.74) | 77.67 (66.22) | 54.28 (45.70) | 62.37 (53.82) |
| DeepSeek V4 | -- | 31.14 (28.47) | 33.27 (30.84) | 25.14 (23.18) | 29.39 (28.35) |

Two-rank draft acceptance is identical across stock/IR samples: Qwen 57.05%,
GLM 57.45%, DeepSeek 69.17%. Three-rank native acceptance varies: Qwen
55.00–61.49%, GLM 55.56–61.48%, DeepSeek 53.38–62.99%. The existing reduction
order changes generated paths and speculative acceptance, so a throughput
difference cannot be attributed entirely to faster kernels or more GPUs.
Native timing ranges are Qwen single 100.52–104.15, two 52.87–53.84, three
53.33–57.32; GLM two 76.80–77.94, three 61.45–64.89; DeepSeek two 30.65–35.61,
three 25.68–30.81 tokens/s. DeepSeek has noticeable timing variation.

Peak Torch allocations during these MTP campaigns (including warmup/validation
and the resident draft on the Pro 6000) were:

| Model | Two GPUs, GiB (physical 2, 0) | Three GPUs, GiB (physical 2, 0, 1) |
| --- | --- | --- |
| Qwen3.8 | 46.95, 21.89 | 36.79, 16.77, 16.76 |
| GLM-5.3 | 81.82, 38.74 | 63.16, 29.09, 29.09 |
| DeepSeek V4 | 76.22, 36.25 | 59.51, 27.44, 27.35 |

These are per-process allocated-memory peaks, not driver-reserved VRAM or
isolated decode-only peaks. Allocation/shard diagnostics identify each device
by UUID and verify tensor sharding across actual participating ranks.

For this measured workload, keep Qwen on the Pro 6000 alone and prefer two-rank
TP for GLM and DeepSeek. Three-rank Devstral is useful for ordinary decode.
Gemma favors two ranks over three here. These are per-model choices, not a
recommendation to enable three-rank TP globally. Existing TabbyAPI settings
have not been changed. Long-context MTP throughput and other concurrency/draft
settings were not benchmarked; the long-context tests above use ordinary decode.

### Single-GPU numerical comparisons and Qwen limitation

`native-results/tp/cross-topology.json` compares prefill and first-decode logits
at identical inputs, with matching source, extension, checkpoint and config
metadata. Gemma's maximum absolute difference was 0.2461, maximum RMSE 0.0403,
minimum cosine similarity 0.999977; Devstral's were 0.0928, 0.0182 and 0.999981.
Argmax matched at all eight checked positions/batch cases for each model.

Qwen differs substantially more between single-GPU and existing TP execution:
maximum absolute error 5.7461, maximum RMSE 0.7239, minimum cosine 0.8742.
Argmax matched in all four checked cases, but top-10 overlap was only 8–9/10 and
KL(single || TP) reached 0.1191 nats. See `qwen-cross-topology-detail.json`.
This difference is already present in large prefill, where both paths use stock
operations, and two-rank stock TP and IR TP remain bitwise identical. Thus this
integration preserves upstream TP behavior; it does **not** establish numerical
or quality equivalence of Qwen TP with the single-GPU path. An upstream TP
numerics investigation is separate work. Along with the large speed loss, this
supports retaining Qwen's existing single-GPU configuration.

The earlier saved Qwen campaign reported 70.95 tokens/s for ordinary native
decode on physical GPU 2; a later 1000-token campaign reported 46.76. The fresh
71.79-token/s baseline here confirms that the existing fast single-GPU path is
still active. Historical workloads are not substituted for the matched table.

## Compiled release

The user-requested wheel is
`exllamav3-1.5.4+nativeplan.v2.tp1-cp313-cp313-win_amd64.whl`, built with
Python 3.13.12, Torch 2.9.0+cu128, CUDA Toolkit 12.8 and Visual Studio 2022.
It contains freshly compiled Ada (`sm_89`) and Blackwell (`sm_120`) images,
verified with `cuobjdump`, plus the TP-aware Python implementation. Copying only
the extension is insufficient. Wheel SHA256:
`06c8a217da8d4f882e1985762fadd5b6abd72646f67543f04837d153e1dc3e91`.

`util/build_native_tp.bat` builds an isolated wheel without installing it. It
defaults to 32 compiler jobs on this 56-core machine and respects a supplied
`MAX_JOBS`. Compilation and wheel packaging use separate invocations to avoid
PyTorch BuildExtension mutating flags and repeating the compilation.

The extracted wheel passed all 176 regression tests without skips and the
standalone TP signal regression. GLM native-stage parity passed on both GPU
architectures; Qwen GR parity passed on both; actual-weight DeepSeek trunk and
draft stages passed across the two GPUs. Real-model Gemma two-rank parity and
GLM two-rank MTP generation also passed, with native execution verified and
65536-token model/draft caches. Package source hashes match the feature branch;
the copied extension matches the newly linked binary. Evidence is under
`native-results/tp/package` in the sibling worktree.

After these gates, the user authorized installing this wheel into TabbyAPI.
Its previous `1.5.1+nativeqwen.concurrent1` wheel and exact installed package
files were backed up in `native-results/tp/package/tabby-rollback`. Installation
used `--no-index --no-deps --force-reinstall`; dependency versions and the normal
TabbyAPI `config.yml` were verified unchanged. Other backends were not installed
or reconfigured.

The installed TabbyAPI backend then passed a two-GPU (`CUDA_VISIBLE_DEVICES=2,0`)
GLM test with MTP enabled and 65536-token target/draft caches. Text completions,
templated chat completions and streaming chat each generated 64 tokens. Runtime
diagnostics came from the installed package and recorded 2,925 completed native
block executions on each of the two ranks. Draft acceptance was present in the
server logs. This is a backend smoke test, not a replacement for the matched
benchmark tables above. The temporary loopback-only server unloaded and exited;
the normal configuration stayed unchanged and the new wheel remains installed.

The release directory is `D:\exllamav3\dist\native-ir-tp-1.5.4-tp1`, containing
the wheel, installation notes, report, source ZIP, validation evidence, hashes
and a manifest identifying the feature commit. The branch is checked out in
`D:\exllamav3`; the sibling validation worktree is retained detached at the same
commit so its artifacts and rollback backup remain accessible.
