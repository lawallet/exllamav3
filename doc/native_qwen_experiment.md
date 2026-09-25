# Qwen3.8 native orchestration experiment

The sections below retain the original non-MTP and MTP experiments. The final
"Dynamic batching / concurrency extension" section describes the current build.

## Recorded environment (2026-09-24)

- Checkout: `6b84a21b6f1e5da3f291b9e1019061f0de788279`, v1.5.1 plus the MSVC `__builtin_popcount` fix only.
- Python: 3.13.12, from `D:\tabbyAPI\venv\Scripts\python.exe`.
- Torch: 2.9.0+cu128; Torch CUDA: 12.8.
- ExLlamaV3: 1.5.1+cu128.torch2.9.0.
- triton-windows: 3.8.0.post28; flash-linear-attention: 0.5.2.
- Dependency snapshot: `D:\tabbyAPI\before-exllama-native-test.txt`.
- Original installation: official release wheel, identified by installed `direct_url.json`.
- Original wheel SHA256: `ac338afd706e071109c24d845f60c438fc36235420927ba0dc6e0985d2ca8ada`.
- Physical GPU 2: RTX PRO 6000 Blackwell Max-Q Workstation Edition, WDDM, 300 W.
- CPU: AMD EPYC 7663, 56 cores / 112 logical processors. WMI/psutil report a
  nominal 2 GHz; this does not measure the running core's boost frequency.
- Model: `D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw`.

The original wheel can be restored with `powershell -File util\rollback_exllamav3.ps1`.
It verifies the original archive hash before reinstalling with `--no-deps`.
No dependency changes are needed for this experiment.

## Source findings

The model architecture is `Qwen4ExpForConditionalGeneration`, built in
`exllamav3/architecture/qwen4_exp.py`. Its gated residual mixers include normalization
and return FP16; the blocks have no separate attention/MLP input RMSNorm modules.
Each ordinary block dispatches six top-level Python methods: attention mix,
attention/GDN forward, attention apply, MLP mix, MoE forward, MLP apply.
For 48 layers this is 288 calls, plus 48 block calls, nested methods, embedding,
PLE, final mixer/head, and generator/model dispatch. This is a source count,
not a measured timing result.

GDN already invokes `BC_GatedDeltaNetSplit.run_bszN` and attention uses a
cache-specific `BC_Attention` object. MoE separately dispatches routing and
`BC_BlockSparseMLP.run_bszN`. Gated residual workspaces are shared per device:
the attention site's injection gate must be consumed before the MLP mix
overwrites it. A native orchestration boundary must preserve this ordering.

## Profiling and reproduction

`util/profile_qwen_decode.py` uses the real paged Generator, a single sequence,
greedy sampling, seed 1234, no draft model, no n-gram speculation, a 173-token
prompt, 64 warmup tokens, and three 1000-token measured generations. The prompt
is in `doc/native_qwen_prompt.txt`. It imports checkout Python with the installed
extension and records both locations. It does not install packages.

Example (PowerShell, Tabby's GPU mapping):

```powershell
$env:CUDA_VISIBLE_DEVICES = '2,0,1'
& D:\tabbyAPI\venv\Scripts\python.exe util\profile_qwen_decode.py `
  --model D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw `
  --prompt-file doc\native_qwen_prompt.txt --device cuda:0 `
  --output native-results\baseline-mapped --trace
```

Throughput uses `(new_tokens - 1) / time_generate` because the generator's
generation interval starts at its first token. Prompt cache hits are recorded;
do not compare first-token latencies across unequal cache-hit configurations.
Completion hashes detect text differences but are not a replacement for logits,
token-ID, or cache/state parity tests.

Instrumentation is installed only on the loaded instances and is removed even
after exceptions. It counts only `(batch, q_len) == (1, 1)` model forwards.
Host inclusive times overlap; exclusive times subtract instrumented children.
Neither metric measures GPU idle time. CUDA trace analysis is necessary to
quantify device gaps, and profiled timings must not be used as throughput results.
The uninstrumented measured generations run before the separate profiling pass.

Unit checks: `D:\tabbyAPI\venv\Scripts\python.exe tests\test_decode_dispatch_profile.py`.

## Prototype boundary

`EXL3_NATIVE_QWEN_BLOCK=1` selects an experimental two-call block boundary:
`BC_QwenBlock.begin` mixes the attention input, the existing attention/GDN
Python method runs normally, and `BC_QwenBlock.finish` applies its residual,
mixes the MoE input, routes, invokes `BC_BlockSparseMLP.run_bszN`, and applies
the MoE residual. The extension calls the original `gr_mix`, `hc_apply`, and
`routing_std` entry points; mathematical operations and their order are unchanged.
Dedicated per-block mix buffers avoid repeated Python workspace lookup/view
construction and preserve injection gates until consumed. Buffers and the native
object are released on unload and rebuilt on load.

The entry point rejects non-trunk/MTP blocks, TP, multiple active devices,
prefill, batches or query lengths other than one, history recording, capture,
state export, CPU expert offload, nonstandard routing, extra norms/scalars,
and unsupported MoE layouts. An older extension without the new class falls
back as well. The original block implementation remains intact.

This is a partial block orchestration prototype: attention cache setup and GDN
state lookup stay in Python. It does not add CUDA graphs or alter prefill kernels.

Build with `cmd /c util\build_native_qwen.bat`. The initial experiment targeted
SM120; the final wheel includes SM89 and SM120 for this machine's 4090 D and
RTX Pro 6000 GPUs. The script stages a wheel without changing installed packages.
To test the staged extension, append these options to the benchmark command:

```text
--extension-dir build\lib.win-amd64-cpython-313 --ab
```

The A/B mode first compares 128 deterministic steps byte-for-byte: logits,
token IDs, all 48 block residual outputs, and GDN/PLE plus initialized KV/raw
QSA/complete pooled QSA state at steps 0, 1, 7, 31, and 127. It refuses to benchmark
on a parity failure or if any block failed to construct the native path.
It then warms both paths and alternates A/B, B/A, A/B, with 1000 tokens each.
Host timing without Torch profiler runs separately for each path afterwards.

## Validation and installation

The SM120 staged build passed all 128-step exact checks, with all 48 blocks using
the native path. Stock and native 1000-token completions also matched in all six
measured runs. Five guard tests cover CPU embedding placement, TP/multiple GPUs,
unsupported parameters, batch/query shapes, CPU offload, and old extensions;
two instrumentation tests cover counting and restoration after exceptions.

Two early validation attempts refused to benchmark because the eligibility
guard rejected every block. Explicit single-device load does not fill
`active_devices`, and the stock embedding intentionally remains on CPU. The
guard now checks actual placement while allowing stock `prefer_cpu` modules.
These were fallback/coverage failures, not numerical parity failures.

Final wheel: `native-results/wheels/exllamav3-1.5.1-cp313-cp313-win_amd64.whl`.
SHA256: `2b05482da16727a1bac036550be6727bd48be111839f87f144878d19cbf8e9aa`.
Installed with the Tabby venv's pip using `--no-deps --force-reinstall`.
Imports from outside the checkout resolve to the venv's site-packages for both
Python and the compiled extension. The before/after pip snapshots differ only
in ExLlama's wheel origin/version suffix; Torch, CUDA, Triton, FLA and all other
dependencies are unchanged. The installed package also passed all exact checks.

The original official wheel is saved in `native-results/rollback/`, verified
against its original SHA256. The rollback script prefers this offline copy.
No Text Generation WebUI files were changed. The native feature remains off by
default. To enable it in a new Tabby process (cmd.exe):

```bat
set EXL3_NATIVE_QWEN_BLOCK=1
cd /d D:\tabbyAPI
call start.bat
```

To disable it, clear the variable before starting a new process:
`set EXL3_NATIVE_QWEN_BLOCK=`. To restore the original package entirely, run
`powershell -File D:\exllamav3\util\rollback_exllamav3.ps1` with Tabby stopped.

## Staged-build results

All runs: 173 prompt tokens, 1000 generated tokens, MTP off, batch one, FP16
cache, same greedy sampler/seed, same 300 W limit. A/B ordering: A/B, B/A, A/B.

| Metric | Stock | Native |
| --- | ---: | ---: |
| Decode t/s, run 1 | 54.52 | 74.41 |
| Decode t/s, run 2 | 55.32 | 74.31 |
| Decode t/s, run 3 | 55.34 | 74.55 |
| Mean decode t/s | 55.06 | 74.42 |
| Mean total generation seconds | 18.350 | 13.650 |
| Mean prefill/first-token interval seconds | 0.203 | 0.224 |
| Mean GPU utilization, 1 Hz samples | 60.5% | 77.7% |
| Mean GPU draw, 1 Hz samples | 165.3 W | 194.1 W |

Mean throughput increased **35.2%**. External GPU samples include short prefill
intervals and are approximate. Per-core CPU utilization was not captured for
this series. First-token time was not the optimization target.

Lightweight host instrumentation measured 344 Python module/model calls per
decode on stock and 104 on native (240 fewer, or five fewer per block; native
begin/finish extension calls are additional). Stock gated residual methods
accounted for 9.31 ms/token of instrumented host wall time and MoE for 4.61 ms;
the native block body accounted for 8.18 ms after combining them. These are
instrumented dispatch costs, not isolated CPU execution or device idle time.
The improvement also includes fewer tensor-view/workspace operations and fewer
pybind transitions, so it cannot be attributed solely to Python function calls.

Raw results: `native-results/ab/{benchmark,parity,metadata,dispatch-0,dispatch-1}.json`.
GPU telemetry: `native-gpu-telemetry.csv`. Summarize with
`python util/summarize_qwen_ab.py native-results/ab --telemetry native-gpu-telemetry.csv`
using the Tabby venv executable.

The prototype supports the host-orchestration hypothesis for this workload.
Long-context sparse-QSA selection, MTP, concurrent batches, TP, other models,
and CPU offload have not been validated for native execution. The narrow
prototype intentionally leaves attention/GDN preparation in Python; no second-stage
runtime redesign was attempted.

## Final installed-build verification

The final SM89+SM120 wheel was tested using `--installed --ab`, which removes
the checkout package from the import path. Both package and extension paths in
`native-results/installed-ab/metadata.json` point into Tabby's site-packages.
Exact parity passed again for every checked tensor/state and all 48 native blocks.

| Metric | Stock | Native |
| --- | ---: | ---: |
| Decode t/s, run 1 | 42.18 | 60.59 |
| Decode t/s, run 2 | 42.23 | 61.01 |
| Decode t/s, run 3 | 41.03 | 60.37 |
| Mean decode t/s | 41.81 | 60.66 |
| Mean total generation seconds | 24.113 | 16.694 |
| Mean prefill/first-token interval seconds | 0.214 | 0.221 |
| Mean GPU utilization, 1 Hz samples | 46.1% | 63.6% |
| Mean GPU draw, 1 Hz samples | 141.7 W | 169.7 W |

This series improved **45.1%**, with identical completion hashes across all six
runs (and matching the staged/official-extension runs). Absolute performance
changed substantially between sessions. The cause has not been isolated; the
final session also uses the rebuilt dual-architecture wheel. Do not interpret
74 t/s as guaranteed installed throughput. The controlled comparison is within
each alternating series, where both paths share the same extension, model,
cache, sampling, GPU mapping, and power limit. The cache allocation was 1280
FP16 tokens for both paths, versus Tabby's configured 65536; this experiment
targets short context, not a full Tabby service workload.

The experiment provides strong evidence that this native orchestration boundary
reduces host dependence while preserving behavior in the tested scope. It does
not isolate a pure Python-only cost or precisely quantify uninstrumented GPU
idle time. CPU utilization/boost was not sampled continuously; a final-session
spot sample reported 0.6% system CPU utilization, which is not a per-token metric.

The feature is installed but remains opt-in. Benchmark processes and telemetry
have been stopped, releasing the model's VRAM. No dependency or system settings
were changed.

Installed-extension smoke checks also passed on both the 4090 D and RTX Pro
6000 using Qwen's 2560-wide residual operation. An initial 8-wide synthetic
smoke input hit an existing `hc_apply` host integer-division-by-zero edge case
(`D / chunks_c` becomes zero); the actual Qwen width is unaffected. That unrelated
kernel edge case was not changed as part of this prototype.

## Initial baseline

Physical GPU 2 addressed as `cuda:2` with the default current device measured
32.40, 32.29, 32.35 t/s. With Tabby's `CUDA_VISIBLE_DEVICES=2,0,1` mapping and
`cuda:0`, stock measured 51.74, 52.29, 53.81 t/s. Completion hashes matched across
all six runs. Device mapping is therefore held constant for the experiment;
the slow unmapped run is not used as the optimization baseline. The precise
driver/context cause of that difference has not been isolated.

The initial Torch traces strongly perturb execution (roughly 47–65 ms/forward),
so they establish dispatch counts and ordering, not uninstrumented host idle
time. The trace confirmed 96 mix, 96 apply, 48 MoE, 36 GDN, and 12 attention
calls per token. The later lightweight host timings omit record_function and
Torch profiler and are recorded separately from measured generation throughput.

## Native MTP extension (2026-09-24)

This extension continues from the protected non-MTP checkpoint `90ec653`
on branch `native-qwen-mtp`. The checkpoint tag, backup branch, Git bundle,
and tested non-MTP wheel under `dist/checkpoints/native-qwen-90ec653` remain
unchanged. Core implementation: `38c281f`; parity/benchmark tools: `9a68dc4`.

### Execution trace and original rejection guards

`Qwen4ExpMTPModel.attach_to` borrows the trunk embedding/head and requests the
trunk final mixer's pre-collapse hidden stack through `export_state_norm_keys`.
That export occurs in the final `GatedResidual.forward`, outside transformer
blocks. The MTP component has one full-attention/QSA and MoE block; it has no
GDN or PLE recurrent layers of its own.

`Generator.iterate_draftmodel_mtp_gen` performs four sequential one-token draft
forwards. `iterate_gen` then verifies five target positions with
`recurrent_history=True`. The previous native implementation rejected this
through four independent conditions: no native owner on the non-trunk MTP
block, query length other than one, recurrent history, and any norm-state
export. Its C++ residual and router scratch also had only one row, so removing
the Python guards alone would have been incorrect.

The original attention/GDN forwards remain in control of cache writes and
history recording. Model recurrent-state advancement records the query length
and four history steps. After rejection, `reject_remainder` calls
`GDNState.rewind`, restoring GDN, convolution and PLE state, then rolls back
page positions. Fully accepted windows use `rewind(0)` to normalize carried
history. Rejected KV/QSA suffix bytes need not be erased: page positions define
the committed prefix and subsequent writes replace the suffix.

After sampling, accepted draft-cache positions are repaired by the existing
`draft_model.prefill` using the corresponding exported target states. The next
draft's `mtp_last_hidden` is cloned from the last accepted target hidden state.
These prefill/repair calls deliberately retain the stock path. End-of-generation
unused proposals are counted as rejected even when the job simply releases its
state instead of performing another rewind.

### Implementation and supported scope

`BC_QwenBlock` now owns scratch for its fixed row count, including multi-row
router outputs, and uses the existing residual, routing and `run_bszN` MoE
kernels. One-row contexts retain the previous constructor behavior and
workspace layout. Five-row contexts are separately cached on each block and
released on load/unload. No GDN, attention, routing, residual or MoE math kernel
was changed; there is no new graph capture, scheduler or rollback code.

The Python guard recognizes an attached, single-layer Qwen4Exp MTP head and
target on the same GPU, with batch one. Target verification permits query
length five with history capacity at least four, one cache-bound recurrent
state, and exactly the final mixer export. A one-token bootstrap is also
eligible. The draft path permits one token with the expected target hidden
shape and no recurrent history/export. Other verification lengths (including
two through four), TP, cross-GPU components, offloaded MoE, prefill, block-local
state exports and the previous unsupported options fall back. An older native
extension without `native_qwen_mtp_max_q_len=5` also falls back for MTP while
retaining the original non-MTP native path.

Validated configuration: the same Qwen3.8 model, physical GPU 2 mapped to
logical CUDA 0 by `CUDA_VISIBLE_DEVICES=2,0,1`, 300 W limit, FP16 caches with
1280 tokens, one active sequence, greedy sampler/seed 1234, four fixed draft
tokens, dynamic drafting off and n-gram drafting off. The existing
`EXL3_NATIVE_QWEN_BLOCK=1` flag enables both native paths; it does not itself
enable MTP in the application.

### Exact parity coverage

`util/check_qwen_mtp.py` drives the real generator for stock and native runs.
It compares byte hashes of every target and draft block residual, target and
draft logits, exported hidden states, actual proposal IDs and generated IDs.
It additionally compares full active-slot GDN/convolution/PLE state and history,
target and draft KV/QSA tensors (including raw and pooled QSA planes), page
positions, acceptance counters, and the carried target hidden state. Snapshots
are taken before/after recurrent rewind, after accepted-token cache repair, and
before final state release. Cache pages are read in logical order and backing
storage is zeroed before each correctness run, making unwritten tails defined.

Natural rejection and a separate test-only proposal mutation both run through
the actual rejection path. Mutation rotates through draft positions; original
draft logits/proposals and the injected IDs are recorded separately. Tests
require both accepted tokens and nonzero rewind depth, and independently count
proposals to check that proposed = accepted + rejected. Tensor hashing and state
snapshots are absent from the performance runs.

The original 128-step non-MTP parity test and all 17 guard/profiling unit tests
also pass. The tested context is short; long-context sparse-QSA selection,
concurrent jobs, other models and other MTP configurations are not validated.

Staged natural parity compared 2,632 events over 128 output tokens and 37
verification rounds: 148 proposed, 91 accepted and 57 rejected drafts. Rewind
depths 0/1/2/3/4 occurred 14/8/7/1/7 times. Forced parity compared 2,170 events
over 64 output tokens and 30 rounds: 120 proposed, 34 accepted and 86 rejected;
rewind depths 0/1/2/3/4 occurred 1/4/6/9/10 times. Both comparisons were exact.
These cases exercise every rejection depth and accepted-token commit behavior,
not merely 100% acceptance.

### Staged MTP performance

The initial staged build produced stock decode rates of 97.31, 84.83 and 96.93
t/s versus native rates of 103.28, 103.56 and 103.45 t/s: means 93.03 and 103.43
t/s (+11.2%). The slower second stock run is retained. All six completion
hashes and acceptance counts matched. This series had no GPU telemetry; the
installed-build series below is the final measured result. These legacy decode
rates use `(output_tokens - 1) / time_generate`; effective output throughput
uses `output_tokens / time_generate` (0.1% higher for 1000-token generations).

### Final installed MTP results

The installed build passed both parity suites again with the same event counts,
token counts and rewind histograms above. Natural runs used 1,924 native block
calls and 78 fallbacks; forced runs used 1,560 native calls and 69 fallbacks.
The non-MTP 128-step exact regression also passed with all 48 native blocks.

| Metric | Stock MTP | Native MTP |
| --- | ---: | ---: |
| Effective output t/s, run 1 | 87.37 | 102.51 |
| Effective output t/s, run 2 | 96.84 | 102.51 |
| Effective output t/s, run 3 | 95.95 | 102.76 |
| Mean effective output t/s | 93.39 | 102.59 |
| Mean total generation seconds | 10.959 | 9.974 |
| Mean prefill/first-token interval seconds | 0.2234 | 0.2226 |
| Acceptance rate, every run | 49.85% | 49.85% |
| Proposed / accepted / rejected drafts, every run | 1336 / 666 / 670 | 1336 / 666 / 670 |
| Native / fallback block calls, every run | 0 / 17681 | 17368 / 313 |
| Mean GPU utilization, 1 Hz samples | 65.0% | 71.4% |
| Mean GPU power draw | 187.2 W | 198.7 W |
| GPU telemetry samples in measured intervals | 33 | 29 |

Mean effective output throughput improved **9.9%**. Every native run was faster
than every stock run, but the slow first stock run increases the mean relative
gain; absolute throughput varies between sessions. The power limit stayed at
300 W throughout. The 173-token prompt, 1000-token output length, model, device
mapping, sampler and cache/MTP configuration were identical for all six runs.
The benchmark counts actual draft proposals independently of acceptance stats.

Each native generation used 16,032 native target block calls (334 five-token
verification rounds across 48 layers) and 1,336 native draft calls. The 313
fallback calls consist of 48 target prefill blocks, one draft prefill block,
and 264 accepted-token draft-cache repair calls. No target verification or
ordinary draft step silently fell back in this measured configuration.

All six completion SHA-256 hashes were
`7264d3e6445e1597bf1eca6cd0218f4ad01ace3bfeb76a0cf8c81f8c8037bad7`.
Full numeric results, including end-to-end throughput, are in
`native-results/mtp-installed/benchmark.json`; joined GPU metrics are in
`summary.json`, and telemetry is `native-results/mtp-installed-gpu.csv`.
The validation/benchmark processes and telemetry collector have stopped;
physical GPU 2 returned to 19 MiB allocated after completion.

### Build and reproduction

The installed wheel is `exllamav3-1.5.1+nativeqwen.mtp1-cp313-cp313-win_amd64.whl`
(132,508,408 bytes), SHA-256
`4e969785cc3c0a32704a16199b7b6c602f6b25ec6bad4a267ffa6375954ef592`.
It contains SM89 and SM120 code and was built with the existing
`util/build_native_qwen.bat` workflow (four compiler jobs). All 241 packaged
Python files match the checkout and the installed extension matches the wheel
byte for byte. Both imports resolve inside Tabby's site-packages. Python,
Torch 2.9.0+cu128, CUDA 12.8, Triton and FLA were unchanged; the before/after
pip freeze differs only in ExLlama's local wheel reference.

Reproduce in PowerShell using the existing Tabby environment:

```powershell
$env:CUDA_VISIBLE_DEVICES='2,0,1'
& D:\tabbyAPI\venv\Scripts\python.exe -u util\profile_qwen_mtp.py --model D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw --installed --output native-results\mtp-installed
& D:\tabbyAPI\venv\Scripts\python.exe util\summarize_qwen_ab.py native-results\mtp-installed --telemetry native-results\mtp-installed-gpu.csv
```

The runner performs non-MTP, natural-MTP and forced-rejection parity before
warming both paths and alternating stock/native, native/stock, stock/native
1000-token generations. Lightweight call/proposal counters run in both timed
paths. External `nvidia-smi` telemetry uses one-second samples; those samples
include the short prefill interval. Raw data lives under
`native-results/mtp-{staged,installed}`. The recorded `revision` is the base
checkpoint because the tests started before committing the working tree;
the final wheel hash above and implementation commits identify the actual build.

To restore the protected non-MTP build without changing dependencies:

```powershell
& D:\tabbyAPI\venv\Scripts\python.exe -m pip install --no-deps --force-reinstall D:\exllamav3\dist\checkpoints\native-qwen-90ec653\exllamav3-1.5.1-cp313-cp313-win_amd64.whl
```

`util/rollback_exllamav3.ps1` remains available for restoring the original
official wheel instead.

## Dynamic batching / concurrency extension (2026-09-24)

Development continues on `native-qwen-concurrency` from MTP checkpoint
`34e7a06` / tag `validated-native-qwen-mtp1-2026-09-24`. That installed wheel
and the earlier non-MTP checkpoint remain preserved in their separate release
directories. The concurrency prompt is `concurrency_prompt.md`.

### Actual generator execution and isolation

The source trace and live pre-change trace (`util/trace_qwen_concurrency.py`,
`native-results/concurrency-trace.json`) show that ready jobs are compacted
along the **batch** dimension. Transformer residual stacks are `[B,Q,4,2560]`:
ordinary decode has Q=1; fixed-four MTP verification has Q=5; each of the four
draft forwards has Q=1. Prefill is performed job by job before the decode
iteration. Arrivals/completions change B; they do not concatenate independent
jobs along the sequence dimension. All target rows in one verification call
share Q. Acceptance length and subsequent cache-repair query lengths can differ
between jobs; those repairs are separate per-job calls.

`Generator.iterate_gen` builds a separate block-table row and cache length for
each compacted sequence, and a corresponding list of recurrent state objects.
GDN tensors are indexed by each state's allocated slot, not its transient batch
row. PLE convolution and CPU token-ID history use the same slot ownership.
Each job's page table maps its logical positions to physical target KV/QSA
pages; the draft uses the same page layout in separate backing tensors.
The live trace saw slots 0/1/2/3 and distinct initial pages, later reusing freed
recurrent slots and assigning a separate page after a staggered arrival.
QSA raw keys, pooled summaries and KV
follow these page mappings.

MTP drafts all ready jobs together and verifies them together. Each job samples
and accepts/rejects independently, rewinds its own GDN/PLE slot and page
positions, repairs its own draft-cache prefix and retains its own
`mtp_last_hidden`. Live rows in a single call had different accept/reject
counts and therefore different positions in subsequent calls. Scheduling,
sampling, recurrent advancement, rewind and cache repair remain unchanged.

### Native boundary and scratch ownership

The old Python guard rejected B != 1 before constructing a context. The MTP
guard also required one recurrent state and a batch-one target-hidden shape.
C++ `begin` required `[1,rows,H,D]` and returned `[1,rows,D]`. Widening the Python
guard alone would still fail that check; flattening jobs into its old sequence
dimension would corrupt attention/recurrence state selection.

The extension permits ordinary B=2/3/4, Q=1 only with a matching paged table,
cache-length vector and distinct cache-bound recurrent slots. It gates the new
ABI with `native_qwen_max_batch=4`, so older extensions safely fall back. C++
keeps `[B,Q,D]` on the attention boundary and flattens B*Q only for row-local
residual/routing/MoE operations. Contexts use the existing per-row-count cache:
each block owns separate residual gates, mixed inputs and router outputs for
each supported row count. The row-one and row-five contexts remain unchanged.

This is batching inside one serialized generator, not concurrent model.forward
calls on different Python threads/CUDA streams. Native scratch is consumed
before the next call on the same stream. Stock MoE output and other existing
device workspaces also depend on that ordering. Each row's residual is updated
independently; mutable recurrent/cache state remains in its original slot/page
storage. No scheduler or math kernels were changed.

**Multi-job MTP target verification deliberately falls back.** Two/three jobs
produce 10/15 flattened rows, beyond both `GatedResidual.FUSED_MAX_R=8` and
`BC_BlockSparseMLP.MAX_BSZN=8`. Stock changes to tiled residual mixing and another
MoE execution path there. This extension does not split those rows or substitute
the small-row kernels. MTP draft steps with B=2/3, Q=1 are eligible for native
execution; once only one target job remains, its existing Q=5 native path resumes.
B=4 MTP draft steps, larger ordinary batches, other query lengths, TP, offload,
prefill and existing unsupported options retain the stock path.

### Concurrency validation procedure

`util/check_qwen_concurrency.py` uses independent coding, cooking, science,
fiction and database prompts, seeds 1234 onward, and different output lengths.
It compares stock/native logits and residuals by row, final IDs/completion
hashes, per-job RNG state, recurrent/PLE state and history, KV/QSA contents,
logical page positions and committed MTP carries. Physical page/slot identities
may differ between runs, so it asserts disjoint ownership and correct block
tables before comparing logically ordered state. Independent prompts in this
test do not share prefix pages.

State snapshots cover batch transitions, fixed checkpoints and final release;
MTP additionally snapshots every rewind and commit. Forward and prefill calls for a subset
of jobs, including accepted draft-cache repair, and every MTP rewind assert that other jobs' complete state/cache bytes
remain unchanged. A forced MTP case mutates proposals only for alternating job
IDs, causing different speculative histories in simultaneous jobs. Actual
proposal counting slices only active rows: the generator returns a full-capacity
draft backing view with stale unused rows after the batch shrinks.

Ordinary cases include B=2/3/4, a 4->3->2->3 staggered arrival, seeded stochastic
sampling, and jobs crossing cache-page boundaries. MTP cases include two/three
jobs, staggered arrival with forced rejection, and page-boundary crossing.
The original single-job non-MTP, natural MTP and
forced-rejection regression suites remain separate checks.

### Serving configuration and measurement scope

The existing TabbyAPI config already sets `model.max_batch_size: 4` and currently
has `draft_model.draft_mode: disabled`. No service configuration or scheduling
policy was changed. With the new wheel and `EXL3_NATIVE_QWEN_BLOCK=1` set before
startup, eligible ordinary decode batches use the extended path automatically.
MTP remains a separate application setting.

The test cache has 5120 FP16 tokens and four recurrent slots, with history
capacity four so the same loaded model can run ordinary and MTP suites. This
differs from the service's configured 65536-token context. Prompts are short;
the experiment does not establish long-prompt prefill latency or end-to-end
HTTP/OpenCode latency. Measurements use the actual ExLlama generator loop.
Per-job first-token latency is time from enqueue to first delivered token,
completion latency includes queue/prefill time, and delivery gaps expose stalls
around a new arrival. Aggregate throughput includes prefill and the complete
cohort lifetime; output lengths differ so the batch shrinks during a run.
The report additionally separates throughput by actual batch/query shape,
preventing a faster single-job tail from hiding multi-job MTP fallback cost.

### Build and installed validation provenance

Core concurrency change: `ada76f6`. The wheel is
`exllamav3-1.5.1+nativeqwen.concurrent1-cp313-cp313-win_amd64.whl`,
132,509,380 bytes, SHA-256
`0be06997f5ad556aa9e4d69f44e93132dfb9eaef44546991f403e33425875c74`.
It uses the same SM89/SM120 build workflow with four compiler workers. No kernels, dependencies,
CUDA/Torch versions or service configuration were changed. All 31 guard/profile
unit tests passed. Installed Python files and the extension were compared
byte-for-byte with the wheel; wheel Python was also compared with the checkout.
The dependency freeze differs only in the ExLlama wheel reference.

The staged build passed all ten concurrency cases and the existing non-MTP
128-step, natural-MTP 128-token and forced-MTP 64-token exact regressions.
Those proofs are retained in `native-results/concurrency-staged`. Since the
installed build has identical code/extension bytes, the unchanged proofs are
carried into `concurrency-installed` with `validation-provenance.json` recording
their origin. Strengthened staggered/page-boundary cases run again against the
installed package; they add explicit prefill/repair isolation checks, three
decode cycles at the reduced batch before arrival, and overlapping 256/272
ordinary and 256/280 MTP output lengths at page boundaries.
The installed runner started before the harness commit `34629fd`, so its
metadata records parent `931eb3f`; the working tree already contained the
prefill, page-reuse and phase-timing changes committed as `34629fd`.

### Final validation results

All cases below passed exact stock/native comparison. Event counts count
the recorded comparisons, not generated tokens. The four strengthened cases
ran against the installed wheel; the remaining cases use the identical staged
wheel as described above.

| Case | Compared events | Native / fallback block calls |
|---|---:|---:|
| ordinary-2 | 2138 | 1920 / 96 |
| ordinary-3 | 3003 | 2688 / 144 |
| ordinary-4 | 3868 | 3456 / 192 |
| ordinary-stagger | 4738 | 4224 / 240 |
| ordinary-stochastic | 2546 | 2304 / 96 |
| ordinary-page-crossing | 13972 | 13056 / 96 |
| mtp-2 | 1838 | 768 / 604 |
| mtp-3 | 1586 | 268 / 901 |
| mtp-stagger-forced | 2689 | 464 / 1439 |
| mtp-page-crossing | 6575 | 1352 / 3346 |
| ordinary-page-reuse (1024-token cache) | 4738 | 4224 / 240 |

The constrained four-page cache reassigned 1 physical page(s)
to a new job after previous owners completed, with exact state parity.
The installed ordinary stagger case observed 4 -> 3 -> 2 -> 3 -> 2 -> 1.
Original single-job regressions also passed: ordinary 128 steps; natural MTP
128 tokens (91 accepted / 57 rejected drafts); forced MTP 64 tokens
(34 accepted / 86 rejected).

### Alternating installed benchmarks

Three stock/native pairs per cohort, ordered A/B, B/A, A/B, after full-cohort
warmup of both paths. All 48 measured runs had identical per-job token IDs,
completion hashes, RNG state and MTP acceptance counts within their cohort.
Output lengths were 384/512/640/768 tokens; staggered arrivals added 256.
The same model, physical GPU 2, mapping 2,0,1, 300 W limit, prompts, greedy
sampler, seeds, FP16 cache and fixed-four MTP settings were used throughout.

| Cohort | Stock output t/s | Native output t/s | Change | GPU utilization stock/native | GPU watts stock/native | Native / fallback calls per native run |
|---|---:|---:|---:|---:|---:|---:|
| ordinary-1 | 53.96 | 70.95 | +31.49% | 59.5% / 73.8% | 166.2 / 186.5 | 18432 / 48 |
| ordinary-2 | 84.07 | 104.98 | +24.88% | 62.8% / 75.7% | 186.8 / 208.4 | 24576 / 96 |
| ordinary-3 | 110.77 | 131.79 | +18.98% | 66.9% / 77.8% | 208.2 / 227.3 | 30720 / 144 |
| ordinary-4 | 138.87 | 155.20 | +11.75% | 72.7% / 78.8% | 225.4 / 236.5 | 36864 / 192 |
| ordinary-4-stagger | 146.13 | 154.40 | +5.66% | 72.2% / 75.3% | 223.6 / 226.6 | 37008 / 240 |
| mtp-2 | 97.52 | 97.83 | +0.32% | 53.6% / 52.3% | 184.4 / 183.8 | 3248 / 4970 |
| mtp-3 | 93.34 | 98.18 | +5.19% | 45.1% / 47.3% | 168.5 / 171.8 | 2800 / 7917 |
| mtp-3-stagger | 97.74 | 97.83 | +0.10% | 45.8% / 45.5% | 172.2 / 171.9 | 776 / 9953 |

GPU figures are one-second telemetry means restricted to measured run intervals.
Every sampled power limit was 300 W. Stock runs have zero native calls.
Fallback counts include prefill and MTP cache-repair calls. Raw per-shape
call counts, individual run rates and sample counts are in `benchmark.json`
and `summary.json`.

#### Per-job responsiveness

Each cell below is stock / native, averaged over three runs. Job IDs are
zero-based; the last job in a staggered cohort arrives later. Effective
per-job t/s includes its own enqueue-to-completion time.

| Cohort / job | Output t/s | First token ms | Completion s | Maximum delivery gap ms | Gap across arrival ms |
|---|---:|---:|---:|---:|---:|
| ordinary-1 / 0 | 53.96 / 70.95 | 102.68 / 96.13 | 7.12 / 5.41 | 27.25 / 18.22 | 0.00 / 0.00 |
| ordinary-2 / 0 | 45.97 / 56.84 | 185.81 / 180.66 | 8.35 / 6.76 | 27.66 / 19.14 | 0.00 / 0.00 |
| ordinary-2 / 1 | 48.04 / 59.99 | 185.81 / 180.66 | 10.66 / 8.53 | 27.66 / 19.14 | 0.00 / 0.00 |
| ordinary-3 / 0 | 43.75 / 50.14 | 304.62 / 297.44 | 8.78 / 7.66 | 28.29 / 20.78 | 0.00 / 0.00 |
| ordinary-3 / 1 | 44.40 / 51.87 | 304.62 / 297.44 | 11.53 / 9.87 | 28.29 / 21.52 | 0.00 / 0.00 |
| ordinary-3 / 2 | 46.16 / 54.92 | 304.62 / 297.44 | 13.87 / 11.65 | 28.29 / 21.52 | 0.00 / 0.00 |
| ordinary-4 / 0 | 43.45 / 45.34 | 376.89 / 371.98 | 8.84 / 8.47 | 28.40 / 22.44 | 0.00 / 0.00 |
| ordinary-4 / 1 | 44.19 / 46.91 | 376.89 / 371.98 | 11.59 / 10.91 | 28.40 / 22.44 | 0.00 / 0.00 |
| ordinary-4 / 2 | 44.75 / 48.84 | 376.89 / 371.98 | 14.30 / 13.10 | 28.40 / 22.44 | 0.00 / 0.00 |
| ordinary-4 / 3 | 46.29 / 51.73 | 376.89 / 371.98 | 16.59 / 14.85 | 28.40 / 22.44 | 0.00 / 0.00 |
| ordinary-4-stagger / 0 | 42.44 / 41.28 | 377.72 / 379.46 | 9.05 / 9.49 | 29.36 / 91.12 | 0.00 / 0.00 |
| ordinary-4-stagger / 1 | 43.30 / 43.44 | 377.72 / 379.46 | 11.82 / 11.94 | 29.36 / 91.12 | 0.00 / 0.00 |
| ordinary-4-stagger / 2 | 43.45 / 44.65 | 377.72 / 379.46 | 14.73 / 14.46 | 108.18 / 143.32 | 108.18 / 101.17 |
| ordinary-4-stagger / 3 | 43.97 / 46.44 | 377.72 / 379.46 | 17.47 / 16.65 | 108.18 / 143.32 | 108.18 / 101.17 |
| ordinary-4-stagger / 4 | 45.48 / 54.46 | 108.15 / 101.14 | 5.63 / 4.70 | 32.96 / 20.68 | 0.00 / 0.00 |
| mtp-2 / 0 | 51.20 / 50.79 | 254.36 / 262.35 | 7.50 / 7.56 | 91.77 / 95.89 | 0.00 / 0.00 |
| mtp-2 / 1 | 55.73 / 55.91 | 254.36 / 262.35 | 9.19 / 9.16 | 91.77 / 95.89 | 0.00 / 0.00 |
| mtp-3 / 0 | 38.95 / 39.32 | 379.94 / 340.73 | 9.86 / 9.77 | 105.73 / 103.96 | 0.00 / 0.00 |
| mtp-3 / 1 | 33.79 / 35.52 | 379.94 / 340.73 | 15.18 / 14.43 | 117.20 / 103.96 | 0.00 / 0.00 |
| mtp-3 / 2 | 38.89 / 40.91 | 379.94 / 340.73 | 16.49 / 15.65 | 117.20 / 103.96 | 0.00 / 0.00 |
| mtp-3-stagger / 0 | 39.32 / 39.10 | 381.55 / 377.87 | 9.77 / 9.82 | 108.91 / 108.74 | 0.00 / 0.00 |
| mtp-3-stagger / 1 | 33.62 / 33.67 | 381.55 / 377.87 | 15.23 / 15.21 | 183.72 / 176.86 | 183.72 / 176.86 |
| mtp-3-stagger / 2 | 34.91 / 34.94 | 381.55 / 377.87 | 18.33 / 18.32 | 183.72 / 176.86 | 183.72 / 176.86 |
| mtp-3-stagger / 3 | 30.81 / 31.10 | 183.68 / 176.82 | 8.31 / 8.24 | 102.54 / 119.16 | 0.00 / 0.00 |

#### MTP acceptance and concurrent phases

Acceptance counts are identical in stock and native runs, including the
different histories of simultaneous jobs. Counts below are per run.

| Cohort / job | Proposed | Accepted | Rejected | Acceptance |
|---|---:|---:|---:|---:|
| mtp-2 / 0 | 388 | 287 | 101 | 73.97% |
| mtp-2 / 1 | 608 | 360 | 248 | 59.21% |
| mtp-3 / 0 | 388 | 287 | 101 | 73.97% |
| mtp-3 / 1 | 616 | 358 | 258 | 58.12% |
| mtp-3 / 2 | 784 | 444 | 340 | 56.63% |
| mtp-3-stagger / 0 | 388 | 287 | 101 | 73.97% |
| mtp-3-stagger / 1 | 612 | 359 | 253 | 58.66% |
| mtp-3-stagger / 2 | 776 | 446 | 330 | 57.47% |
| mtp-3-stagger / 3 | 376 | 162 | 214 | 43.09% |

Phase throughput divides delivered tokens by total generator-iteration time
at that target shape across all repeats. It includes draft/repair work and
prefill on iterations that perform it; it is not kernel-only timing.

| Cohort | Target shape | Stock phase t/s | Native phase t/s |
|---|---|---:|---:|
| ordinary-1 | b1/q1 | 53.98 | 70.99 |
| ordinary-2 | b1/q1 | 55.56 | 72.01 |
| ordinary-2 | b2/q1 | 91.98 | 113.73 |
| ordinary-3 | b1/q1 | 54.87 | 71.81 |
| ordinary-3 | b2/q1 | 92.98 | 115.81 |
| ordinary-3 | b3/q1 | 131.30 | 150.48 |
| ordinary-4 | b1/q1 | 55.94 | 73.51 |
| ordinary-4 | b2/q1 | 94.39 | 117.00 |
| ordinary-4 | b3/q1 | 139.76 | 157.15 |
| ordinary-4 | b4/q1 | 173.83 | 181.45 |
| ordinary-4-stagger | b1/q1 | 55.95 | 73.79 |
| ordinary-4-stagger | b2/q1 | 93.63 | 117.09 |
| ordinary-4-stagger | b3/q1 | 135.15 | 154.42 |
| ordinary-4-stagger | b4/q1 | 169.84 | 161.91 |
| mtp-2 | b1/q5 | 113.85 | 120.28 |
| mtp-2 | b2/q5 | 93.88 | 93.12 |
| mtp-3 | b1/q5 | 105.38 | 111.59 |
| mtp-3 | b2/q5 | 71.36 | 81.53 |
| mtp-3 | b3/q5 | 103.37 | 104.37 |
| mtp-3-stagger | b2/q5 | 79.76 | 79.59 |
| mtp-3-stagger | b3/q5 | 101.80 | 101.95 |

### Interpretation and limitations

Ordinary cohort throughput improves by 24.9%, 19.0% and 11.8% for initial
two/three/four-job cohorts. The gain while **all** initial jobs remain active
is smaller: 23.6%, 14.6% and 4.4%, respectively. The faster smaller-batch tail
contributes to cohort throughput. Four-job native effective per-job rates are
45.34/46.91/48.84/51.73 t/s, versus stock 43.45/44.19/44.75/46.29. First-token
latency is about 372 ms versus 377 ms for these short prompts.

No job starved: all requested outputs completed, shorter ordinary jobs finished
before longer jobs, and new jobs advanced immediately after their prefill.
Ordinary staggered arrival interrupted existing jobs for an average 108 ms stock
versus 101 ms native; the new job's effective rate was 45.48 versus 54.46 t/s.
This only establishes responsiveness for the tested short prompts, not a
latency guarantee for arbitrary request queues or long prefill.

There is one retained ordinary-stagger native outlier: aggregate rates were
162.94, **137.27**, 162.99 t/s versus stock 145.12, 147.14, 146.12. The slow run
had a 228 ms delivery gap affecting all four initial jobs **before** the new job
arrived. It raised the shortest job's completion time to 11.48 s versus about
8.50 s in the other native runs. The new job still completed in 4.69 s and its
arrival gap was 102 ms. The cause is not established by these measurements;
it is not justified to attribute it to scheduling, the driver or the native
code. No result was excluded. Consequently the mean earliest-job latency in
this staggered cohort regresses despite the aggregate improvement.

Concurrent MTP is correct but offers little consistent native speedup here.
Two-job cohort gain is 0.3%; its actual B=2,Q=5 phase is 0.8% slower. The
three-job cohort mean improves 5.2%, but stock runs span 89.71--98.98 t/s and
native 94.49--100.48, while its B=3,Q=5 phase improves only 1.0%. That cohort
mean is not evidence of a reliable five-percent concurrent-target optimization.
Staggered MTP, with no single-job tail, changes only 0.1%. It has average
arrival gaps of 184 ms stock / 177 ms native and native effective per-job
rates of 39.10/33.67/34.94/31.10 t/s. Unequal draft acceptance affects delivered
tokens per iteration even though all ready jobs participate. For these measured
workloads ordinary native batching outperforms MTP at two/three active jobs.
The concurrent target continues to use the stock residual/MoE kernel family.

Support remains limited to the tested Qwen3.8/Qwen4Exp single-device serialized
generator: ordinary B<=4,Q=1; attached one-head MTP draft B<=3,Q=1; existing
single-job fixed-four MTP verification. Multi-job target verification,
four-job MTP drafting, other speculative windows/query shapes, TP, offload,
prefill and the pre-existing exclusions fall back. Simultaneous independent
model.forward threads/streams are not supported. The existing AsyncGenerator
serializes generator iterations before dispatching results; this work changes
neither that behavior nor the service's request scheduling.

### Reproduction, installation and checkpoint

The final checkpoint is tag `validated-native-qwen-concurrent1-2026-09-24` on
branch `native-qwen-concurrency`. Implementation and harness commits are
`ada76f6`, `7f8af9b`, `931eb3f` and `34629fd`; the tag also contains this report.
The release directory is `dist/exllamav3-native-qwen-concurrent1-<tag-commit>/`.
It contains the exact tested wheel, source ZIP with validation results, a
standalone Git bundle including the previous checkpoint tags, a portable README,
build manifest and SHA-256 checksums. The ZIP and wheel receive archive CRC
verification, and the bundle is verified by Git. The wheel is already installed
in the existing TabbyAPI venv; dependencies and service configuration are unchanged.
Old checkpoint/release directories remain untouched.

Full reproduction with the installed wheel (using the exact environment):

```powershell
$env:CUDA_VISIBLE_DEVICES='2,0,1'
& D:\tabbyAPI\venv\Scripts\python.exe -u util\profile_qwen_concurrency.py --installed --regressions --benchmark --output native-results\concurrency-repeat
& D:\tabbyAPI\venv\Scripts\python.exe -u util\profile_qwen_concurrency.py --installed --suite ordinary --case ordinary-page-reuse --cache-tokens 1024 --output native-results\concurrency-reuse-repeat
```

The profiler selects stock/native internally; production uses
`EXL3_NATIVE_QWEN_BLOCK=1` set before application startup. MTP remains separately
configured in the application. Record GPU data in another shell during timing:

```powershell
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,power.draw,power.limit,memory.used --format=csv -l 1 -f native-results\concurrency-repeat-gpu.csv
```

`util/summarize_qwen_concurrency.py` combines `benchmark.json` with that CSV.
Raw proofs are in the release's `results/concurrency-installed` and
`results/concurrency-reuse`; `validation-provenance.json` identifies staged
proofs carried over from the identical wheel. The model itself is not packaged.

Install the portable wheel with the compatible environment's Python and
`-m pip install --no-deps --force-reinstall <wheel-path>`. To restore the
previous validated MTP build on this machine:

```powershell
& D:\tabbyAPI\venv\Scripts\python.exe -m pip install --no-deps --force-reinstall D:\exllamav3\dist\exllamav3-native-qwen-mtp1-34e7a06\exllamav3-1.5.1+nativeqwen.mtp1-cp313-cp313-win_amd64.whl
```

Clearing `EXL3_NATIVE_QWEN_BLOCK` and restarting instead selects stock execution
in the current wheel. The original official-wheel rollback script remains
`util/rollback_exllamav3.ps1`.
