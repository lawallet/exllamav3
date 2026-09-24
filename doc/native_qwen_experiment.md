# Qwen3.8 native orchestration experiment

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
