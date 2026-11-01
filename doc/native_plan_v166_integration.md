# ExLlamaV3 v1.6.6 integration

Branch: `integration/v1.6.6-native-ir-moe`.
The final merge commit is the commit containing this report (see `git log -1`).
No remote push is part of this release.

## Provenance and preservation

The starting branch was `feature/native-ir-moe-assignment` at
`d62891dee3e4cf233207c0a7ec765adba4b7bc6f`. Its uncommitted validated Nemotron
implementation was preserved in checkpoint `27cf16fe` on the new integration
branch. The original branch was not advanced or rewritten. Original modified
and untracked files were copied, with their relative paths and status, to
`native-results/v1.6.6-integration/premerge`. Unrelated untracked prompts and
reports remain in the working tree.

Official upstream tag `v1.6.6` resolves to
`bd1f519172406a9560bd5c90e61000691bb69a07`. Its main changes include stock
DeepSeek V4.1 and LFM3/VL architectures, MoE assignment synchronization changes,
CPU MoE fractional weights/promotion/pinning, packed FP4/FP8 conversion and MoE
affinity, generator/sampler batching, loop detection, disk-cache changes, and
loader/prefill updates. New architectures retain stock execution; this merge
does not add native support for them.

The only textual merge conflict was `exllamav3/version.py`, resolved to
`1.6.6+nativeplan.v2.tp.moe1`.

## Necessary fixes

* DeepSeek V4.1 carries hyperconnection pre weights between sites. The existing
  native plans encode same-site collapse. Registration, TP export and cached
  plan lookup now reject `carry_pre` before native execution. GLM target/draft
  guards also reject this unsupported semantic change. Existing architectures
  and their native plans remain eligible.
* Nemotron native assignment now explicitly declines the upstream conversion
  capture/preservation/affinity modes. The existing opt-in, device, stream,
  architecture, shape and deterministic-accumulation restrictions remain.
* Upstream's grouped MoE branch contained a second dispatch block under the
  no-scratch branch. With atomic accumulation enabled, it executed experts
  twice. A targeted numerical test reproduced doubled results before the fix
  for 4 and 20 rows. The duplicate block was removed; the new vectorized planner
  and first dispatch remain intact.
* Two test doubles were updated for upstream's new device/capability metadata.
  Their correctness assertions were unchanged. The benchmark harness now takes
  an explicit package root so an installed or extracted wheel can be tested
  without accidentally importing checkout code.

## Regression evidence

All evidence below is retained under `native-results/v1.6.6-integration`.
The final `release-check.json` audit verifies the consolidated outcomes,
26 successful runtime/control gates, wheel hashes, HTTP results and the GLM
single-GPU skip against their saved evidence.

| Gate | Result |
| --- | --- |
| Consolidated full suite | 7,020 passed, 508 skipped; all 7,528 collected cases accounted for; zero unresolved failures |
| Installed-wheel regression subset | 1,758 passed and 180 subtests passed; zero failures or skips |
| Opt-in slow tests | 8 passed, 191 skipped; included in the consolidated total |
| Variable-width MGEMM | 192 cases passed on each of Blackwell and Ada |
| GLM primitive stages | Seven production batch/query shapes passed |
| Actual-weight DeepSeek target/draft stages | 414 specializations passed |
| Gemma native sites | Passed |
| Atomic grouped MoE negative control | Failed before the fix, passed after it |
| Fresh-process stdio bootstrap | 2 existing regression canaries passed; UCRT limit verified as 512 before and 4,096 after |

The consolidated suite is **not** a claim that one uninterrupted pytest run
passed. The initial run exhausted the Windows UCRT stream limit late in the
suite. The same failure exists in saved v1.6.5 logs. Failed/unrun cases were
resumed in a fresh process; two warning tests then encountered Transformers'
lazy optional `torchvision` import after the Hugging Face tests. Those tests
passed unchanged in an isolated native-TP run. Per-node outcomes and their
evidence sources are in `coverage-summary.json`; the original failure logs are
retained. No assertions were relaxed.

The user's `_setmaxstdio(4096)` workaround addresses this exact stream ceiling,
but only in the process executing it. Pytest does not run `convert.py`. A probe
of the new extension failed after 63 collections at a limit of 512 (eight
streams per collection), then opened 96 collections successfully after the
limit was raised to 4,096. `stdio-proof.json` records that negative control.
The validation progress plugin now applies and verifies the setting in each
fresh pytest process; no global loader or conversion behavior was changed.

The first DeepSeek TP3 MTP run failed when two ranks compiled a previously
unseen kernel through the shared Windows Triton cache: a worker raised
`PermissionError` reading `_dsa_attn_combine_kernel.json`, followed by the other
ranks' collective timeout. The old validation harness already serialized its
null-backend warmup to avoid this class of cache replacement race, but that
warmup did not cover every generator specialization. The rerun uses a separate
Triton cache directory for each rank, configured only by the validation harness.
The failed log is retained in `runtime-followup/deepseek-three.log`. No compiler
dependency, model arithmetic, collective timeout or parity assertion was changed.
This is not a product-level fix for Windows shared-cache contention.

## Runtime matrix

Validation uses `D:\tabbyAPI\venv\Scripts\python.exe`, Python 3.13,
PyTorch 2.9.0+cu128, CUDA 12.8, Windows, and physical GPU order `2,0,1`.
One- and two-GPU cases use the corresponding prefix of that order.
The runtime harness stages only tests/utilities, checks import origins, and
compares selected-package hashes before and after execution.

GLM is deployed exclusively as TP2+MTP. Its validation prioritizes that setup;
the queued single-GPU run was canceled following the user's clarification.
No GLM single-GPU performance target or result is asserted. The harness requires
a specific shared-IR or regression reason for any future GLM single-GPU check,
and disables its timing runs. Existing primitive and installed-wheel regression
coverage already exercises the shared native IR functionality.

| Model | Completed configurations and checks |
| --- | --- |
| GLM 5.3 Flash | Priority TP2+MTP: B1–4 exact decode/live-state/generation parity, graph replay, concurrent requests and unload/reload. TP3 supplemental checks had already completed before the user's TP2-only clarification. No single-GPU run. |
| Qwen | Single GPU, TP2 and TP3, B1–4, concurrent MTP and prompt/cache reuse; existing host ngram table explicitly allowed for the single-GPU baseline |
| DeepSeek V4 | TP2 and TP3, B1–4 decode and concurrent MTP; TP3 completed with isolated per-rank Triton caches as explained above |
| Gemma | Single GPU, TP2 and TP3, B1–4, concurrent generation and a full unload/reload cycle on each topology |
| Devstral | Single GPU, TP2 and TP3, B1–4 decode and concurrent generation |
| Nemotron | Single GPU, 18 audit cohorts including reload, exact stock/native logits, tokens, acceptance, cache/state/refresh checks, R12/R16 dispatch and plain/cap8 stock controls |

Successful model evidence is divided between `runtime-priority`,
`runtime-followup` and `runtime-remaining`. Nemotron's audit generated 6,912
tokens across its paired cohorts and reload check; all parity assertions passed.

The real TabbyAPI HTTP smoke used the installed wheel, a temporary localhost
server and the Nemotron checkpoint: 8 stock/native cohorts, 20 concurrent/sequential
requests and 2,560 generated tokens passed. Its 120 same-input routed-expert
oracles were exact, and R12/R16 native dispatch was observed. The server exited
normally. Persistent `config.yml` and authentication-file hashes were unchanged.
Evidence is in `http/result.json`, `http/client.log` and the staged HTTP cohort
records. HTTP audit timings are not used as performance benchmarks.

TP3 decode uses the existing stock-repeat rounding envelope and exact local
native-stage checks. TP3 generation uses exact local-stage oracles; whole
trajectories can differ through reduction rounding. Qwen TP2/TP3 MTP
verification at batches 2–4 retains the existing stock fallback; its B1 native
path and ordinary B1–4 decode paths are separately exercised. Nemotron's B1
native-assignment flag is a control: its custom R12/R16 stage is ineligible there.

## Performance comparisons

GLM uses the user's actual deployment target, TP2+MTP, under the same TabbyAPI
Python/Torch environment. Both releases used the same checkpoint, GPU order,
warmup and benchmark harness. Medians of three uninstrumented measurements:

| GLM TP2+MTP | Previous wheel, tokens/s | v1.6.6 wheel, tokens/s | Change |
| --- | ---: | ---: | ---: |
| 128 tokens, eager native | 84.24 | 84.13 | -0.1% |
| 128 tokens, native CUDA graphs | 91.98 | 91.76 | -0.2% |
| 1,024 tokens, eager native | 83.28 | 82.76 | -0.6% |
| 1,024 tokens, native CUDA graphs | 90.70 | 90.56 | -0.2% |

Completion hashes and draft-acceptance counts match between releases in every GLM comparison. These
measurements show no material GLM TP2+MTP regression. Raw samples and ranges are
retained in `performance-summary.json`; the previous package's import origins
and hashes are in `baseline-priority/inputs.json`.

Qwen TP2+MTP used the same 128-token request in both releases. The initial
three-pair comparison suggested a slowdown, so both wheels were rerun with
five paired samples, retaining all original data:

| Qwen TP2+MTP native | Previous wheel, tokens/s | v1.6.6 wheel, tokens/s | Change |
| --- | ---: | ---: | ---: |
| Initial three-pair median | 58.85 | 53.24 | -9.5% |
| Fresh five-pair repeat median | 53.51 | 58.29 | +8.9% |

Completions, draft acceptance and native-stage execution counts match between
releases. The reversed timing difference does not reproduce a version-specific
slowdown. Short-run performance varies across processes; neither comparison is
presented as a reliable version-specific speedup. Evidence is in the original
priority runs and `qwen-repeat-candidate` / `qwen-repeat-baseline`.

Nemotron's cross-release stock token trajectories can differ. For example,
16K/B3 produced 504 measured tokens under v1.6.5 and 490 under v1.6.6 in the
same 48 MTP rounds, with different acceptance counts. This difference also
occurs with native assignment disabled; native and stock match exactly within
each release. Consequently, the performance summary includes fixed-round
throughput as well as tokens/s and does not attribute all token-rate changes
to compute speed. No cross-release bitwise guarantee is asserted for Nemotron.

| Nemotron context / batch | Previous native, tokens/s | v1.6.6 native, tokens/s | Change | v1.6.6 native gain over its stock path |
| --- | ---: | ---: | ---: | ---: |
| Short / B3 | 186.89 | 189.46 | +1.4% | +45.6% |
| Short / B4 | 232.50 | 212.27 | -8.7% | +33.9% |
| 4K / B3 | 191.10 | 186.69 | -2.3% | +32.6% |
| 4K / B4 | 220.23 | 214.68 | -2.5% | +13.8% |
| 16K / B3 | 186.35 | 183.10 | -1.7% | +47.0% |
| 16K / B4 | 222.40 | 216.36 | -2.7% | +14.0% |

These are the original 12-warmup/48-measured-round samples. The short/B4
medians were investigated with a separate longer-window repeat; all original
samples, including their overlapping ranges, remain in the evidence.
With 64 warmup and 128 measured rounds, native medians were 216.63 tokens/s
for the previous wheel and 220.96 for v1.6.6 (+2.0%). Fixed-round throughput
was 15.30 versus 15.37 rounds/s (+0.4%). The apparent short-window regression
did not reproduce. The candidate native path remained 24.0% faster than its
stock path in this extended-window control. Evidence is in
`nemotron-repeat-baseline` / `nemotron-repeat-candidate`.

The original 16K/B1 control varied sharply even though native MoE assignment
does not execute at B1. Repeating both releases with 128 warmup and 256 measured
rounds gave native-flag medians of 119.86 and 121.34 tokens/s respectively
(+1.2%). This resolves the original apparent slowdown without a product change.
The repeat evidence is in `baseline-priority/nemotron-control` and
`runtime-final/nemotron-control`; `runtime-final/selection.json` also records
the skipped GLM single-GPU gate.

The completed fixed-decode matrix also agrees with the historical v1.6.5
native baselines: Qwen changes range from +0.6% to +3.3%, Gemma from approximately
0% to +8.4%, Devstral from -1.2% to +0.4%, and DeepSeek TP2 is -0.6%. DeepSeek
TP3 measured +82%, but this historical comparison does not isolate the cause
or establish a gain attributable to this merge. Historical results are labeled
separately from fresh previous-wheel controls in `performance-summary.json`.

## Wheel

The extension was rebuilt from all 200 compilation units with MSVC, CUDA 12.8
and SM89/SM120 targets. The final wheel contains 171 CUDA ELF images for each
target. Packaging was refreshed after the Python compatibility fixes.

`D:\exllamav3\dist\native-ir-tp-moe-1.6.6\exllamav3-1.6.6+nativeplan.v2.tp.moe1-cp313-cp313-win_amd64.whl`

SHA-256:
`389d0eaf33bf57ae3b67de452af9db14cdd5057bf2851f4f576873a209a09f74`

The wheel was installed into TabbyAPI. Isolated imports, version metadata,
native ABI/entry points, ZIP integrity and every installed package/extension
payload byte were verified against it. Packaged source was also checked against
the checkout. The former installed wheel is preserved under
`dist/native-ir-moe-assignment`; its verified payload was extracted for fresh
baseline measurements without replacing the candidate installation.
Final checks again verified every packaged source against the checkout and
every installed payload byte against the wheel (`final-source-check-proof.json`
and `installed-proof-final.json`). The wheel hash remained unchanged.

## Adversarial review and limits

Review covered the HC kernel argument layout and CUDA-graph pointer patching,
transformer residual bridges, module pinning/device assumptions, native cached
plan guards, conversion capture modes, MoE scratch and arena lifetimes, and
stock grouped dispatch. Native IR v2, TP execution core and MGEMM dispatch
fixes were compared with the preserved checkpoint. Existing graph-disable and
invalid-norm regression protections remain. No native optimization was removed
merely because upstream added a similar optimization.

Final review found no unresolved numerical failure in the executed gates and
no repeatable material version-specific performance regression in the priority
workloads. The necessary compatibility fixes are guarded by regression tests.
Short benchmark variability and cross-release Nemotron trajectories are reported
above rather than being hidden by selecting favorable samples. Original failed
logs and all repeat measurements remain available for review.

Five trailing-whitespace lines reported by `git diff --check` are unchanged
from the official v1.6.6 `quantize.py`; no unrelated formatting edits were made.

There are no local full checkpoints for Laguna, DeepSeek V4.1 or LFM3/VL in this
campaign. Their available unit/guard coverage does not establish full-model
runtime validation. The local DeepSeek V4 weights total about 117.5 GB, so the
96 GB GPU cannot run that checkpoint alone. This campaign does not certify
Linux/ROCm or repeat the previous fully populated 64K/128K context campaigns.
Nemotron timing contexts extend to 16K; a large configured cache is not evidence
of a fully populated long-context test.
