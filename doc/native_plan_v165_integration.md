# ExLlamaV3 v1.6.5 native IR / TP integration

The subsequent merge of the deferred GLM optimization commits is documented in
[the follow-up report](native_plan_v165_final_ir.md). This report retains the
scope and measurements of the original `native-plan-v1.6.5-integration-v1` checkpoint.

## Provenance

The original branch is `feature/native-ir-tensor-parallel`, at
`68e4c7954a74ebea0df14e5bb7cce42619d406c5`. Integration takes place on
`integration/v1.6.5-native-ir-tp`, created from that exact commit. The original
branch is retained. The user authorized retaining the 20 pre-existing untracked
development artifacts; their initial SHA-256 hashes are recorded separately.
The newer `feature/native-ir-final-optimization` branch (`232f305c`) is two
commits ahead of this base. After reviewing that distinction, the user explicitly
directed completing this integration first and incorporating those commits later.
They are outside this checkpoint's scope.

The official `https://github.com/turboderp-org/exllamav3.git` tag `v1.6.5` was
fetched explicitly and verified with `git ls-remote`. It resolves to
`cdc785db89e263e65e5849cb53e1ada156411414` (`Bump to v1.6.5`). This is a normal
merge of that commit, with no fetch or merge of upstream master. No remote push
or production package installation is part of this integration.

## Merge and compatibility changes

The only textual conflict was `exllamav3/version.py`. Its resolution is
`1.6.5+nativeplan.v2.tp1`, retaining an identifiable custom build of the upstream
release. Existing native IR implementations, architecture registrations,
fallback guards, TP binding and diagnostics, and cold-start warmup remain.

Semantic review and validation required these targeted adaptations:

* A stock block can now leave a dense residual add or mHC update pending for
  its successor. Native entry points materialize those pending updates before
  their first instruction. Ineligible/native-disabled paths keep upstream's
  folding protocol. Regression tests cover stock-to-native transitions, all
  three native entry paths, consumption exactly once, and stock fallback.
* Upstream's new forced-tile validation rejected MGEMM widths that the existing
  custom kernel handles with its 128-column fallback. CUDA MGEMM validates that
  actual N contract while retaining K divisibility and shared-memory checks.
  Single GEMM and ROCm retain upstream's original validation contract. No
  kernel arithmetic or autotuning policy was changed by this adaptation.
* Generator startup recognizes model stubs without a `loaded_tp` attribute as
  non-TP. Actual loaded TP models retain the existing automatic warmup.
* The upstream Qwen optimizer test fixture now supplies `kda_direct=False`,
  matching the production constructor and keeping its topology assertions.
* The upstream PTX portability test's CUDA translation unit includes ATen
  rather than Python binding headers, avoiding a Torch 2.9/MSVC compilation
  error. Its kernel and exact PTX/NumPy assertions are unchanged.
* The upstream TP collective test harness closes its redirected worker logs
  before removing the temporary directory. Windows forbids deleting those
  open files; previously this cleanup error masked successful collective runs.
* A TP context test passes the creator's shared-memory extent to its child;
  Windows reports a page-rounded extent when reopening the mapping, placing
  the old child's completion marker at a different address.
* Warmup and DFlash2 test stubs now supply the production constructor's fields.
  The stale DFlash2 TP-rejection test checks successful attachment, matching
  the tagged release's newly supported TP-target path. Production DFlash2
  behavior is unchanged.
* Synthetic linear and sparse-head test builders close their checkpoint
  collections after loading weights. Their repeated cases otherwise exhaust
  Windows' C-runtime file handles. Numerical assertions are unchanged.
* Shared-memory tests mock the POSIX capacity query without modifying the
  process-wide OS module, count Windows handles instead of Linux file
  descriptors, and check mapping removal after the last Windows handle closes.
  Two affinity round trips explicitly skip processes spanning multiple Windows
  processor groups, which the production single-group planner already rejects.
  The queue-drained fixture supplies the new sparse-head field.
* The GLM smoke benchmark reserves enough cache for its fixed 256-token parity
  check and 128-token warmup, even when the timed smoke request is shorter.
  Its validation budget reproduces the bespoke backend's existing 35/10 layer
  placement under the new loader; the production eligibility guard is unchanged.
* The DeepSeek reload probe saves both placements and both native results,
  then checks exact stock/native parity on the reloaded placement. It requires
  cross-reload equality when placement is unchanged. Autosplit moved layer 34
  from Ada to Blackwell here, so cross-placement equality is reported as false,
  not silently treated as a same-placement determinism failure or a pass.

The TP validation probe was adapted to v1.6.5's two GDN scan rows per sequence:
it hashes the committed parity row, while convolution and other recurrent
families continue to use sequence-slot indexing. A regression checks this
distinction. Its optional `--single-parity` mode additionally compares stock
and native logits and generator results on a single device.

## Build and validation environment

The existing `D:\tabbyAPI\venv\Scripts\python.exe` provides Python 3.13.12,
PyTorch 2.9.0+cu128 and the existing Triton runtime. The wheel is built with
MSVC 2022 and CUDA 12.8 for Windows x64, with `TORCH_CUDA_ARCH_LIST=8.9;12.0`.
`cuobjdump --list-elf` confirmed both sm_89 and sm_120 images in the binary.
The three available GPUs are two RTX 4090 D cards and an RTX PRO 6000 Blackwell
Max-Q. Validation uses noncontiguous physical-device mappings, independently
of production device selection.

`pytest`, `accelerate` and `zstandard` were absent from the pinned interpreter. Test-only dependencies were
installed under `native-results/v1.6.5-integration/test-deps`; the installed
TabbyAPI environment was not changed. Initial attempts, failures, and reruns
remain in the evidence directory.

Reproduction tools:

* `util/build_v165_integration.bat` builds an isolated wheel. Its
  `--python-only` option refreshes Python packaging without relinking an
  already validated native binary.
* `util/validate_v165_integration.py` runs selected existing gates in separate
  processes and records commands, mappings, timings, source hashes and binary
  identity. `--suite-paths` selects pytest paths for diagnostic reruns.
  `--devices` selects visibility order; absent an explicit visibility setting,
  the runner orders the available GPUs by memory rather than physical index.
* `util/verify_v165_wheel.py` checks package-source bytes against the checkout,
  verifies ZIP integrity, extracts the wheel, and imports that exact extracted
  package and extension without installing them.
* `util/test_v165_integration.bat` initializes MSVC/CUDA before invoking the
  validation driver, for tests such as the portability probe that JIT-build
  their own small extensions.

## Validation status

Available validation is complete. Across the initial run and focused reruns,
all 7,234 collected test cases are accounted for: **6,740 passed, 494 skipped,
zero remaining failures and zero unrun collected cases**. This is a consolidated
result, not a claim that the first uninterrupted run passed. Original failure
logs remain available. `coverage-summary.json` maps each case to its final
evidence under `native-results/v1.6.5-integration`.

| Gate | Result |
| --- | --- |
| Native/dispatch regression rerun and focused fixtures | 215 passed, plus 51 subtests |
| Opt-in slow suite | 8 passed, including large-offset kernels, CPU pool stress and TP timeout behavior |
| Mixed-width MGEMM | 192 cases on Blackwell and 192 on Ada passed |
| GLM, DeepSeek and Gemma native primitive/site gates | Passed |
| Qwen stock/bespoke/plan layer comparison | 128 steps, exact logits, tokens, boundaries and state |
| GLM stock/bespoke/plan layer comparison | 256 steps, exact logits, tokens, boundaries and state |
| DeepSeek layer-split shapes | Exact B1–4/Q1 and B1–2/Q6; live state and boundaries included |
| DeepSeek reload | Native bindings cleared/rebuilt; exact stock/native results after autosplit changed placement |
| Tiny HF references | DeepSeek V4, Mistral4 and Muse Glimmer numerical/mutation gates passed |
| Frozen n-gram corpus modules | 9 additional tests passed after adding the isolated optional dependency |

The final layer evidence is in `legacy-r1/qwen-layer`, `glm-layer-r3`,
`legacy-r1/deepseek-layer` (all six shape comparisons), and
`deepseek-reload-r3` (successful focused reload). The earlier GLM cache/budget
failures and DeepSeek cross-placement assertion remain recorded separately.

Skipped cases require unavailable registry checkpoints or unsupported
platform/ISA/hardware paths. All five checkpoints in the local model directory
were exercised by the real-model matrix below. A Laguna checkpoint, ROCm/Linux
hardware, full vision-input parity for every architecture, long-duration
serving, and a new full-64K prompt campaign were unavailable or not covered.
The old DeepSeek 64K report describes historical validation, not a new v1.6.5
result. No native eligibility contract or numerical tolerance was relaxed.

### Real-model candidate matrix

All thirteen candidate model/topology runs passed with batches 1–4 and
65,536-token cache capacity. Qwen, GLM and DeepSeek additionally exercised MTP.
Gemma passed unload/reload and repeated correctness/generation in all three
topologies. Qwen's single-device run uses the existing host-resident n-gram
table; it is not a wholly VRAM-resident baseline. GLM and DeepSeek were tested
with TP because their full workloads exceed the largest device's capacity.

Two-device and single-device checks require exact stock/native logits and
prefill output, with two-device trace and initialized-cache/state comparisons.
Three-device CPU reductions have arrival-order rounding variation. Those
checks require exact local native stages against the same reduced inputs,
identical rank traces, and final errors within the existing measured
stock-repeat envelope. Three-device generation similarly checks local stages;
it does not assert globally bit-identical stochastic reduction trajectories.
Qwen batched TP MTP retains its explicit, pre-existing small-row stock fallback.

Median ordinary decode throughput (tokens/s, three paired measurements;
stock/native within v1.6.5, not a comparison with the previous release):

| Model | Two GPUs stock / native | Three GPUs stock / native | One GPU stock / native |
| --- | ---: | ---: | ---: |
| Qwen | 31.39 / 31.46 | 25.18 / 31.61 | 45.38 / 69.48 |
| GLM | 39.85 / 51.43 | 40.03 / 49.19 | — |
| DeepSeek | 21.13 / 23.96 | 11.76 / 12.87 | — |
| Gemma | 44.84 / 45.84 | 32.81 / 36.03 | 41.41 / 41.42 |
| Devstral | 19.30 / 19.38 | 24.83 / 24.89 | 15.22 / 15.27 |

Model paths, allocation, native execution evidence, timings, generator results
and lifecycle memory records are retained under `models-r1/<model>-<topology>`.
These are short development measurements, not a long-duration serving benchmark.

The cold-start probe imports the extracted wheel itself, loads GLM with its
vision and MTP components, and uses a fresh Triton cache. All 18 automatic
warmup passes completed in 209.20 seconds before the first request. Two
352-token-prompt / 32-new-token generations then completed successfully;
both TP ranks recorded 810 native block executions. Evidence is
`models-r1/cold/result.json` and `cold.log`. Cache capacity is not a claim of
full 64K-prompt accuracy or a long-context serving stress test.

### Previous-wheel comparison

The earlier DeepSeek [state investigation](deepseek_v4_state_investigation.md)
identified unwritten narrow MGEMM projections, not benign numerical noise.
Its custom per-matrix fallback is retained and passes the expanded mixed-width
regression on both GPU architectures. The later [TP report](native_ir_tp.md)
separately documents three-rank reduction-order variation and the compressor's
partially written overlap ring; those are distinct from that resolved bug.

The original-head artifact is
`dist/native-ir-tp-1.5.4-tp2/exllamav3-1.5.4+nativeplan.v2.tp2-cp313-cp313-win_amd64.whl`,
whose release manifest identifies original commit `68e4c795`. Both versions
used the same two-device mapping, models, batch/step settings and three paired
timing samples. The baseline runs also passed their correctness gates.

| Model | Previous / new native decode, tokens/s | Change | Previous / new native MTP, tokens/s |
| --- | ---: | ---: | ---: |
| Qwen | 31.91 / 31.46 | −1.4% | 58.50 / 59.56 |
| GLM | 49.67 / 51.43 | +3.5% | 77.83 / 84.24 |
| DeepSeek | 24.17 / 23.96 | −0.9% | 40.28 / 29.91 |
| Gemma | 45.83 / 45.84 | +0.0% | — |
| Devstral | 18.31 / 19.38 | +5.9% | — |

These short measurements do not establish statistical significance for small
differences. DeepSeek's observed MTP decrease is material (25.8%): the
128-token greedy completion changed between versions, with accepted draft
tokens changing from 83 to 70. Its stock path also slowed (36.13 to 28.41
tokens/s), while stock/native output parity within each release passed. A
fresh stock-only control produced the identical completion hash and 70/35
accepted/rejected counts in every sample using both the merged package and
exact-tag Python, with identical TP allocation. Warm medians were 26.59 and
23.04 tokens/s respectively; these short controls were noisy. Both used the
same validated custom extension, so this isolates the Python integration,
not every custom C++ change. Evidence is in `deepseek-{merged,upstream}-control`.
The changed completion and speculative workload also occur with exact-tag
Python; the observed version-to-version slowdown is not being reported as
preserved MTP performance or attributed solely to native IR.

## Wheel and checkpoint

`exllamav3-1.6.5+nativeplan.v2.tp1-cp313-cp313-win_amd64.whl` was built in
`native-results/v1.6.5-integration/package/wheels` (32,429,019 bytes), with the
final copy in `dist/native-ir-tp-1.6.5`.
Its SHA-256 is
`7765fc5524c3246c0576f498dd8d4e12cf587bef66d951a9c7c9f6863851a7be`.
The extension SHA-256 is
`bdc27be918e6a2501c32c14f0f12896cfe1a991f197a1399fbbcdc8e34d9ac1c`.
The `package/extracted-final-proof.json` evidence verifies source-byte equality,
ZIP integrity, imports, and the four existing native execution APIs. All
subsequent test-only fixes leave this package payload unchanged.

Checkpoint tag: `native-plan-v1.6.5-integration-v1`. The final merge commit,
parents, original branch, tag and artifact hashes are recorded in
`dist/native-ir-tp-1.6.5/checkpoint.json`. The original branch and all 20
pre-existing untracked artifacts remain unchanged. Nothing was installed into
the production runtime or pushed remotely. The two final-optimization commits
remain deferred for a subsequent integration, as requested.
