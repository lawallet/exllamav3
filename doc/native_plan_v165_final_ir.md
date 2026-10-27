# GLM stage graphs on the v1.6.5 integration

This follow-up merges `feature/native-ir-final-optimization` at
`232f305c21b20369b9a660b34e6b3be4e183284e` into
`integration/v1.6.5-native-ir-tp`, starting from the validated v1.6.5 merge
`b81b61bba6a54165061e3f47db930ff60aaa38b5`. It brings in both deferred commits:

* `ea9b615c`: GLM TP native HC stage graphs and their validation tools/tests.
* `232f305c`: the historical installed-wheel TabbyAPI validation record.

The runtime optimization enables graphs only for GLM TP. DeepSeek remains
eager, and the graph policy tests retain that explicit restriction. Qwen,
Gemma, Devstral and Laguna activation policies are unchanged. Set
`EXL3_NATIVE_TP_GRAPHS=0` before process startup to opt out; an extension lacking
the graph API also retains eager behavior.

The only textual conflict was `hc_mix.cuh`: v1.6.5's fused HC declaration and
the incoming graph declarations were inserted in the same location. Both APIs
are retained. The new block-plan capture uses v1.6.5's `GraphCapture` scope,
preserving automatic capture cleanup if execution raises. Other graph and HC
changes merged without conflicts. The earlier deferred-residual and MGEMM
compatibility fixes remain intact.

The original branch, source optimization branch, previous checkpoint tag and
wheel are preserved. No remote push or installation into TabbyAPI is part of
this follow-up. The installed-runtime statements in
[the original optimization report](final_ir_optimization.md) describe the
October 25 campaign, not installation of this v1.6.5 build.

## Build and validation

Evidence is under `native-results/v1.6.5-final-ir`. A clean isolated build uses
the existing Python 3.13, Torch 2.9.0+cu128, CUDA 12.8 and MSVC toolchain, with
both sm_89 and sm_120. The wheel retains version `1.6.5+nativeplan.v2.tp1` and
adds the distinct wheel build tag `1finalir`, preserving the previous artifact.
The build script and sequential validation commands are retained with the logs.

The wheel is
`exllamav3-1.6.5+nativeplan.v2.tp1-1finalir-cp313-cp313-win_amd64.whl`
(32,435,155 bytes), SHA-256
`adf325515e9260a458bbb283a1ef0f091b72cea2d185ef55a41b1e59597fa41c`.
Its extension SHA-256 is
`d52233f5f71613e1b4ea1c6f4656abc65739e67ac412c04e49477aeff019859d`.
`package/extracted-proof.json` verifies every packaged production source,
ZIP integrity and imports of that exact extracted package and extension.
`cuobjdump` confirms 169 sm_89 and 169 sm_120 images in the binary.

Focused validation uses this new binary. The initial guard invocation
incorrectly enabled the global IR flag expected to be off by legacy mocked
constructor tests. The corrected launcher passes without changing those tests
or production behavior; the initial log is retained in `validation`.
Final results are recorded under `validation-r2`:

| Gate | Result |
| --- | --- |
| Native/dispatch/graph guards, shared HC and generator failure containment | 390 passed, 108 subtests; no skips |
| Shared HC cases repeated on Ada | 171 passed |
| GLM HC primitive cases with synthetic operands | All seven tested batch/query shapes passed |
| Actual DeepSeek target/draft HC specializations | 414 passed, retaining the eager activation policy |
| GLM TP2, batches 1–4, MTP and reload | Both load cycles passed exact logits, state, traces and generation |
| GLM TP3, batches 1–4 and MTP | Exact local-stage/rank checks and the existing stock-repeat error envelope passed; all generator cases passed |

For TP2 B1/Q1, each rank recorded 360 native block executions, 90 stage graph
captures and 720 replays. The validator requires exactly two replays per native
execution. The reload run destroys/rebuilds plans and repeats correctness and
generation checks; final unload returns parent allocations to 138,789,888 bytes
after garbage collection.

Three-rank end-to-end runs retain the documented CPU reduction-order variation;
independent global logits/completions are not claimed bit-identical. The exact
stage oracle uses the same reduced inputs, and no tolerance was changed.

## Matched GLM MTP comparison

The same loaded two-rank model, 65,536-token caches, batch 1, prompt, greedy
sampler and draft settings were used for eager native and graph modes. Each
mode was warmed through the longest request before three alternating timed
pairs per length. These compare modes within this merged v1.6.5 build, not
different package versions or the historical October 25 measurements.

| Generated tokens | Eager native median tok/s | Graph median tok/s | Median paired change |
| --- | ---: | ---: | ---: |
| 128 | 83.66 | 90.59 | +9.11% |
| 1,024 | 81.54 | 89.54 | +9.94% |

Every same-length sample had identical completion hashes and draft acceptance/
rejection counts. No timed sample created a new graph capture. The three
1,024-token paired gains were 10.15%, 9.94% and 9.82%. These are short controlled
measurements, not a claim that every workload gains the same amount. Full
results and counters are in `validation-r2/glm-mtp-comparison`; the compact
summary is `validation-summary.json`.

The extracted wheel also passed a fresh-cache cold start with GLM TP2, MTP,
65,536-token caches and batch capacity four. All 18 automatic warmup passes
completed in 209.43 seconds before the first request. Two subsequent requests
each generated 32 tokens successfully (7.47 seconds combined, including prefill
and remaining first-request compilation). Each rank recorded 810 native block
executions, 90 captures and 1,620 replays. These timings describe cold-start
behavior and are not throughput benchmark samples.

## Checkpoint artifacts

The local checkpoint tag is `native-plan-v1.6.5-final-ir-v1`. The verified wheel,
this report, validation summary, package proof and commit manifest are collected
under `dist/native-ir-tp-1.6.5-final-ir`, with a `SHA256SUMS.txt` manifest.
`checkpoint.json` records the final merge SHA and both parents. The earlier
`native-plan-v1.6.5-integration-v1` checkpoint remains available separately.

## Scope

This follow-up reruns the tests affected by shared HC/graph code and GLM TP.
It does not claim a new run of the full 6,740-pass v1.6.5 campaign, all other
models' end-to-end workloads, an 8,192-token benchmark, or TabbyAPI HTTP tests.
The inherited DeepSeek sustained-MTP repeatability limitation remains recorded
in `final_ir_optimization.md`; this merge does not enable DeepSeek graphs or
claim to resolve that issue.
