# Stock CSA/indexer reproducibility investigation

## First milestone: earliest divergence

Three instrumented stock runs agree on layer 36's attention input, projection
input, projection weight bindings, output pointer table, widths, and pre-write
ring row. They first diverge at `exl3_mgemm` outputs 4 and 5, the indexer KV/gate
projections. Outputs 0..3 (query, shared KV and compressor projections) agree.
The write sources already differ; `dsv4_compress` writes those differences to
the correct live row. This is before indexer normalization, pooling or ring
write ordering could explain the mismatch.

Artifacts: `native-results/deepseek-v4/investigate-captures/{events,summary}.json`
and `samples.pt`, containing exact inputs, projections and live rows. Captures
include graph and eager execution, three stock repetitions each. Metadata
records pointers, underlying storage, offset, shape, strides, CUDA device and
stream. The suspect eager operation uses logical CUDA1's stream 0. Projection
output allocations are distinct and do not overlap the input or ring storage.
Position=300, slot=0, ring index=40 in all runs.

## Demonstrated mechanism

The projection fan has widths [1024,512,1024,1024,256,256], K=5, mul1.
MGEMM selects/autotunes its launch shape using only the carrier width (1024).
The captured eager calls select shape 4: TILESIZE_N=512. The per-matrix inner
kernel computes `tiles_n = size_n / TILESIZE_N`. For either 256-column indexer
matrix this is zero; every block returns without writing C. The subsequent
output Hadamard/scaling transform nevertheless reads and transforms C.
Consequently the initialized persistent ring receives results derived from
prior scratch contents, not the indexer matrix multiplication.

The minimal reproducer loads only six layer-36 projections and a captured
attention input. It uses no attention, cache, ring, scheduler or device
transfer. 240 calls cover two GPUs, automatic and forced shapes 1..4,
fresh/reused allocations, zeroed scratch, fills 0/1/2, synchronized calls, and
default/alternate streams. With shape 4, all-zero prior output produces an
all-zero indexer projection; prior fills 1 and 2 produce two other hashes.
The four wider outputs remain invariant. Shapes 1..3 produce a single invariant
indexer result regardless of prior output contents.

In this run automatic dispatch chooses shape 2 on the Blackwell GPU and shape
4 on the Ada GPU. Forcing shape 4 reproduces the defect on BOTH devices; it is
not a logical-ordinal, transfer-boundary or physical-GPU correctness condition.
The alternating layer pattern is architectural: CSA layers have the narrow
indexer projections; HCA layers do not. Which layers exhibit the defect depends
on the selected kernel shape, not a special rule for layers 36/38/40/42.

Evidence: `native-results/deepseek-v4/projection-isolated-v2/results.json` and
`indexer-projections.pt`. Early diagnostic failures (CPU/scalar metadata and
an argument-order mistake in the isolated harness) were retained separately.

Rejected causes: differing projection inputs/weights, ring index/slot selection,
cross-device copy, an attention-graph-only problem, and missing stream ordering.
Explicit synchronization and alternate streams preserve the fill dependency.
Fresh nonaliased output allocation also preserves it. This is a skipped matrix
calculation followed by a stale-output read, not ordinary floating-point noise.

## Narrow fix

In the shared MGEMM kernel, retain the selected tile for matrices whose widths
are compatible. For an incompatible narrower or partial width, call the same
inner calculation with a 128-column N tile and the same K tile, thread layout,
precision and row stride. EXL3 outputs are 128-aligned. This handles per-matrix
widths and sliced launches without host metadata reads, synchronization or
architecture-specific restrictions. Uniform compatible matrices keep their
original calculation. No generic-IR, emitter, attention, cache or scheduler
change is involved.

The fixed extension is built separately at `native-results/deepseek-v4/extension`;
the previously validated IR binary and all prior diagnostics are preserved.
Its SHA256 is
`40ca8e4f2910e72655dc42422af94d72dd629f314746c2d565840c9787c72473`.

## Before/after validation

The old extension fails the isolated actual-weight experiment on both GPUs
when shape 4 is forced. Reversing the logical device order (`0,2`) preserves
the shape-dependent failure. A model-free regression also fails with shape 3
at width 128, demonstrating that this is a shared variable-width kernel bug.

With the fixed extension, using the pinned TabbyAPI Python and normal visibility
`2,0,1`:

* All 240 actual-weight projection cases are independent of prior output
  contents. Previously valid forced shapes 1..3 retain exact outputs, as do
  the four compatible outputs of shape 4 (`projection-fixed/passed.json`).
* Five stock runs with attention graphs and five with eager attention match
  exactly across eight decode steps, including persistent state and the live
  indexer rows (`stock-fixed`).
* The original 32-step B1/Q1 stock/plan gate passes exact outputs, boundaries
  and persistent state. It constructs 43 plans and executes 1,376 calls at each
  stage with zero fallback (`stock-fixed/passed.json`).
* All 414 actual-weight DeepSeek native-stage cases pass again
  (`sites-state-fix.json`). The native guard/regression suite passes 113 tests
  (`guards-state-fix.log`).
* The model-free regression passes 75 cases on each participating GPU: mixed
  widths 128..1024, six row counts, FP16/FP32 outputs, forced and automatic
  shapes, and sliced launches with a wider row stride. Every result is exactly
  independent of prior output contents; slices preserve the untouched tail
  (`mgemm-fixed-slices-0.json`, `mgemm-fixed-slices-1.json`).
* Qwen (48 blocks) and GLM (45 blocks) each pass 128-step exact
  stock/bespoke/plan comparisons. Their complete recorded results also match
  the pre-fix extension exactly (`qwen-before.json`, `qwen-fixed.json`,
  `glm-before.json`, `glm-fixed.json`). GLM's initial cross-binary assertion
  compared live placement tuples with JSON lists; the harness now normalizes
  serialization. An exact comparison of both saved result objects passes
  (`glm-fixed-reference-pass.json`). The original failure log is retained.

Paths above are relative to `native-results/deepseek-v4`. All failed artifacts
remain available. No tolerance was increased or live state excluded. These
checks resolved the short persistent-state blocker. MTP, long-context and
performance work remained paused at completion of this investigation; the
subsequently resumed integration is recorded in
[the validation report](deepseek_v4_validation.md). No release was created.

## Reproduction

Use `D:\tabbyAPI\venv\Scripts\python.exe` with `CUDA_VISIBLE_DEVICES=2,0,1`.
Output directories for capture/isolation/stock runs must be new, to preserve
earlier evidence. The diagnostic tools default to the original extension;
post-fix checks must explicitly select the rebuilt one.

```powershell
$env:CUDA_VISIBLE_DEVICES='2,0,1'
$py='D:\tabbyAPI\venv\Scripts\python.exe'
$fixed='native-results\deepseek-v4\extension'
& $py util\investigate_deepseek_state.py --output native-results\deepseek-v4\capture-new
& $py util\isolate_deepseek_projection.py --samples native-results\deepseek-v4\investigate-captures\samples.pt --output native-results\deepseek-v4\isolate-new --extension-dir $fixed --reference native-results\deepseek-v4\projection-isolated-v2\results.json
& $py util\validate_deepseek_stock.py --extension-dir $fixed --output native-results\deepseek-v4\stock-new
& $py util\check_deepseek_plan_sites.py --extension-dir $fixed --output native-results\deepseek-v4\sites-new.json
& $py tests\test_mgemm_variable_width.py --extension-dir $fixed --device cuda:0 --output native-results\deepseek-v4\mgemm-new.json
$env:PYTHONPATH=(Resolve-Path $fixed).Path
& $py -m unittest discover -s tests -p 'test_native*.py'
```

`util/build_deepseek_state_fix.bat` rebuilds the separate fixed binary using the
existing compiler/CUDA installation and pinned environment; it does not install
or update dependencies.
