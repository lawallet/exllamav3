# DeepSeek V4 stock indexer-state reproducibility blocker

Resolved by the subsequent [stock investigation](deepseek_v4_state_investigation.md).
The shared MGEMM kernel skipped narrow matrices under a wider launch tile,
then transformed stale output scratch. A per-matrix tile fallback fixes the
cause. Repeated stock runs with and without graphs and the original 32-step
stock/plan persistent-state gate now pass exactly. The observations below are
preserved as the pre-fix record. Subsequent broader results are recorded in
[the completed integration validation](deepseek_v4_validation.md).

Observed while validating the Python-only native-plan port on 2026-09-27.
No attention, cache or routing implementation was changed by the port.

Environment: pinned D:\tabbyAPI\venv Python, existing native-block-ir-v1
extension (SHA256 afbb02a57e61e5f2731b7026cd3c3a211c609b8ac2a6ed7d8a87a136d080d42a),
actual turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw checkpoint,
CUDA_VISIBLE_DEVICES=2,0,1 and [85,40] GiB loading budgets.

Reproducer:

```powershell
$env:CUDA_VISIBLE_DEVICES='2,0,1'
& D:\tabbyAPI\venv\Scripts\python.exe util\validate_deepseek_plan.py --output native-results\deepseek-v4\new-diagnostic --diagnose --steps 2
```

The harness creates a fresh recurrent state, feeds an identical 300-token
prompt in 256/44 chunks, then performs identical B1/Q1 decode. It repeats with
stock before running the plan. State allocation clears the slot between runs.
The repeated stock inputs, block outputs and logits match; the CSA indexer
buffers do not. Layers 36,38,40,42 are affected on logical CUDA1. Layer 36's
first decode writes row 40 of its 260-row projected-KV ring (300 % 260). All
256 values in that live row differ between stock runs, with maximum absolute
difference 7.6005859375. No nonfinite values are present. Stock/plan differs
by 10.09765625 in the same row. Subsequent steps propagate differences to
indexer overlap snapshots and valid pooled keys.

The stock graph-disabled reproduction (EXL3_BC_DSA=0) also fails in these
buffers. Thus disabling attention graphs is not a workaround. The source of
the instability has not been established; it must not be described as an IR
math discrepancy, harmless rounding, or uninitialized unused storage.

Artifacts:

* `native-results/deepseek-v4/diagnose-state/cold-stock.json`
* `native-results/deepseek-v4/diagnose-state/b1-q1-0.json`
* `native-results/deepseek-v4/diagnose-state/b1-q1-1.json`
* `native-results/deepseek-v4/diagnose-state/state-samples.pt`
* `native-results/deepseek-v4/diagnose-state/state-differences.json`
* `native-results/deepseek-v4/short-third` (32-step output/boundary parity)
* `native-results/deepseek-v4/short-eager` (graph-disabled reproduction)

Short logits can match despite this problem: the compressed pool is below the
512-entry CSA selection threshold, so different indexer keys do not yet affect
top-k selection. Persistent-state parity is consequently a necessary gate.

Resolution requires a separate investigation of the existing stock attention
projection/indexer state path. Per the requested scope, this integration does
not modify that path. Full-model batching, MTP, topology, long-context and
performance gates remain pending; no release is justified.
