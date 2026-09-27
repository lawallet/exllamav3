# Native block IR v1 measurements

Windows/WDDM, existing Python 3.13 / Torch 2.9.0+cu128 environment, unchanged
model quantizations and dependencies. Qwen uses the Pro 6000; GLM retains the
35/10 Pro 6000 / 4090D target split. MTP draft behavior is unchanged.

Each row is the median of three warmed runs per backend, alternating order on
the same loaded model. B1 uses 1000 output tokens. Concurrent cohorts use output
lengths 384/512/640/768 truncated to the indicated batch. Their rates are
aggregate delivered output tokens per second; per-job rates, timings, hashes,
acceptance totals and call counts are retained in the validation archive.
Instrumentation runs separately from timed cohorts. No samples are discarded.

| Model | Mode | Jobs | Bespoke t/s | Plan t/s | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | Ordinary | 1 | 46.67 | 46.76 | +0.19% |
| Qwen | Ordinary | 2 | 106.20 | 106.32 | +0.11% |
| Qwen | Ordinary | 3 | 133.34 | 132.85 | -0.37% |
| Qwen | Ordinary | 4 | 155.22 | 155.12 | -0.07% |
| Qwen | MTP | 1 | 102.74 | 102.59 | -0.15% |
| Qwen | MTP | 2 | 96.81 | 96.75 | -0.06% |
| Qwen | MTP | 3 | 95.98 | 99.32 | +3.48% |
| GLM | Ordinary | 1 | 47.32 | 47.20 | -0.25% |
| GLM | Ordinary | 2 | 48.76 | 48.51 | -0.51% |
| GLM | Ordinary | 3 | 60.74 | 60.48 | -0.42% |
| GLM | Ordinary | 4 | 68.27 | 68.20 | -0.09% |
| GLM | MTP | 1 | 65.06 | 65.01 | -0.08% |
| GLM | MTP | 2 | 71.50 | 71.96 | +0.64% |
| GLM | MTP | 3 | 50.37 | 49.97 | -0.78% |

All paired completions and speculative acceptance totals match. The largest
median loss is 0.79%, below the 3% investigation gate. Qwen B3 MTP varies around
95–100 t/s across samples; its positive median difference is not evidence of a
stable speedup. Short topology runs are correctness/profile checks and are not
included in this performance table.

## Host calls and topology

The ordinary diagnostic windows use 64 Qwen tokens and 32 GLM tokens. Bespoke
and generic have identical module-call and native-call counts within each
architecture, under both `CUDA_VISIBLE_DEVICES=2,0` and `2,0,1`.

| Model | Module calls | Native calls | Generic block invocations | Decode fallbacks |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 6656 | 11904 | 3072 | 0 |
| GLM | 4544 | 9696 | 1440 | 0 |

Qwen retains two block-stage crossings and GLM retains three. Native-call totals
also include existing attention/recurrent/MLP extension calls. Prefill and
unsupported MTP/concurrent shapes keep their established fallback behavior;
the zero counts above apply only to these ordinary diagnostic windows.
Logical transfers, eligibility, outputs and state records are identical across
the visibility mappings. Unused logical CUDA2 has zero peak Torch allocation.

`performance-summary.json` retains all direct samples and observational GPU
utilization/power medians from the telemetry time windows. For example, GLM B2
MTP Pro 6000 medians were 40% / 163.2 W bespoke and 39% / 161.1 W generic.
Telemetry includes the desktop GPU's unrelated activity and is observational,
not a measurement of energy per generated token.

See [the design and engineering report](native_block_ir.md) for the exact-state
matrix, retained adapter failures, restrictions and reproduction tools.
