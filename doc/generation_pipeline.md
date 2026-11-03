# Optional generation-pipeline submission policies

Set these variables before starting the inference process:

```powershell
$env:EXL3_STOP_MASK_CACHE = "1"
$env:EXL3_ASYNC_DRAFT_SEED = "1"
```

Both are independent opt-ins. Unset or `0` preserves the previous behavior.
They work independently of `EXL3_NATIVE_PLAN` and do not change Torch thread
counts, OpenMP settings, GPU kernels, quantization, or CUDA graph policy.
For the validated EPYC GPU-inference configuration, retain the separate
`EXL3_OMP_BLOCKTIME_MS=0` setting. These are process environment variables;
no persistent TabbyAPI or system configuration change is required.

`EXL3_STOP_MASK_CACHE=1` reuses a job's device EOS/stop-token mask while its
minimum-length policy is invariant. The unchanged mask-building code handles
the first use and every cache miss. Crossing `min_new_tokens`, changing stop
IDs/device/stream, or entering a dynamic constraint uses the ordinary path.
Filters (including dormant filters), banned strings, checkpoints, healing,
unknown sampler implementations, and custom sampling steps are excluded.
Built-in samplers and `CustomSampler` stacks composed of built-in steps are
supported. Each job owns its mask; no GPU buffer is shared between jobs.
Requeue initializes a fresh cache. No logits, random draws, sampled tokens,
stop decisions, or MTP verification results are cached.

`EXL3_ASYNC_DRAFT_SEED=1` lets the existing DSpark drafter upload its pinned
CPU int64 seed without blocking host submission behind earlier GPU work.
It applies to the older DeepSeek V4 and V4.1 DSpark implementation. Pageable,
non-token, and device inputs retain the normal device-copy behavior. This is
not a global asynchronous-copy policy. The generator owns the pinned buffer
and keeps it unchanged until the final draft readback; that synchronization
and all confidence/acceptance decisions remain intact. Other draft
architectures are unchanged.

The policies live in `generator/stop_mask_cache.py` and `util/draft_seed.py`,
with small hooks at their existing preparation/upload sites. Sampling metadata
and draft input lifetime belong to the generation pipeline, so these helpers
are separate from native IR rather than extending IR's numerical/state scope.

Measured performance, model coverage, exact-parity checks, limitations, wheel
provenance and rollback details are recorded in
[the V4.1 investigation](deepseek_v41_ir_investigation.md#generation-pipeline-investigation-on-the-validated-openmp-baseline).
Raw results and reproducible benchmark scripts are under
`native-results/deepseek-v41/generation-pipeline/` and `util/`.

On the validated Windows EPYC/Blackwell/4090D system, enabling both policies
improved DeepSeek V4.1's matched MTP3 decode by about 3.6-5.0% in fresh
uninstrumented short-context comparisons, and about 4% at 8K and near 65K.
Prefill was essentially unchanged. These are workload-specific measurements,
not an expected gain for every request or machine.

For DeepSeek V4/V4.1 MTP, both opt-ins are useful in the measured workloads.
For Qwen, GLM, Nemotron and Gemma, only the stop-mask policy applies; its benefit
requires a minimum-length stop-token mask. Unmasked requests generally gain
nothing from that cache. No useful Devstral speedup was established, so leaving
the policies unset is appropriate there. Per-model evidence, including native
TP process variability and concurrent workloads, is in the investigation.
