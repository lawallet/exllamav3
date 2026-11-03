# Native DeepSeek V4.1 projection graphs

Enable this optional native IR stage before importing ExLlamaV3:

```powershell
$env:EXL3_NATIVE_PLAN = "1"
$env:EXL3_NATIVE_DSV41_PROJECTIONS = "1"
```

CUDA graphs must be enabled (`EXL3_GRAPHS` unset or `1`). The projection flag
defaults to off. It can be combined with the OpenMP wait policy, stop-mask
cache and asynchronous draft-seed upload. These are process settings; no
TabbyAPI configuration changes are required.

The stage submits the existing query down-projection, FP16 RMS normalization,
query up-projection and independent KV projection through one native plan and
CUDA graph. It uses the existing quantized GEMM dispatcher, including its
selected int8 GEMV policy, without changing kernels or accumulation precision.
Query rotation and weighted KV normalization/rotation remain in upstream code.
Shared KV pools, latent-index generation, candidate selection, cache updates,
MTP verification/rewind and cross-device ownership are outside the graph.

Initial eligibility is deliberately restricted to:

- `DeepseekV41ForCausalLM`, without the older per-head query norm or TP.
- The existing 5120→1280→32768 query and 5120→512 KV topology.
- K3 mul1 EXL3 weights, no padding, bias, LoRA, scaling or softcap.
- Contiguous FP16 batch-one inputs with query lengths 1–5.
- CUDA SM89 or SM120, with the matching projection-plan extension ABI.

Prefill, unsupported shapes/weights/devices and capture or reconstruction
overrides fall back to the original projection path. Named query-norm state
exports also fall back. Existing architecture and native IR guards remain in
effect. This flag does not enable a new projection path for other models.

Each eligible attention module caches a plan per query length, device and CUDA
stream. The first invocation warms the existing dispatcher before capture;
subsequent invocations patch input pointers and replay. The plan owns its
intermediates and scratch, checks fixed bindings and stream identity on every
run, and is cleared when the attention module unloads. Returned tensors are
borrowed until the next invocation of that specialization; the existing
attention consumers finish using them on the same stream before then. Do not
retain these buffers as new persistent attention or speculative state.

On the validated V4.1 configuration, the extra steady allocation is about
34.33 MiB on the attention GPU. Clean matched MTP3 measurements showed roughly
4–5% faster decode, with exact forward outputs and token/MTP parity. The
optimization does not target prefill. See
[the investigation report](deepseek_v41_ir_investigation.md) for workloads,
ranges, outliers, cross-model validation and release provenance.
