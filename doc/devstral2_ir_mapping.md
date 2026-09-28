# Devstral 2 architecture-to-IR map

The actual checkpoint config selects `Ministral3ForCausalLM`, implemented by
Ministral3Model/Mistral3Model, not the generic Mistral class. It has 88 dense
blocks, H=12288, SwiGLU intermediate=28672, 96 query heads, eight KV heads,
head dimension 128, epsilon 1e-5 and vocabulary 131072. NEOX-style YaRN RoPE
uses theta 1e6, factor 64, original context 4096, beta fast/slow 4/1.
All attention is global paged KV; there is no sliding window or recurrent
state class. No checkpoint-native draft/MTP component is configured.

## Mapping decided before implementation

| Operation | Actual tensors/semantics | Existing IR | Boundary |
|---|---|---|---|
| Input norm | FP32 residual [B,Q,12288] to FP16 scratch | v2 RMS_NORM | Native |
| Attention | Separate quantized Q/K/V and O, FP16 norm input, FP32 output | Neither | Python |
| Attention residual and MLP norm | Fused r += attention; FP16 RMSNorm(r) | Missing in v2 | See below |
| Dense SwiGLU | FP16 norm input, FP32 output | Neither | Python |
| MLP residual | FP32 r += MLP output | Missing in v2 | See below |
| Hidden export | FP16 clamped residual, optional | Neither | Python |
| Final model norm/head | RMSNorm to FP16 then separate untied head | Outside block | Python |

There are no HC/GR streams, sandwich/post norms, residual scalars or layer
scales. V1 does not fit. Existing v2 NORM_ADD computes r += norm(y), which
cannot express norm(r += y). Substituting it would change the model.

**Complete useful plumbing does not fit unchanged v2.** Leaving residual
work in Python is mathematically possible, but the existing three nonempty
native-stage contract then cannot express the finish boundary, and splitting
stock's fused pre-norm residual update would add a launch. A one-norm native
adapter would preserve very little useful work and still need awkward stage
padding. Instead add only two architecture-neutral instructions:

- RES_ADD_NORM: existing `rms_norm_res_in` kernel; explicit source and scratch
  destination, norm binding and resolved residual-role operand. Preserve stock
  residual rounding and reduction exactly.
- RES_ADD: existing ATen in-place addition of source into the explicit FP32
  residual destination; no norm binding, encoded as -1.

Both could serve conventional Llama/Mistral-style dense blocks. They are
needed for a complete native three-stage plan, not for stock correctness.
No new math kernel, architecture-specific C++, graph model or v1 change is
needed. Preserve ABI 1 for existing instructions and add a separate residual
operations capability flag so older v2 extensions reject Devstral safely.
All boundaries/weights are checked before any mutation, as for Gemma.

Attention/cache/projections, MLP, exports and final model norm/head remain
Python-owned. Prefill, TP, LoRA and capture/conversion modes remain stock.
Target decode shapes are B1..4, Q1..6 with B*Q<=12. Bound norm identity and
norm configuration enter specialization keys; load/unload clear plans.

The first loaded inspection exposed a harness assumption (`None.__name__`
for the absent recurrent state class), not a model failure. Its log is
retained; the corrected inspection records null for that field.
