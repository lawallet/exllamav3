# Laguna-S-2.1 architecture-to-IR decision

Inspected canonical 4.00bpw checkpoint and all 48 loaded blocks with the pinned
Tabby runtime and the protected Devstral extension. Evidence: native-results/laguna/inspection.json.
No product changes preceded this inspection.

48 blocks, hidden 3072: layer 0 dense SwiGLU (12288 intermediate), layers 1..47
MoE (256 routed experts, top-10, intermediate 1024, one shared expert of width
1024). Global attention at layers divisible by four (48 Q heads); remaining
36 layers use SlidingAttention (72 Q heads, window 512). All use 8 KV heads,
head dimension 128, separate Q/K/V projections, Q/K RMSNorm, per-head softplus
output gate. NEOX RoPE: global YaRN factor128/theta500000, partial rotation0.5;
sliding default theta10000, full rotation. Attention owns these operations.
Loaded block norm weights are BF16 [3072], epsilon1e-6, bias0/scale1.
Residual and sublayer outputs are FP32; norm outputs are FP16. No HC, GR,
postnorm, sandwich norm or residual/layer scalars. SWAState owns sliding KV;
global layers own paged KV. Final norm emits FP16 to the vocabulary head.

| Operation | Input -> output | Owner / chosen IR | Rationale |
|---|---|---|---|
| Attention input RMSNorm | FP32 residual -> FP16 | v2 RMS_NORM | Exact existing kernel |
| QKV, QK norm, RoPE, attention, softplus gate, O | FP16 -> FP32 | Python attention boundary | Includes sliding/global cache ownership |
| Attention residual + MLP norm | FP32 residual/output -> updated FP32 residual + FP16 | v2 RES_ADD_NORM | Exact stock fused residual-in kernel |
| Dense SwiGLU (layer0) | FP16 -> FP32 | Python MLP boundary | Existing fused implementation |
| Router (47 sparse layers) | FP16 -> int64 IDs/FP16 weights | Python MoE boundary | Existing deterministic native router, see below |
| Routed + shared experts | FP16 input and IDs/weights -> FP32 | Python MoE boundary | Existing BC_BlockSparseMLP.run_bszN merges shared expert |
| MoE residual addition | FP32 sublayer/residual -> FP32 residual | v2 RES_ADD | Exact stock in-place add |
| Exports, final norm/head | FP32 -> clamped FP16 exports/logits | Existing Python ownership | No new semantics |

V1 RMS_NORM/ROUTE/MOE are coupled to HC/GR sites. Laguna has neither.
NORM_ADD is postnorm-add, not the residual-in norm needed here. No residual-mix
primitive applies. V2's existing operands and three operations fit unchanged.

## Routing decision: outcome B

V1 ROUTE calls routing_std (softmax, no correction). Laguna calls
routing_ds3_nogroup with SIGMOID, correction bias for selection only, normalization
over selected raw scores and routed scaling. IDs are int64, weights FP16. The
configured scale2.5 is multiplied by128 to compensate for the routed expert
intermediate downscaling (interm_div128); the effective loaded route scale is320.
Replacing this with v1 ROUTE would be incorrect. Existing MOE's run_bszN accepts
the selections, but v1's site model and router opcode do not express Laguna.

For decode rows<=12 the stock MoE already invokes deterministic native routing
and BC_BlockSparseMLP.run_bszN, with shared-expert execution/combine handled
inside that native object. It does not Python-loop/sort/regroup experts in this
regime. Therefore a major remaining Python routing bottleneck has not been
demonstrated: retain the boundary and benchmark the existing v2 plumbing.
No configured-routing opcode, integer value slots, ABI or kernel changes are
justified in this iteration. DeepSeek's sqrt(softplus)/hash routing remains
another possible future consumer, not grounds alone for expansion.

The checkpoint exposes text only, no native MTP/default drafter. A separate
DFlashLaguna implementation exists for separately supplied compatible drafter
checkpoints; that does not make this target checkpoint a native MTP model.
