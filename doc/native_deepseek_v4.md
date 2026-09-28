# DeepSeek V4 Flash: existing-IR integration

Investigation follow-up: the stock persistent-state blocker described below
is resolved. The [investigation report](deepseek_v4_state_investigation.md)
demonstrates a shared MGEMM tile-width bug and its narrow C++ fix. With the
separately rebuilt extension, repeated stock execution and the original
32-step stock/plan gate now match exactly, including persistent state.
The Python-only port results below describe the original extension; the
subsequent kernel fix is separate. The original integration is now validated
through batching/concurrency, DSpark acceptance/rejection and repair,
topology/reload, 64K context and alternating performance comparisons.
See [post-fix validation](deepseek_v4_validation.md) for the current results,
scope and reproducible commands. No release was created.

Checkpoint: `turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw` in the pinned
Tabby environment. The actual config specifies 43 trunk blocks, H=4, D=4096,
RMS epsilon 1e-6, HC epsilon 1e-6 and 20 Sinkhorn iterations. Attention has
64 query heads, shared 512-wide KV, 64 rotary dimensions, two initial sliding
layers followed by CSA (compression 4) and HCA (compression 128). There are
256 routed experts, top-6, one shared expert, and three initial hash layers.
The checkpoint contains `mtp.safetensors`; its compression schedule has three
additional sliding layers, despite `num_nextn_predict_layers=1`. DSpark drafts
five tokens, uses taps 40/41/42, and has a rank-256 Markov head.

## Architecture-to-IR map (before implementation)

| DeepSeek operation | Existing primitive | Bindings | Python boundary? | Reason |
| --- | --- | --- | --- | --- |
| Embedding / expand streams | none | embedding, H | yes | model-level input |
| Attention residual mix | HC_MIX | fn half [24,16384], base float [24], scale float [3], epsilons, iterations | no | identical stock HC kernel |
| Attention input norm | RMS_NORM | weight [4096], epsilon | no | ordinary weighted RMS norm |
| Query/latent KV preparation, RoPE, attention | none | attention projections and state | yes | stock DSV4Attention owns execution |
| Attention output projection | none | grouped output projections | yes | part of stock attention |
| Attention residual update | APPLY | post gates, combine matrix | no | x = post * y + combine-transpose * x |
| MLP residual mix and norm | HC_MIX, RMS_NORM | second site's HC and norm bundle | no | same operations as attention site |
| Hash or sqrt-softplus routing | ROUTE incompatible | token IDs, bias, scaling, router weights | yes | existing ROUTE is routing_std |
| Routed and shared experts, combination | MOE not independently usable here | native MoE handle, external selections | yes | current plan requires its own ROUTE before MOE; retain whole stock MLP |
| MLP residual update | APPLY | second site's gates/combination | no | identical stock kernel |
| Target tap export | none | stream mean, half cast/clamp | yes | DSpark protocol at block output |
| Final stream collapse / norm / head | none | HyperHead and head weights | yes | model-level output |
| DSpark input, KV updates, Markov/confidence sampling | none | projected target taps and private draft cache | yes | architecture/state ownership |

Each block carries an fp32 [B,Q,4,4096] residual. HC reduces it to fp16
[B,Q,4096], followed by weighted RMSNorm. Attention and MLP return float
outputs; APPLY mutates only the residual. Block-local plan scratch consists of
mix partials, gates, combination matrices and normalized activations. It holds
no attention/cache state. Both HC RMS and following RMS must have the same
epsilon because v1 stores one epsilon per site.

DSV4Attention owns query projections, normalized shared KV, positional state,
sliding rings, compressor buffers, CSA indexer keys/overlap snapshots and paged
compressed pools. This is not the GLM latent-KV protocol. DSV4State owns job
position/window/slot metadata. Rejection rewinds the position within the ring's
rollback capacity; old rows are overwritten on re-advance. Logical pool lengths
are position-derived. Stale pool capacity is not a valid parity range.

Model module `prepare_for_device` owns transfers between blocks. Native sites
bind only tensors resident on the block's actual device; visibility count and
physical GPU identity are irrelevant. Attention handles each job's recurrent
slot, including its existing batch graph/eager dispatch. No scheduler changes.

The inspected [85,40] GiB placement puts embedding on CPU, stream expansion and
layers 0..33 on logical CUDA0, then layers 34..42 and the output head on CUDA1.
This is recorded evidence, not a guard constraint. Hash layers gather six
expert IDs from tid2eid using the current token IDs. Other layers select top-6
using sqrt(softplus(logits)) with correction bias for selection only. Weights
normalize the unbiased affinities over the selected set and scale by 1.5.
Routed and shared experts use capped SwiGLU (limit 10), combine through the
stock BlockSparseMLP path, and produce float output before the second APPLY.

DSpark is a block drafter, not sequential GLM MTP. It embeds a seed/noise block,
runs three transformer blocks, and uses the shared target head plus Markov bias
and confidence-based draft length. Accepted target taps update draft KV;
verification/rejection and cache ownership remain in the generator. Exported
target taps must run after the native residual finish using the stock code.

Load/unload clears native bindings. Specialization must distinguish device,
B, Q and dtype, as well as bound tensor identities and site configuration.
Reject conversion/capture, TP, LoRA, prefill and unsupported shapes before
begin. Never catch a stage failure and retry stock after mutation.

## Future IR opportunities (not implemented)

The separate [future-extension note](deepseek_v4_future_ir.md) details inputs,
outputs, boundaries, wider applicability and expected benefits. Both proposals
are optional optimizations, not correctness requirements.

OPTIONAL PERFORMANCE/BOUNDARY OPTIMIZATION: parameterized router selection.
Inputs: normalized activation, router matrix, score correction bias, optional
token-to-expert table, token IDs, top-k and routed scaling. Outputs: selected
expert indices and normalized/scaled routing weights. `ROUTE` currently binds
only routing_std; neither HC/GR mix, APPLY, RMS_NORM nor MOE expresses sqrt
softplus scoring or token-indexed selection. An architecture-neutral
`ROUTE_CONFIGURED` could select existing routing algorithms at construction
and let MOE consume the results, eliminating the Python MLP boundary when the
expert handle supports the configuration. Bias-corrected and hash routers
occur beyond DeepSeek V4. Benefit would be fewer dispatch/boundary calls;
it needs measurement and is not required for correctness.

OPTIONAL PERFORMANCE/BOUNDARY OPTIMIZATION: `REDUCE_STREAMS` taking residual
streams and reduction/cast/clamp metadata and returning an exported hidden
state could remove a mean/cast/clamp sequence at target taps or heads. Existing
mixes compute learned gates, not an arithmetic mean. It is broadly useful for
multi-stream architectures, but export remains a Python responsibility here.

No new primitive, C++ wrapper, scheduler/cache change or dependency change is
required for the safe HC/norm/residual boundary.

## Initial validation findings (before the stock fix)

The actual loaded checkpoint confirms the HC binding shapes above and BF16
norm weights. The isolated existing generic extension provides BC_BlockPlan;
no C++ source or primitive changes are needed. The architecture registers its
block emitter at model construction, including DSpark blocks. Selection uses
EXL3_NATIVE_PLAN alone. Qwen/GLM reference emitters and guards are unchanged.

113 CPU native guard tests passed, including nine new DeepSeek tests covering
registration dispatch, target exports, lifecycle clearing and error propagation.
The first harness run needed inference_mode around direct cache allocation;
the second exposed a guard assumption about a nonexistent MLP out_dtype field.
BlockSparseMLP guarantees float output by its class contract; this was corrected.
Both failed runs are retained.

The first real B1/Q1 stock/plan run executed exactly 1,376 begin/middle/finish
calls (43 blocks x 32 steps), built 43 BC_BlockPlans, and had zero fallbacks.
Every block output and logit was byte-exact. Persistent state was NOT exact:
CSA indexer buffers in layers 36,38,40,42 differed on the second device. The
gate correctly failed rather than accepting output parity alone.

A separate cold/warm stock/stock/plan reproduction confirms this is also a
stock-versus-stock instability. At position 301, layer 36's indexer KV buffer
differs in exactly row 40 (= 300 modulo 260), all 256 columns. Stock/stock
maximum absolute difference was 7.6005859375; stock/plan was 10.09765625.
These are initialized live rows, not uninitialized capacity, and cannot be
excluded or relaxed to a numerical tolerance. Longer decode also diverges in
indexer overlap snapshots and pool keys. All short logits remain identical
because the indexer pool has not yet reached the sparse top-k threshold.

This unrelated attention issue is being isolated without changing attention,
cache kernels, dependencies or the generic IR. Artifacts are under
`native-results/deepseek-v4/diagnose-state`, including raw state samples and
stock/stock comparisons. Default-path state correctness remains a failed gate;
no release or performance claim is authorized by these results.

Disabling the existing attention graph option (`EXL3_BC_DSA=0`) reproduces the
same stock/stock indexer-state problem. It is not established as a graph-only
issue. See [the separate blocker report](deepseek_v4_stock_state_blocker.md).

The isolated actual-weight native-stage gate PASSED 414 cases: all 43 trunk
and three DSpark blocks, each at B/Q = 1/1, 2/1, 3/1, 4/1, 1/5, 2/5, 1/6,
2/6 and 3/4. Begin and middle activations, both residual updates and final
residuals were byte-exact against stock HC/RMSNorm/APPLY. Invalid boundary
shapes raised without changing residuals. The identical case set passed under
both CUDA_VISIBLE_DEVICES=2,0 and 2,0,1. Peak Torch allocation on unused logical
CUDA2 was zero. These are stateless site/topology tests, not full-model topology
or MTP certification.

The existing tested extension was reused, with SHA256
`afbb02a57e61e5f2731b7026cd3c3a211c609b8ac2a6ed7d8a87a136d080d42a`.
No C++ source, BC_BlockPlan contract, primitive, package or dependency changed;
a rebuild is unnecessary for this Python-only port. No release was created.

| Gate | Status |
| --- | --- |
| Actual config / loaded trunk inspection and IR map | complete |
| Python emitter, registration, guard, lifecycle and target exports | implemented |
| CPU guard/regression suite | 113 pass |
| Actual-weight HC stages, trunk and draft | 414 cases pass in each visibility mapping |
| Full-model B1/Q1 logits and block outputs | exact for 32 steps; 1,376 native calls, zero fallback |
| Full-model persistent state | original FAIL resolved by shared MGEMM fix; 32-step gate now exact |
| Full-model batches, dynamic arrivals, cancellation, page reuse, sampling | not reached after short gate failure |
| End-to-end DSpark acceptance/rejection and repair | not reached |
| Full-model reload / topology / long context | not reached |
| Alternating performance benchmark | not run; correctness gate failed |

The guard permits B1..4, Q1..6 with B*Q <=12 and the validated HC/norm contract.
Larger shapes, prefill, TP, capture/conversion and incompatible site parameters
fall back before begin. Loaded weights are immutable; module load/unload is the
supported rebinding mechanism. The helper also rebuilds on tensor identity or
site parameter changes. Production dispatch has no counters or state hashing.

Reproduce the CPU suite using the pinned Python with
`-m unittest discover -s tests -p test_native*.py`. Run
`util/check_deepseek_plan_sites.py --output <new-file>` for the isolated gate.
`util/validate_deepseek_plan.py --output <new-directory> --diagnose --steps 2`
preserves the stock/stock/plan blocker. Its generator/long/benchmark phases
are validation scaffolding for use only after the short state gate passes;
they have not been certified in this run. All failed outputs were retained.

The IR experiment therefore demonstrates that the entire stateless two-site
HC/norm/residual portion is expressible unchanged, with a 12-line emitter and
Python eligibility/registration. Routing and attention remain intentional
boundaries. Those initial results did not establish end-to-end state correctness
or speed. The subsequent [completed validation](deepseek_v4_validation.md)
now supplies short/64K state, concurrency, DSpark, topology and timing results.
