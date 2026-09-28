# Optional generic IR extensions suggested by the DeepSeek port

Neither proposal is required for correctness. Neither is implemented in this
port. The existing six primitives cover the HC/norm/residual boundary exactly.

## Configurable routing and externally supplied selections

**OPTIONAL PERFORMANCE/BOUNDARY OPTIMIZATION.** DeepSeek's first three blocks
gather expert IDs from a token-to-expert table. Later blocks select top-k using
`sqrt(softplus(logits))` plus a correction bias for selection, then normalize
the unbiased selected affinities and apply the routed scale. The current
`ROUTE` implements `routing_std`, and `MOE` expects that plan-owned route.
HC/GR mix, APPLY and RMS_NORM cannot express the selection semantics.

An architecture-neutral `ROUTE_CONFIGURED` could bind the router weights,
correction bias, optional token-to-expert table, scoring mode, top-k and scale.
Its dynamic inputs would be normalized activations and, for hash routing,
token IDs. Outputs would be expert indices and weights consumed by `MOE`.
The design must explicitly accommodate dynamic token IDs; binding them as
immutable weights would be incorrect. Alternatively, an external-selection
input to MOE could preserve Python routing while moving expert invocation
into the plan.

This could remove one Python MLP boundary per block plus its router/expert
dispatches. Bias-corrected selection and token-indexed routing are useful
beyond DeepSeek. Expected benefit is less dispatch overhead and less bespoke
integration code; the size of a throughput improvement is unmeasured. The
current stock MLP already owns optimized expert execution, so no arithmetic
speedup is assumed.

## Stream reduction for exported hidden states

**OPTIONAL PERFORMANCE/BOUNDARY OPTIMIZATION.** Target taps and the final
hyperconnection head reduce the residual streams with an arithmetic mean,
then cast/clamp as required by the consumer. A `REDUCE_STREAMS` operation
could take an FP32 `[B,Q,H,D]` residual and bound reduction/cast/clamp parameters
and produce `[B,Q,D]` hidden states. The existing mixes use learned gates;
they are not an unweighted mean.

The proposed output would need an explicit export interface at the Python
boundary. It must not move DSpark cache ownership or speculative repair into
the plan. It could remove the mean/cast/clamp dispatch sequence at three
target taps per verification, and potentially serve other multi-stream
architectures. Its benefit would be fewer boundary operations; performance
and implementation savings require measurement. Python exports are correct
and remain supported without this extension.
