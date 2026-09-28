# Resolved block-plan v2 design

Written before implementation. The preserved `gemma4_ir_assessment.md` is the
requirements/evidence baseline; its negative v1 results remain valid.

V1's necessary assumptions are static specialization, bounded small rows,
same-device operations per stage, explicit Python boundaries, owned scratch,
and serialized invocation. Four residual channels, compulsory mixing and
HC-owned norm operands are historical assumptions, not necessary ones.

## Smallest additive contract

Keep `BC_BlockPlan` and its old constructor, instruction verifier and executor
unchanged. Add `BC_BlockPlanV2`, a generic resolved-value executor, and explicit
`native_block_plan_v2_abi=1`. Python requires both; old extensions fall back
before construction. No v1 opcode, constructor or emitter changes. Separate
execution avoids adding a per-operation or per-stage v2 branch to v1.

Values are a fixed constructor vector of `(role, dtype)` descriptors. Roles:
0 writable incoming residual, 1 middle-stage Python boundary, 2 finish-stage
Python boundary, -1 owned scratch. Dtypes: 0 FP16, 1 FP32. Each value has the
specialized `[B,Q,D]` logical shape and contiguous `[B*Q,D]` kernel view.
There is one FP32 residual; at most one external value per boundary. Scratch
is allocated once. Each instruction resolves indices into direct `Value*`
and norm-binding pointers; hot execution has no string/dictionary lookup,
callback, graph traversal or architecture decisions.

Instructions are `(opcode, source_slot, destination_slot, norm_binding)`.
`RMS_NORM=3` reads any initialized same-stage input and writes scratch.
`NORM_ADD=6` reads an initialized activation and accumulates into the residual.
Norm bindings own contiguous FP16/BF16 `[D]` weights and finite epsilon,
constant bias and scale. Groups=1, span_heads=false. Input/output dtypes are
explicit slot contracts. Use the existing `rms_norm` kernel without algebraic
rewrites. `NORM_ADD` sets its existing `add_residual=true`: this is exactly
stock's fused post-norm/accumulation, not a rounded temporary plus a second add.
It merits a separate opcode because the writable output/alias contract differs.
Kernel inspection rules out FP32 norm weights: unlike the v1 HC constructor's
permissive check, the actual RMS kernel has no FP32-weight specialization.
V2 rejects these at construction, before any residual update.

Each stage has an explicit returned slot (-1 for finish). The verifier tracks
initialized values anew per stage: residual and that stage's boundary are live;
scratch must be written before reading. Cross-stage hidden scratch dependencies
are deliberately excluded. Scratch is reusable only after its Python consumer
returns; concurrent calls/streams on one plan are unsupported as in v1.

Validate the entire constructor before scratch allocation. At stage entry,
validate all external operands (shape, dtype, device, contiguity and aliasing)
before launching any mutating operation. Reject residual aliasing boundary,
weights or internal scratch. Never retry stock after native entry. No dynamic
DAG or new attention, routing, MLP or cache operation is introduced.

## Gemma lowering

Slots: residual FP32, attention boundary FP16, MLP boundary FP32, two FP16
scratch activations. Begin: norm residual into attention scratch. Middle:
norm-add attention output into residual, norm residual into MLP scratch.
Finish: norm-add MLP output into residual. Four checkpoint norm bindings.
Stock attention and GatedMLP remain Python; preserve `params['residual']`,
optional unscaled state export, then final learned scalar multiplication.
No SCALE opcode: Python multiplication is already exact and a small boundary.

Register only Gemma text blocks, using EXL3_NATIVE_PLAN. Eligibility checks
actual module structure, tensor layout/dtype/placement, bounded B/Q and modes;
no device-name/count/ordinal assumption. Cache keys cover B/Q/device/dtype;
signatures cover norm tensor identity/version/layout, norm configuration,
module/configuration identities and layer scalar. Clear on load/unload.

## Compatibility and validation order

Separate class is chosen over rewriting v1 into slots: less compatibility risk,
no protected path overhead, while removing HC/GR coupling for new plans. This
is an additive versioned generic API, not a Gemma backend. Existing v1 sites
remain supported, not migrated merely for architectural uniformity.

Build an isolated v2 extension. First exact generic/actual-weight norm tests,
invalid boundary/alias tests and all 60 Gemma sites; then existing architecture
GPU regressions and guards. Only then certify Gemma full decode, valid cache
state, concurrency, reload, topology, long context, and warmed alternating
performance. Preserve every failed gate and do not run later gates past it.
