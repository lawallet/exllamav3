# Devstral 2 native-plan engineering report

## Architecture and contract

Checkpoint: `D:\text-generation-webui\user_data\models\turboderp_Devstral-2-123B-Instruct-2512-exl3_4.00bpw`.
No model files were copied. See [the pre-implementation mapping](devstral2_ir_mapping.md).
Loaded inspection is retained in `native-results/devstral2/inspection-r2.json`.
The text-only Ministral3 model has 88 identical dense blocks, H12288, SiLU gated
MLP28672, 96 Q heads/eight KV heads with D128, separate quantized Q/K/V/O,
global paged KV and NEOX YaRN RoPE. Norm weights load as FP16 [12288], epsilon
1e-5, bias zero, scale one. Embeddings emit FP32; both sublayer outputs and
the single residual are FP32. Norm activations are FP16. There are no Q/K
norms, HC/GR mixing, sandwich norms or learned layer/residual scalars.
The final model norm produces FP16 for the separate untied head.

**V2's operand model generalized, but its initial operation set was insufficient.**
Devstral requires `norm(residual += attention)`, whereas Gemma uses
`residual += norm(attention)`. V1's four-stream contract does not fit.
Two additive generic instructions reuse existing kernels: RES_ADD_NORM (7)
calls `rms_norm_res_in` exactly; RES_ADD (8) calls ATen's in-place add.
No math kernel was changed. A separate `native_block_plan_v2_residual_ops=1`
capability protects these instructions while ABI1 and Gemma's existing
instructions remain valid. V1 constructors, opcodes, executor and emitters
are unchanged. The prior v2 binary safely rejects Devstral before operand
access; it continues to support Gemma.

The static three-stage emitter does input RMSNorm, fused attention residual
update/MLP RMSNorm, then MLP residual addition. Attention, RoPE, paged KV,
projections, dense MLP and optional clamped hidden export remain Python-owned.
There is no Devstral-specific C++ backend or environment flag. Registration
is on Ministral3 text models; structural eligibility derives devices from
loaded tensors/modules. Selection uses `EXL3_NATIVE_PLAN=1`.

Supported decode is B1..4/Q1..6 with B*Q<=12. Prefill, TP, LoRA and conversion/
capture modes remain stock. Bound weight identity/storage/configuration
changes rebuild plans; load/unload clear bindings. All boundary and norm
preflight precedes mutation; native failures propagate without stock retry.

There is no recurrent-state class, recurrent slot state, checkpoint-native
MTP or default drafter. The generator has optional external-draft/ngram
mechanisms, which are not native components of this checkpoint and are not
certified here. Ordinary generation is tested; no MTP campaign is invented.

## Development environment and evidence

Pinned Python `D:\tabbyAPI\venv\Scripts\python.exe`, Python3.13.12,
Torch2.9.0+cu128, CUDA12.8, Windows amd64. No runtime dependency changes or
production installation. Isolated build:
`native-results/devstral2/extension/exllamav3_ext.cp313-win_amd64.pyd`.
SHA256: `b89a859f85765405f77e517f6faed02456cd23a46e48226258ba27c5dd1d6fd3`.
All new evidence is under `native-results/devstral2`; prior checkpoints,
Gemma/v1/DeepSeek evidence and the separate MGEMM investigation remain intact.

Initial inspection assumed a recurrent class existed and failed while reading
its name, after successful loading. The corrected inspection records null.
No stock correctness change resulted. PowerShell represented Triton compiler
remarks on stderr as NativeCommandError records; direct Python campaign
drivers preserve actual subprocess exit codes. Compiler remarks are not
inference failures.

## Correctness gates

- 127 native guard tests passed.
- Architecture-independent residual-op tests passed on two GPUs, FP16/BF16
  weights, nondefault bias/scale, exact fused outputs and residuals, invalid
  instruction/boundary/alias/device rejection and changed-binding safety.
- Actual-weight sites: all 88 blocks times six shapes = 528 exact cases.
  Begin/middle/finish outputs and mutations match stock, and invalid boundaries
  leave the residual unchanged.
- Short primary visibility `2,0,1`: 32 steps each for B1/B2/B3/B4 Q1, B1 Q5,
  B2 Q6. Each supported shape constructs 88 plans and executes 2816 calls per
  stage with zero unexpected fallback. Outputs, every block and device-transfer
  boundary, hidden exports and logically initialized paged KV compare exactly.
- Q7 uses real stock fallback. A stock-valid but unsupported norm configuration
  uses one stock block and 87 native blocks with exact parity.
- Weight identity and mode/config guards passed. Unload/reload preserves
  placement and exact outputs/state/counts after clearing harness-held bindings
  and the process-wide stock workspace cache, following the Gemma methodology.
- The extra logical GPU has zero allocated/reserved/peak Torch memory.

## Generator, topology and long context

`campaign/generator-201` passed all seven ordinary cases: simultaneous B1..4,
staggered arrival/replacement, live cancellation/replacement and seeded B2
sampling. Context240 plus 48 output tokens crosses the 256-token page boundary.
Outputs, logits, every block boundary and initialized live paged KV matched
exactly. Expected long-query/layer-prefill fallback was counted separately;
ordinary decode had zero unexpected fallback. Simultaneous cases execute
4224 calls per stage; staggered and cancellation cases execute 6336 and 4488.

`campaign/reuse-201` repeats stagger/cancel with a 1536-token cache and proves
two physical page reuses in each case. There is no recurrent slot to reuse.
Both runs compare exact live state; the cancellation case confirms an actual
live cancellation, not merely an empty queue operation.

Visibility `2,0,1` and `2,0` use the same 79/9 text-block split and match all
short outputs, cache state, transfers, placements and construction/invocation/
fallback counts exactly. Both unload/reload checks pass. The extra visible
GPU under `2,0,1` receives zero Torch allocation. `0,2` with budgets16/70
produces a 21/67 split and passes stock/plan parity for all short shapes and
fallback checks. Cross-GPU-family stock bitwise equality is not claimed.

`campaign/long-8192` passes B1 Q1 and Q5 for 16 steps after 8192 input tokens.
Each constructs 88 plans and executes 1408 calls per stage with zero fallback.
Exact outputs, hidden exports, block/transfer boundaries and initialized KV
are checked; the Q5 run reaches position8272. It also passes configuration
fallback. The unused third GPU remains at zero allocation. The advertised
262144 context is not certified.

## Implementation cost

The emitter is eight physical lines. Guard/registration is 69 lines plus two
architecture-hook lines. Generic C++ changes are 21 added/seven removed lines
(14 net, including binding/header changes); the value model is unchanged.
Generic Python adds two opcode/capability lines, two cache-lifecycle lines
and emitter separation whitespace. The validation cache clearer gains one
entry. V1 C++ and Qwen/GLM/DeepSeek/Gemma architecture implementations and
emitters are unchanged.

This is much less generic implementation than Gemma's original v2 creation.
It resembles DeepSeek's small emitter/guard integration, plus two narrowly
scoped kernel exposures. Validation remains the dominant effort. No unrelated
stock math/state bug was found. No routing, MoE, fusion, TP, scheduler/cache
rewrite or compiler expansion was introduced. Further architecture work can
reuse pre-norm and post-norm residual operations; other features remain out
of scope.

## Devstral performance

Five warmed alternating pairs per batch, context512, 256 timed output steps,
eight warm steps, identical checkpoint/placement/sampling. Instrumentation and
state hashing are excluded from timing. All samples are retained in
`campaign/performance/samples.json`; B4 is aggregate tokens/second.

| Batch | Stock median t/s | Plan median t/s | Median paired change |
|---|---:|---:|---:|
| B1 | 13.0452 | 13.0566 | +0.1107% |
| B4 | 51.1756 | 51.1663 | -0.0094% |

| Batch/pair | Stock t/s | Plan t/s | Paired change |
|---|---:|---:|---:|
| B1/0 | 13.2661 | 13.1277 | -1.0428% |
| B1/1 | 13.0795 | 13.0566 | -0.1749% |
| B1/2 | 12.9998 | 13.0564 | +0.4360% |
| B1/3 | 12.9872 | 13.0119 | +0.1899% |
| B1/4 | 13.0452 | 13.0596 | +0.1107% |
| B4/0 | 51.2169 | 51.1179 | -0.1934% |
| B4/1 | 51.2068 | 51.1212 | -0.1672% |
| B4/2 | 51.1756 | 51.2922 | +0.2278% |
| B4/3 | 51.1742 | 51.1971 | +0.0447% |
| B4/4 | 51.1712 | 51.1663 | -0.0094% |

**No material throughput improvement was measured.** This is near-neutral,
like Gemma, rather than DeepSeek's large ordinary-decode improvement. No
repeatable regression above 2-3% was observed. Paired medians are computed
from pair ratios, not from the ratio of separately computed medians.

## Compatibility scope

The user subsequently requested compatibility work proportional to changed
execution paths. V1 execution, kernels, common forward dispatch and protected
emitters are unchanged. Shared v2 C++ is changed, so Gemma retains focused
actual-weight sites, short decode, cancellation/page reuse and topology checks.
Its broad historical generation/MTP/long-context/performance campaign is not
repeated. The full Devstral campaign and extracted-wheel proof remain required.

Before that scope adjustment, all 127 guards, GLM primitive sites, DeepSeek
actual-weight sites, Qwen/GLM ordinary parity and DeepSeek ordinary/reload
checks had passed. Qwen natural/forced MTP also passed, with 936/1560 native
calls and 1307/2170 compared events. The running GLM MTP check was allowed to
finish: natural/forced cases passed 3452/4621 event comparisons with exact
acceptance and rewind state.

The driver was stopped without interrupting that GLM child, preventing the
queued DeepSeek MTP and broad Gemma campaign from starting. GLM's exit code0
is saved separately in `glm-mtp-completion.json`; the driver termination is
intentional, not a failed correctness gate. DeepSeek MTP is not rerun for this
change; its prior protected validation remains available. This distinction is
also reflected in release metadata.

Gemma's scoped checks all passed (`gemma-smokes`): 360 actual-weight sites
under each of `2,0,1` and `2,0`; all six supported short shapes for eight
steps (480 calls per stage, zero unexpected fallback); Q7/configuration
fallback; exact cross-visibility outputs/state/placement/counts; and a live
cancellation case with two physical page reuses and one recurrent-slot reuse.

## Checkpoint and packaging

All required checkout gates under the revised scope passed. The protected
baseline branch `native-block-ir-v2` remains at `a1bfc9b`. New checkpoint
branch: `native-devstral2-ir-v2`; intended release tag:
`native-devstral2-ir-v2-v1`. Package version:
`1.5.1+nativeplan.v2.devstral1` (Windows CPython3.13 amd64).

The source checkpoint intentionally precedes candidate packaging. Its exact
hash, final tag, wheel filename/hash and native-extension hash are recorded
in the release directory's `checkpoint.json` and README, avoiding a circular
self-hash in this committed report. The release tag is created only after
extracted-wheel verification passes. Those results are recorded separately
in `extracted-wheel-runs.json`, `validation-summary.json` and the final
`package-proof.json`; the candidate's original proof remains retained.

`util/check_devstral2_release_gates.py` verifies required campaign results,
exact short outputs, actual page reuse, reload/topology, source hashes and
the tested extension SHA256 before packaging. The package builder copies that
binary, uses `build_py`/`bdist_wheel --skip-build`, verifies package metadata
and every packaged Python file against validated source, and sets
SOURCE_DATE_EPOCH from the commit. It does not recompile or install anything.

Extracted-wheel verification checks import origins for all product modules
and the native extension, package metadata, Devstral actual-weight sites,
short parity/reload/topology, Gemma sites/short and representative ordinary
DeepSeek/Qwen/GLM parity. Promotion requires every check to pass. Release
artifacts include the wheel, this report/mapping, v2 historical reports,
source archive, verified Git bundle, validation summaries and SHA256 manifest.
Large evidence trees remain at `native-results/devstral2`.

The initial extracted-wheel Qwen launch exposed an unsafe harness invocation:
the profiler sized its cache from `--tokens 64` plus the 173-token prompt
(256 positions), while its parity oracle always runs 128 steps (301 positions
needed). Stock execution hit an illegal CUDA access before native comparison.
The failure and metadata remain in `wheel-validation/qwen-short.log`.
Using `--tokens 128` allocates 512 positions and passes the unchanged exact
128-step stock/bespoke/plan checks. The wheel driver now sizes Qwen for128 and
GLM for256 to cover their fixed parity oracles. This is a validation invocation
fix; no product source or native binary changed, and no assertion was relaxed.

The source checkpoint was updated only for that harness correction and its
documentation. Packaging reuses the same native binary and verified product
sources. When successful checks from the first candidate are retained, the
release includes a comparison proving every ZIP member's uncompressed bytes
identical between candidate wheels (only container timestamps may differ).
The failed Qwen gate is rerun with the corrected capacity; GLM then completes
the remaining package gate. Original failure evidence is never overwritten.

Approximately 50 minutes elapsed from the first inspection helper through
checkout validation; final packaging/verification elapsed time is recorded
in the release metadata. Token/credit usage is not exposed. This includes
historical checks already running before the user narrowed compatibility
scope; future additions should use that narrower scope from the start.

**Did the v2 value model generalize without another native backend?** Yes.
The existing value slots, stages, scratch lifetimes and dispatch support
Devstral with an eight-line emitter and structural Python integration.

**Was further generic expansion needed?** Yes: two additive operations,
RES_ADD_NORM and RES_ADD, reusing existing kernels. The operand model and v1
contract did not need redesign.

**Did native execution materially improve throughput?** No. Both tested batch
sizes are near-neutral, with no repeatable material regression.

**Is the resulting release restorable and validated across existing paths?**
The checkout passes the scope-matched gates above. The final release README
and proof files certify extracted-wheel validation and identify the exact
restorable source/binary artifacts; no release is promoted before that proof.
