# Native block-plan v2 engineering report

## Design and implementation

See [the pre-implementation design](native_ir_v2_design.md) and the preserved
[Gemma v1 assessment](gemma4_ir_assessment.md). V1 requires GR/HC sites, four
residual streams, and HC-owned norm operands. V2 removes that coupling for
new plans using explicit value slots, while retaining the old frontend and
executor unchanged.

`BC_BlockPlanV2` is an additive generic class, gated by
`native_block_plan_v2_abi == 1`. Existing `BC_BlockPlan`, opcode IDs 0–5,
constructors and Qwen/GLM/DeepSeek emitters keep their original contract.
There is no Gemma C++ class, kernel or native architecture decision. Old
extensions remain usable for v1 and cause Gemma to fall back before construction.

The constructor resolves `(role,dtype)` value slots and four-integer
instructions into function pointers and direct operand/norm pointers.
Roles identify residual, each Python boundary and owned scratch. Each value
has fixed B/Q/D, device, dtype and contiguous row layout. Runtime stages bind
the incoming tensors and validate their entire boundary before launching.
Invalid shapes/dtypes/devices/aliasing reject before residual mutation.
Native exceptions propagate; there is no stock retry after native entry.

`RMS_NORM` operates independently on an initialized residual, boundary or
scratch value. `NORM_ADD=6` uses the stock `rms_norm(..., add_residual=true)`
kernel directly. Supported norm weights are FP16/BF16, inputs and norm outputs
FP16/FP32, with FP32 residual accumulation. Epsilon, constant bias and scale
use the same float conversion and kernel behavior as stock. There is no
decomposition or changed rounding. No math kernel was modified.

Scratch is owned once per specialization, and its initialized-value ordering
is verified at construction. A stage cannot read another stage's scratch
implicitly. Invocations remain serialized, as in v1. Runtime string dispatch,
Python callbacks, arbitrary graph construction, routing expansion, MLP fusion,
attention/cache/scheduler changes and TP were not introduced.

Gemma's emitter lowers four checkpoint norms into begin/middle/finish:
input RMS; post-attention norm-add plus MLP-input RMS; post-MLP norm-add.
Python retains attention, dense GatedMLP, residual-parameter bookkeeping,
unscaled hidden-state export and learned final scalar multiplication. The
structural guard uses actual module/tensor placement, not GPU identities or
visible count. Specializations invalidate on load/unload and changed bound
tensor identity/storage or norm configuration. Selection uses only
`EXL3_NATIVE_PLAN=1`.

## Environment and evidence

Pinned `D:\tabbyAPI\venv\Scripts\python.exe`; Python 3.13.12,
PyTorch 2.9.0+cu128 and CUDA 12.8. No dependency/install changes, checkpoint
copies, or release. `util/build_native_plan_v2.bat` builds only the development
extension at `native-results/block-ir-v2/extension-r2`. The initial extension
remains in `extension` for the retained v1 regression campaign.

Evidence is separated from all prior v1/DeepSeek artifacts under
`native-results/block-ir-v2`. Existing dirty-worktree DeepSeek/MGEMM changes
remain intact and are not attributed to v2. The shared MGEMM fix is retained
in the isolated build, not modified by this task.

The initial `v1-regressions` driver attempt ran guard tests with the plan
environment enabled. Several old mock-based tests explicitly expect the
default bespoke backend; they failed before any GPU regression ran. The
corrected driver uses the default backend for guards and enables the plan for
GPU tests. Original failure evidence is retained. This was a harness setting,
not a production correctness change.

The legacy GLM MTP harness also asserts visibility `2,0` at startup. Its
first `2,0,1` launch stopped there without running inference; the resumed
driver uses the harness's prescribed topology. No production guard was changed.

Native API review added preflight validation of every bound norm's metadata,
including norms consumed later in a stage, and an immutable device contract.
This prevents a direct caller's post-construction tensor resize from allowing
an earlier norm-add to mutate the residual before a later failure. That
revision is `extension-r2`. An attempted link to the still-loaded first binary
failed with Windows LNK1104; the old binary was retained, and the revised build
succeeded in the separate directory. Build/failure logs are retained.

## Validation status

All requested acceptance phases completed. Certification is limited to the
checkpoint, shapes, placements and contexts recorded below.

- Initial actual Gemma weights: 360 specializations (60 blocks × six shapes),
  exact begin/middle/finish activations and residuals. Constructor/boundary
  rejection tests passed. Updated campaign adds explicit cross-CUDA rejection
  and generic normalization directly from a Python-boundary value.
- Expanded native guard suite: 120 tests passed.
- Existing GLM primitive sites and DeepSeek actual-weight sites passed.
- Qwen and GLM short exact parity and instrumented plan/reference runs passed.
- DeepSeek B1..4/Q1, B1/Q6 and B2/Q6 passed exact output, block and valid-state
  comparisons, unsupported-mode/config rejection and unload/reload.

- Qwen natural and forced-rejection MTP passed (64 generated tokens each),
  including exact events, state and rewind/acceptance behavior. GLM natural and
  forced MTP also passed (3452 and 4621 compared events). DeepSeek natural and
  forced B1 MTP passed, with target/draft stage counts and state comparisons.
- Final `extension-r2` site gates passed all 360 specializations under both
  `2,0,1` and `2,0`, including explicit cross-CUDA rejection, independent
  boundary-input normalization, invalid config and changed-late-binding safety.
- The retained older DeepSeek extension rejects v2 before accessing operands;
  final v2 guard suite again passed all 120 tests.

### Reload investigation

The first Gemma short campaign passed all six supported shapes for 32 steps,
real Q7 fallback, and a valid stock Q1 normalization configuration rejected by
the v2 guard. It then stopped at the exact reload comparison. No later gate
was run past that failure.

Saved diagnostic traces located the first difference at layer 43, which
autosplit moved from logical GPU1 to GPU0 after reload. Stage counts remained
identical. Removing the identity probe's retained plan/weight references alone
did not resolve it. The stock process-wide `g_tensor_cache` retained decode
workspaces after model unload. The loader correctly treats `use_per_device`
as additional headroom over current reservations, so the same numeric budgets
did not imply the same split after warming new shapes.

In the isolated harness, dropping that workspace cache after model unload
reduced reservations from 583008256/687865856 bytes to 29360128/25165824 bytes.
Reload then preserved placement and passed exact outputs, block/transfer
boundaries, exports, initialized state and reconstruction counts. The exact
assertions were retained and an explicit placement assertion was added. No
production loader/cache change or relaxed numeric tolerance was made.

Evidence: `gemma-campaign-r1`, `gemma-reload-diagnostic`,
`gemma-reload-diagnostic-r2`, and successful `gemma-reload-diagnostic-r3`.
The resumed full campaign is `gemma-campaign-r2`.

## Completed Gemma campaign

Evidence: `native-results/block-ir-v2/gemma-campaign-r2`. Short runs tested
B1/B2/B3/B4 Q1, B1 Q5 and B2 Q6 for 32 steps each. Logits, tokens, block and
device-transfer boundaries, hidden exports and logically initialized SWA/KV
state matched exactly. Each supported shape constructed 60 plans and executed
1920 begin, middle and finish calls, with zero fallback. Q7 exercised real
stock fallback with zero native calls. A valid stock norm configuration rejected
by eligibility exercised one stock block and 59 native blocks (118 calls per
stage over two steps, two expected fallbacks). Mode guards, changed-weight
reconstruction and unload/reload also passed.

Ordinary generator runs used context 240 and 48 output tokens, crossing a
256-token page boundary. B1 through B4 each executed 2880 calls per stage.
Staggered arrivals executed 4320; cancellation/replacement executed 3060;
seeded B2 sampling executed 2880. All compared tokens, logits, blocks and live
local/global state exactly. Prefill correctly used stock (60/120/180/240
fallbacks for B1/B2/B3/B4); there were zero unexpected ordinary decode
fallbacks. Dedicated 1536-token-cache stagger/cancel runs each demonstrated
two page reuses and one slot reuse. Cancellation was actually performed.
Gemma has no checkpoint-native MTP; these are ordinary generation tests.

Visibility `2,0,1` versus `2,0` preserved the 43/17 block split and matched
outputs, state, transfers, construction/stage/fallback counts and reload
exactly. The unused third logical device had zero allocated, reserved and
peak Torch memory. Remapped visibility `0,2` with budgets 16/50 produced a
34/26 split and passed stock-versus-plan parity for every short shape and
fallback case. Cross-physical-placement bitwise equality is not claimed:
stock itself can round differently when a block moves between GPU families.

Long-context tests used 8192 input tokens, then 16 steps of B1 Q1 and B1 Q5.
Both passed exact output, export, block and initialized-state checks, with
60 plans, 960 calls per stage and zero fallback. Q5 reached position 8272
with window beginning at 6912, exercising repeated local-window movement and
long global paged KV. This does not certify the maximum 262k context.

## Performance

Five alternating warmed pairs per batch used identical placement, context 512,
256 timed steps and eight warm steps. Validation hashing/instrumentation was
excluded from timing. All samples are retained under the campaign's
`performance` directory. B4 rates are aggregate tokens/second.

| Batch | Stock median t/s | Plan median t/s | Median paired change |
|---|---:|---:|---:|
| B1 | 37.6523 | 37.6145 | -0.2308% |
| B4 | 139.6109 | 139.4078 | -0.1485% |

| Batch/pair | Stock t/s | Plan t/s | Paired change |
|---|---:|---:|---:|
| B1/0 | 37.709 | 37.622 | -0.232% |
| B1/1 | 37.692 | 37.676 | -0.041% |
| B1/2 | 37.652 | 37.547 | -0.279% |
| B1/3 | 37.651 | 37.615 | -0.097% |
| B1/4 | 37.596 | 37.509 | -0.231% |
| B4/0 | 139.622 | 139.414 | -0.148% |
| B4/1 | 139.611 | 139.582 | -0.021% |
| B4/2 | 139.618 | 139.408 | -0.150% |
| B4/3 | 139.494 | 139.390 | -0.074% |
| B4/4 | 139.503 | 139.208 | -0.211% |

This is near-neutral throughput, not a speedup. The small negative changes are
well below the 2-3% investigation threshold. Median paired change is computed
from paired ratios, not the ratio of the two medians.

V1 native execution and protected Qwen/GLM branches are unchanged. Their
single-pair bespoke/plan smoke rates were respectively 72.2435/72.8381 and
45.7462/45.9791 t/s; these are not statistical before/after evidence. DeepSeek
shares the adjusted Python dispatch, so a separate five-pair test reconstructs
the exact pre-v2 method by reversing only this task's edits (saved alongside
the results), using the same final extension and plan emitter on both sides.
It uses B1, context 300, 128 timed steps and eight warm steps. Final logits,
tokens and valid state match across all runs. Final-build stock/plan four-step
parity also passed, constructing 43 plans with 172 calls per stage.

| Pair | Pre-v2 dispatch t/s | Current dispatch t/s | Paired change |
|---|---:|---:|---:|
| 0 | 43.4152 | 39.5799 | -8.8341% |
| 1 | 41.2882 | 46.6244 | +12.9242% |
| 2 | 38.6150 | 38.9320 | +0.8209% |
| 3 | 38.8322 | 38.7415 | -0.2336% |
| 4 | 39.1562 | 46.6235 | +19.0705% |

Medians are 39.1562 versus 39.5799 t/s; median paired change is +0.8209%.
The samples are noisy and do not establish a speedup or a repeatable regression.
Evidence: `deepseek-dispatch-performance`. Final GLM primitive checks also
passed (`final-glm-sites.json`). The full old-architecture generation/MTP
campaign used the first isolated binary; the final revision only strengthens
v2 norm preflight and leaves v1 native code unchanged. Final guard, Gemma,
GLM primitive and DeepSeek parity/dispatch checks used `extension-r2`.

## Size, scope and reproduction

Physical lines including comments/blanks: new generic C++ is 35 header plus
116 implementation lines, with nine binding lines. Gemma adds a 72-line
guard/registration module, ten-line emitter and three-line architecture hook
(85 architecture-specific lines). Generic Python adds the capability/opcode
declarations and approximately eight net shared TransformerBlock lines; the
validation cache clearer gains the Gemma cache. Guard tests occupy 123 lines.
The extensive standalone validation utilities and evidence dominate this task.
Once the generic operand contract exists, the emitter is similarly small to
DeepSeek's; implementing and validating that new contract was the extra work.

Attention, dense MLP, state exports and final scalar remain Python boundaries.
Routing, external MOE selection, stream reduction, fusion, cache/scheduler
changes, TP and compiler/JIT work remain deliberately out of scope. No stock
math correctness defect was found here; failed launches and reload placement
were harness/setup issues documented above. The earlier shared MGEMM fix and
its investigation remain separate.

Build with `cmd /c util\build_native_plan_v2.bat`. Run the pinned Python with
`util/validate_ir_v2_regressions.py --output <new-directory>` and
`util/validate_gemma4_v2_campaign.py --output <new-directory>` for the campaigns;
each driver manages its required visibility. The shared-dispatch comparison
uses `util/compare_deepseek_v2_dispatch.py --output <new-directory>` with
`CUDA_VISIBLE_DEVICES=2,0,1` and `EXL3_NATIVE_PLAN=1`. These utilities load the
checkout against the isolated development extension, not an installed update.

Extension SHA256:

- Initial: `6DE9EB84D60B83127257C355DC935CFEE7FC6782C159A6C43341E1CC5EDDDC01`
- Final: `F623E316F427DA44002E77E1CD6E99C281EC2A7722B5E2DD954C61B4A85AB6BC`

Approximate elapsed design/implementation/testing time: 55 minutes, starting
at 18:42 EDT on September 27, 2026. Token/credit usage is unavailable. No
production package was installed or released, and pinned dependencies remain
unchanged.

**Did v2 remove HC/GR operand coupling without breaking validated v1 paths?**
Yes, within the tested coverage: independent values and norm operands serve
v2, while v1 retains its executor/API and passed its regression campaign.

**Can Gemma now use `EXL3_NATIVE_PLAN=1` without Gemma-specific C++?**
Yes. Supported decode shapes execute all 60 blocks through generic v2 norm
stages, with exact stock parity and measured native invocation counts. A
v2-capable extension is required; older extensions safely retain stock Gemma.
