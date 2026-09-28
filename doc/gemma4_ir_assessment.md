# Gemma 4 31B: unchanged v1 IR assessment

## Result

**Gemma 4 cannot become a native-plan architecture using the current
`BC_BlockPlan` contract unchanged.** This is a negative architecture-mapping
result, not a completed port. No production dispatch, primitive, C++ wrapper,
reference path, installed package, or dependency was changed.

The six opcode names are not six independently composable operations.
`libtorch/block_plan.cpp` requires every site to be either a `GR_MIX` bundle
or an `HC_MIX` bundle, and every site must be mixed and applied. Both site
types have four residual channels. HC additionally fixes hidden size to 4096.
`execute()` requires FP32 `[B,Q,4,H]` residuals. Gemma has FP32 `[B,Q,5376]`
residuals, no mixing weights, and no stream dimension.

`RMS_NORM` only reads an HC site's FP16 `mixed` scratch and writes its FP16
`normalized` scratch. The instruction verifier requires a preceding HC mix.
It cannot normalize the incoming FP32 residual or a Python boundary output.
`APPLY` calls `hc_apply` with live post gates and optional combination weights;
it cannot express Gemma's post-normalization followed by plain residual add.
Changing those contracts while retaining the opcode names would still expand
the IR. Artificial streams/gates would force a different tensor/math contract
and are not an exact port strategy.

## Actual checkpoint architecture

Checkpoint: `D:\text-generation-webui\user_data\models\turboderp_gemma-4-31b-it-exl3_6.00bpw`.
The inspection reads this directory directly; it creates no model copy or junction.

- 60 text blocks, hidden size 5376, dense gated MLP width 21504,
  GELU tanh approximation, FP16 intermediates and FP32 MLP output.
  `enable_moe_block=false`: no routing or expert computation in this checkpoint.
- Repeating five sliding layers then one full-attention layer: 50 local,
  10 global. Local window includes 1024 tokens; the module receives 1023
  previous positions. Local Q/K/V use 32/16/16 heads of dimension 256.
- Global attention uses 32 Q heads and 4 KV heads of dimension 512.
  `attention_k_eq_v=true`: no global V projection; K projection output also
  supplies V, with the existing V normalization path. This is within-layer
  K/V sharing, not inter-layer KV sharing (`num_kv_shared_layers=0`).
- Weighted Q/K RMS norms, unweighted V RMS norm, attention softmax scale 1.0.
  Local NEOX RoPE uses theta 10000. Global proportional RoPE uses theta
  1000000 and partial rotary factor 0.25. All projection, RoPE, masking,
  cache writes/reads, and attention output projection remain owned by attention.
- Four weighted block RMS norms, epsilon 1e-6, no constant weight bias or
  scale, no grouping. Post norms accumulate directly into the FP32 residual.
- A learned scalar multiplies the entire block output after the MLP residual
  update (and after optional hidden-state export). No per-layer embedding
  (`hidden_size_per_layer_input=0`) or attention/MLP residual scalar.
- Ordinary text embedding is scaled in its loaded dtype by the BF16-rounded
  square root of 5376, then converted to FP32. Final RMS norm emits FP16; the vocabulary projection uses the
  tied-embedding alternative key and applies logit softcap 30.
- Default sliding attention owns ring-cache state through `SWAState`; global
  attention owns ordinary paged KV. Model input preparation runs recurrence
  and attention preparation. State slots, positions, window beginnings,
  stashing, rollback and page ownership remain entirely on the stock path.
  Optional `swa_full` uses paged attention with the same local masks.
- Available model components are text and vision, with no checkpoint-native
  MTP/default draft component. No speculative integration is warranted here.
  Batched multimodal prefill is explicitly unsupported by input preparation.

## Architecture-to-IR map (before implementation)

Here `N(x,w)` denotes the existing RMS kernel with the checkpoint epsilon and
its actual input/output dtypes, not a license to reorder casting or addition.

| Operation, in execution order | Actual inputs/output | v1 mapping | Decision/reason |
| --- | --- | --- | --- |
| Attention input RMS | FP32 residual, learned H-vector -> FP16 activation | `RMS_NORM` mathematically related but unavailable on residual | Python; v1 only normalizes HC mixed scratch |
| Q/K/V projections, head norms, RoPE | Activation, loaded projections/norms, positions | None | Existing attention path |
| Local/global attention and O projection | Q/K/V, live cache/state, masks -> sublayer output | None | Stateful Python boundary |
| Post-attention RMS and add | `x += N(attn_out, post_attn_w)`, FP32 accumulation | Neither `RMS_NORM` nor `APPLY` represents this contract | Existing fused residual-output RMS kernel |
| MLP input RMS | Updated FP32 residual, learned H-vector -> FP16 | Same standalone norm gap | Python; also preserves `params['residual']=x` |
| Dense gated MLP | `down(gelu(gate(y))*up(y))` -> FP32 | `MOE` is not dense MLP | Existing GatedMLP path |
| Post-MLP RMS and add | `x += N(mlp_out, post_mlp_w)`, FP32 accumulation | Same post-norm/add gap | Python |
| Optional hidden-state export | Unscaled residual -> FP16 clamped export | None | Python |
| Learned layer scale | `x *= layer_scalar_f` | None | Python |
| Residual mixing | Absent | `GR_MIX`, `HC_MIX` do not apply | No synthetic mixing |
| Routing / experts | Absent | `ROUTE`, `MOE` do not apply | No fabricated MoE |

Thus **zero operations enter `BC_BlockPlan`** under the unchanged contract.
There is no useful emitter to register. `EXL3_NATIVE_PLAN=1` leaves these
unregistered blocks on stock execution; it must not be reported as successful
native eligibility or stock/plan parity. No native bindings/scratch exist to
invalidate. A future implementation must invalidate norm tensor identities,
normalization configuration, scalar configuration and device/shape bindings on
load/unload and changes, and reject unsupported modes before mutation.

## Future architecture-neutral opportunities (not implemented)

1. **Standalone `RMS_NORM` site/binding.** Input is a resolved activation or
   residual tensor, norm weight and epsilon/bias/scale; output is an explicit
   tensor with specified dtype. Requires removing its dependency on HC scratch
   and accepting single-stream residuals. Existing kernels already implement
   the math. Necessary to put Gemma input norms into the plan, not necessary
   for stock correctness. Could remove two Python norm calls per block and
   apply to many pre-norm decoder families. This is a contract extension even
   if the opcode name is retained.
2. **`NORM_ADD` (proposed new primitive).** Input: sublayer output, norm weight,
   epsilon and writable residual; output: `residual += RMSNorm(output)` using
   stock dtype/rounding and fused accumulation semantics. Current `APPLY` needs
   mix gates and `RMS_NORM` neither accepts the boundary output nor accumulates.
   Necessary for a fully native Gemma residual boundary, only an optimization
   over stock. Could remove two Python post-norm calls per block; Gemma-style
   sandwich-normalized decoders could reuse it.
3. **`SCALE` (optional proposed primitive).** Input/output: writable residual
   plus loaded scalar. None of the six operations scales a plain residual.
   Only an optimization; can remain Python indefinitely. At most one Python
   multiply dispatch per block, reusable by learned-depth/residual scaling.

These are recommendations, not changes authorized or made by this task.
The first two could combine four Python norm dispatches per block (240 per
decode model pass) into native stages; this is a boundary count, not a measured
speedup or a claim that all four GPU launches disappear.

## Validation and cost

The reproducible loaded-weight negative gate is `util/inspect_gemma4_ir.py`.
It uses the pinned TabbyAPI Python and existing isolated DeepSeek extension,
records all loaded block/norm/projection structures and layer scalars, and
attempts standalone norm bundles under all three possible site labels.
It asserts constructor rejection for every block at B1/B2/B3/B4 Q1, B1 Q5
and B2 Q6. It does not fabricate a plan or mutate residuals.

Full decode, concurrency, page crossing/reuse, cancellation, seeded sampling,
state parity, long context and performance must not proceed as stock/plan
gates: the prerequisite architecture/site eligibility gate fails. No measured
native throughput, speedup, successful stage counts, or long-context parity
can be reported. A repeatable regression threshold is consequently not tested.

### Observed results

All runs used Python 3.13.12, PyTorch 2.9.0+cu128 and CUDA 12.8, with the
existing isolated extension in `native-results/deepseek-v4/extension`, SHA256
`40ca8e4f2910e72655dc42422af94d72dd629f314746c2d565840c9787c72473`.
No build or dependency change was needed.

| Visibility | Loading budgets (GiB) | Block split | Rejected constructions | Native invocations | Inspection seconds* |
| --- | --- | --- | --- | --- | --- |
| `2,0,1` | 20, 40 | 44 / 16 | 1080 | 0 | 7.87 |
| `2,0` | 20, 40 | 44 / 16 | 1080 | 0 | 7.39 |
| `0,2` | 16, 50 | 35 / 25 | 1080 | 0 | 7.38 |

*Config/model creation, loading, inspection/rejection and unloading, excluding
Python/import startup; these are not inference benchmark times.*

The first two runs have exactly equal recorded block structures, module
placement and rejection records. Logical GPU2's peak Torch allocation is
zero in the three-visible-device run. Reversed physical ordering and a new
split produce the same 1080 contract rejections. This proves only loaded
structure/rejection topology behavior, not output/state/transfer parity.
The loaded-module device transitions are before layer 44 in the first two
runs and before layer 35 in the remapped run; the normal model transfer path
remains responsible for activation movement. No production ordinal checks
or new transfer/scratch logic were added.

All 240 loaded block norm vectors are contiguous BF16 `[5376]` tensors.
Layer scalars are BF16 scalar tensors loaded on CPU and converted to Python
floats; the first is 0.08935546875. The actual embedding multiplier is 73.5.
Both attention variants and all 60 layer-specific scalar/norm bundles are
preserved in the inspection JSONs, including global layers' absent V modules.

Evidence under `native-results/gemma4/`: `inspect-201.json`, `inspect-20.json`,
`inspect-02.json`, and `visibility-comparison.json`. Logs are in the parent
`native-results/` directory with prefix `gemma4-`.

The existing `test_native*.py` suite passed **113 tests**, covering generic
plans and Qwen, GLM and DeepSeek guards. See `gemma4-guards.log`. No GPU
regression parity runs were necessary for production changes, because none
were made; this does not claim newly certified GPU parity for those models.

An initial inspection used 16 default recurrent cache slots and loading
budgets 12/30 GiB, and the loader reported insufficient VRAM in that split.
The retained `gemma4-inspect-201.log` records this failed attempt. Matching
the cache to the intended four slots and using 20/40 GiB resolved it.
This was inspection configuration, not an isolated stock correctness bug.

Reproduce a successful run in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES='2,0,1'
& D:\tabbyAPI\venv\Scripts\python.exe util\inspect_gemma4_ir.py --output native-results\gemma4\inspect-201.json
```

### Scope and economics

Added 128 physical Python lines (119 nonblank) in the inspection utility,
plus this report and runtime evidence. Architecture-specific production
Python LOC: **0**. C++ changes: **0**. New primitives: **0**. No release.
Existing dirty-worktree DeepSeek/generic changes were left intact.

Effort consisted of source/checkpoint architecture analysis, the negative
loaded-weight audit, three successful topology runs, one corrected loading
attempt and the existing guard suite. There was no long-context, sampling,
concurrency or throughput campaign after the architectural prerequisite failed.
No unrelated stock correctness bug was encountered by this limited audit.
Exact agent usage/credits and a successful-port cost are unavailable.

Unlike DeepSeek's roughly ten-line emitter over an already-supported HC
contract, Gemma needs a generic contract extension before it can have a valid
emitter. The difference is structural, not emitter complexity. This bounded
investigation should not be compared to the cited five-hour/700–800-credit
DeepSeek implementation and stock-bug-fix campaign as if both delivered ports.
The next work item is a separately scoped standalone-norm/single-residual IR
extension, followed by the requested site, decode/state, topology/reload,
long-context and performance gates. It is outside this task's unchanged-IR
constraint and has not been implemented.
