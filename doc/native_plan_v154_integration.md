# ExLlamaV3 1.5.4 native-plan integration

## Provenance

- Validated branch: `native-laguna-ir-v2`, commit `d70a1de81f7fb72d45d95ffd87d76dd0c60cf2c0`.
- Safety branch: `backup-native-laguna-pre-v1.5.4-2026-10-03`, at the same commit.
- Integration branch: `native-plan-v1.5.4-integration`.
- Upstream: `origin` (`https://github.com/turboderp-org/exllamav3.git`), fetched with tags.
- Target: `v1.5.4`, commit `16a49792a3c93d8432d72e6c4bce800841566577`.
- Actual common ancestor: `6b84a21b6f1e5da3f291b9e1019061f0de788279`
  (the MSVC popcount fix immediately after 1.5.1).
- Method: merge. The native branch has 22 commits with checkpoint/release records
  tied to their original identities. A merge preserves that evidence and resolves
  shared-kernel compatibility once against the final validated native state.
- Package version: `1.5.4+nativeplan.v2.integration1`.

## Upstream review and conflict resolution

The upstream delta spans 192 files. The complete commit/file inventory and patch
are retained under `native-results/v1.5.4-integration/upstream-*` and `upstream.patch`.
All nonconflicting upstream changes are retained, including new upstream model
registrations; no additional architecture implementation was introduced locally.

Two textual conflicts occurred:

1. `version.py`: retain the upstream base version and identify the native integration
   with a local version suffix.
2. `quant/exl3_gemm_kernel.cuh`: retain the DeepSeek variable-width MGEMM fallback
   to 128-column tiles. Apply upstream's `TILESIZE_M` row bounds and pointer advances
   consistently, including every narrow-width fallback. Upstream's 32/48/64-row
   kernels do not supersede the narrow-column fix. The regression now exercises
   all seven shapes, multi-tile row counts, both output dtypes and sliced outputs.

Semantic compatibility changes, reported before adapting the native callers:

- Upstream `366463d` removes Qwen GatedResidual's folded `fn_h` table and moves
  norm weighting onto the stream side. Native Qwen now passes a view of the
  unpadded projection rows from `proj_h`; both the bespoke executor and generic
  GR primitive call the new `gr_mix` signature. No duplicate folded table is
  recreated. A capability marker rejects older native extensions for this emitter.
- Upstream `hc_apply` takes two additional optional weighted-output operands.
  Native GR/HC callers supply explicit null operands. The native GR kernel uses
  upstream's inner-loop weighting path. Stock predecessor weighted-copy state is
  consumed before native residual mutation, preventing stale state from escaping
  into a later stock site. Linked-stock/native primitive parity covers this math.
- Upstream keeps folded expert compensation in the routing configuration instead
  of the module's `per_expert_scale`. Qwen native eligibility now rejects nonunit
  `interm_div`, preserving stock routing for that unsupported case. Calibration
  sensitivity capture also stays on the stock path.

The following shared upstream changes were reviewed for native boundary effects:

- Transformer execution: no upstream textual change; all native dispatch,
  exports, lifecycle clearing and Python attention/MLP boundaries remain.
- Architecture loaders/registration: native ownership/attachment hooks coexist
  with Qwen `link_sites`, unified TP-aware MTP head sampling, and DeepSeek's
  resident Markov embedding and inference-mode changes.
- Loading: batched deferred tensor jobs, codebook markers, arena exclusions,
  embedding tables and autosplit resident-state preparation remain upstream.
  Native operand storage stays owned by loaded modules and resolved plans.
- MoE: fractional-trellis instances, shared-expert split-k scheduling, CPU-offload
  host arena/worker changes and expert-output fixes are retained. Native guards
  continue excluding offload/TP and unsupported routing arrangements.
- Attention/state: shared-memory-aware attention dispatch, DSA staging bounds,
  DeepSeek initialization resets, GDN prefill changes, and recurrent HostPool
  checkpoint storage remain upstream; native plans do not own these states.
- Generator/MTP: page-boundary hashing/requeue fixes, rewind cloning, filter-mask
  alignment, draft confidence and unified sampling remain upstream.
- Kernels/build: row-tile expansion, shared-memory footprint guards, RoPE/reduction
  synchronization fixes, quant-cache bounds, TP fixes and compressed CUDA builds
  are retained. No upstream fix was dropped to resolve the merge.
- Conversion, tokenizer, Transformers integration, examples, documentation and
  upstream tests remain as supplied by 1.5.4.

BC_BlockPlan and BC_BlockPlanV2 interfaces and instruction sets remain intact.
The GR operand's representation necessarily follows the new upstream kernel;
the new capability marker identifies that distinction. Native GLM, DeepSeek,
Gemma, Devstral and Laguna emitters are otherwise unchanged.

Parity means exact native-versus-stock behavior within the integrated 1.5.4
runtime. Upstream changed GR numerical evaluation and other shared kernels, so
historical 1.5.1 logits/tokens are not assumed bit-identical to 1.5.4. The native
GR path uses upstream's uncached weighted-input kernel path, while stock may
reuse a predecessor-produced weighted copy. No performance equivalence to the
historical binary is inferred from correctness checks.

## Validation and release status

Integration in progress. Do not treat this branch as a validated checkpoint yet.
Initial existing native guards: 135 passed. Expanded CPU guard suite: 155 passed.
The rebuilt-extension suite additionally exercises exact linked-GR primitive parity.

Reproduce the isolated build with `util/build_v154_integration.bat`; it uses the
existing Python 3.13 / Torch 2.9.0+cu128 / CUDA 12.8 / MSVC workflow and builds
sm_89 and sm_120 into a new directory. Existing binaries are not overwritten.
All runtime validation uses `D:\tabbyAPI\venv\Scripts\python.exe` and
`CUDA_VISIBLE_DEVICES=2,0,1`. The GLM MTP harness accepts this visibility as well
as its historical two-visible-GPU mapping; native placement still uses the
actual model/tensor devices. Existing TabbyAPI models and TextGen source paths
are reused without copying model files or creating an environment.

Run `util/validate_v154_integration.py --extension-dir
native-results/v1.5.4-integration/extension --output <new-directory>` for the
sequential shared/kernel/model gates. Product source and extension hashes are
recorded before validation and checked again at completion.

Packaging uses `util/package_v154_integration.py`, derived from the existing
Laguna packaging workflow, with the integration version. It stages the tested
binary without recompilation and verifies wheel payload hashes against source
and extension inputs. Production installation is outside this release workflow.

Pending: compiled extension, GPU/model results, final wheel, extracted-wheel
validation, checkpoint tag and artifact hashes. Historical 1.5.1 results do not
stand in for these checks; old release documents describe their original tags.
