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

All 17 checkout gates passed against integration commit
`3389c059563bb3fca00d3e29e20e30a781882b3f`. The release-record commit changes
documentation/tooling only; packaging checks the same product-source hashes.
Evidence: `native-results/v1.5.4-integration/validation-r1`.

| Gate | Result |
| --- | --- |
| Native/dispatch guards, including linked stock/native GR GPU parity | 158 passed |
| Shared HostPool, checkpoint interval, job and GR tests | 23 passed |
| MGEMM variable-width, row-tile and sliced-output regression | 192 cases passed on CUDA 0; 192 on CUDA 1 |
| GLM, DeepSeek and Gemma primitive/site checks | Passed |
| Qwen v1 ordinary | 128 steps; exact stock/bespoke/plan logits, tokens, block residuals, recurrent/PLE and KV/QSA state |
| DeepSeek v1 ordinary | Exact B1/B2/B3/B4 Q1, B1/B2 Q6; reload and fallback checks passed |
| Gemma v2 ordinary | Exact B1/B2/B3/B4 Q1, B1 Q5, B2 Q6, Q7 fallback; reload passed |
| GLM MTP | Natural: 3,456 exact events; forced rejection: 4,621 exact events; 64 tokens each |
| Qwen MTP | Natural: 1,307 exact events; forced rejection: 2,170 exact events; 64 tokens each |
| Qwen concurrency | Exact B2/B3/B4, staggered arrivals, stochastic sampling and page crossing |
| DeepSeek MTP | Ordinary and forced-rejection generator cases passed |
| Gemma reuse | Generator case 5 passed, including page/slot reuse |
| Devstral and Laguna | Exact seven-shape short regressions, including Q7 stock fallback |

Qwen concurrency's existing three-pair performance gate passed. Plan versus
bespoke median aggregate throughput: B2 107.15 versus 106.99 tokens/s (+0.154%);
B3 133.84 versus 133.84 (-0.003%); B4 154.70 versus 155.32 (-0.396%). These are
comparisons within 1.5.4, not against the old 1.5.1 wheel. Single-pair ordinary
Qwen measurements are retained as smoke evidence, not a stable speedup claim.

The isolated build succeeded for sm_89 and sm_120. Initial CPU-only runs also
passed (135 original guards, 155 expanded guards, 13 upstream unit tests), and
270 Python files parsed successfully before the compiled-runtime campaign.

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

The finalizer requires extracted-wheel Qwen three-way and Gemma v2 parity to
pass before producing a validated release directory. Its generated README,
`checkpoint.json`, `package-proof.json` and hash manifest record the final
commit, tag, wheel identity and extracted-wheel results.

Release paths under `native-results/v1.5.4-integration`:

- Extension: `extension/exllamav3_ext.cp313-win_amd64.pyd`.
- Checkout gates: `checkout-gates.json`.
- Candidate wheel and extraction: `package/wheels` and `package/extracted`.
- Extracted-wheel results: `wheel-validation`.
- Final checkpoint: `release`, with wheel, source ZIP, validation ZIP, Git bundle,
  README, package proof, checkpoint metadata and SHA-256 manifest.
- Checkpoint tag: `native-plan-v1.5.4-integration-v1`.

## Remaining scope and limitations

No known correctness failures remain in the completed checkout gates. This is
representative integration validation, not a rerun of every historical campaign.
The 32K/64K campaigns, full long-context architecture campaigns, TP, CPU offload,
external DFlash, newly added upstream architectures and other GPU architectures
were not model-validated here. Their upstream changes were retained. Existing
native eligibility restrictions remain in force. Historical Laguna's two-GPU
B4 performance limitation is not cleared by this short regression campaign.

The wheel is staged for explicit deployment; the production TabbyAPI environment
is not reinstalled or upgraded. Old checkpoint tags, binaries and release reports
remain intact. To return to the original source, use `git switch native-laguna-ir-v2`.
