# Native block IR v1 checkpoint

Version: `1.5.1+nativeplan.v1`

Branch: `native-block-ir`

Checkpoint tag: `native-block-ir-v1`

This checkpoint adds one resolved native block planner with six primitives and
Python emitters for the existing Qwen and GLM paths. It preserves the existing
kernels, state ownership, Python attention boundaries and supported shapes.
The bespoke implementations remain available as comparison oracles.

## Selecting the implementation

The generic backend is opt-in. Set `EXL3_NATIVE_PLAN=1` before starting the
process, with the existing `EXL3_NATIVE_QWEN_BLOCK=1` and/or
`EXL3_NATIVE_GLM_BLOCK=1` architecture flag enabled. `EXL3_NATIVE_PLAN=0`
(the default) selects the retained bespoke backend. Disabling an architecture
flag selects its stock path. An older extension without the generic binding
falls back to stock when the plan is requested.

The production Tabby environment has not been upgraded or reinstalled. Test or
deploy the new wheel separately using the existing pinned environment. The
validated configuration is Python 3.13, Torch 2.9.0+cu128, CUDA Toolkit 12.8,
triton-windows 3.8.0.post28 and flash-linear-attention 0.5.2 on Windows/WDDM.
No dependencies, quantizations or production placements were changed.

## Validation

* 104 native guard tests; seven exact native primitive specializations with
  transactional boundary rejection and distinct batch/query contracts.
* Exact stock/bespoke/generic ordinary and MTP parity for Qwen and GLM.
* Dynamic batching, staggered arrivals, sampling, speculative rejection,
  state isolation, page crossing/reuse and GLM cancellation.
* Equivalent `CUDA_VISIBLE_DEVICES=2,0` and `2,0,1`: identical eligibility,
  state, outputs, calls and transfers, zero Torch allocation on unused CUDA2.
* Qwen 64K ordinary and 32K MTP; GLM 64K ordinary and MTP on the 35/10 split.
* Three paired performance runs per case: largest median loss 0.79%; all
  samples and outliers retained. No increase in ordinary hot-path call counts.
* The packaged extension is byte-identical to the model-tested extension.
  Final Qwen and GLM ordinary three-way checks use the extracted wheel.

See `native_block_ir.md` for the engineering report, restrictions, initial
inventory and design. See `native_block_ir_benchmarks.md` for the comparison
table. Validation-adapter failures and their fixes are documented and archived.

## Artifacts

The release directory contains the wheel, source ZIP, validation ZIP, Git
bundle, this README, design and benchmark documents, package proof and a
SHA-256 manifest. `checkpoint.json` identifies the commit and protected refs.

Wheel SHA-256:
`db527b28071f37fae37b3032339614127df6e2ba7439391439ed4c593792bf10`

Native extension SHA-256:
`afbb02a57e61e5f2731b7026cd3c3a211c609b8ac2a6ed7d8a87a136d080d42a`

The protected GLM wheel remains unchanged at SHA-256
`283885767a606dcf9a8e22a7c805b0207ca77334b66c81faba1be0eb966bc5bc`.
Protected tags and branches are not moved. The bundle contains the new branch
and checkpoint with their history; restore it with ordinary Git bundle/clone
commands. The source archive excludes unrelated user prompt files.

## Reproducing checks

Build the isolated extension with `util/build_native_plan.bat`. Run
`util/validate_native_plan.py --phase short --output <new-directory>`, then
`--phase remaining` with another new output directory. The runner invokes the
existing architecture harnesses sequentially and stops on failures. The
`--phase long` option reruns just the long-context gates. Primitive and guard
commands, output structure and the packaging gate are in the source tools and
engineering report. Use new directories to preserve previous diagnostics.

The v1 planner is intentionally restricted: no new architecture, fusion,
compiler/JIT, scheduler, TP or cache rewrite. HC retains the validated H4/D4096
contract and current hardware guards. Loaded weights are immutable; load/unload
clears specializations. Forwards remain serialized on the existing streams.
