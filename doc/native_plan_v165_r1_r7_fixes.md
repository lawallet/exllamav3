# v1.6.5 native IR: R1 and R7 fixes

Branch: `fix/v165-native-ir-r1-r7`, based on
`9501e76a6e13ca1230270c689777cd746635d86f`.

## Changes

**R1:** `BC_BlockPlan` now checks upstream `Graph.disabled` before capture or
replay and executes its existing eager stage sequence when disabled. This
honors `EXL3_GRAPHS=0` for direct C++ plan calls as well as GLM TP construction.
The separate `EXL3_NATIVE_TP_GRAPHS` opt-out, default GLM activation, fixed-binding
checks and eager DeepSeek policy remain intact. Disabled execution does not
increment capture or replay counters.

**R7:** HC norm operands must be contiguous one-dimensional weights of the
expected hidden width, in FP16 or BF16. Python eligibility checks and both
`BC_BlockPlan` and bespoke `BC_GlmBlock` constructors now match the existing
RMSNorm kernel contract. GLM guards check norm metadata before returning cached
plans. Native stage entry also rejects invalid rebound norms before a residual
update, including eager execution and the global graph-disabled path. No
implicit dtype conversion or kernel support expansion was introduced.

Invalid operands remain errors for direct native callers; Python eligibility
can decline native execution, but that does not promise stock execution can
accept an operand unsupported by its own norm kernel.

## Validation

The new regression cases detected the graph-policy and constructor-validation
issues against the preserved pre-fix extension. Initial negative-control
output is retained in `native-results/v1.6.5-ir-fixes/before.log`.

The fixed extension was rebuilt using the existing compiler object cache,
recompiling the changed C++ translation units and relinking into a separate
output directory. Previous wheels and extracted packages were preserved.
The environment remains Python 3.13, Torch 2.9.0+cu128, CUDA 12.8 and MSVC,
with the existing sm_89/sm_120 build settings. No package was installed.

| Check | Result |
| --- | --- |
| Native/dispatch guards and kernels, shared HC, generator failure containment | 401 passed, 180 subtests passed; no skips or failures |
| Fresh-process global/native graph flag matrix | All nine unset/0/1 combinations passed on all three GPUs; direct calls, GLM policy and eager DeepSeek covered |
| Supported norm parity | FP16/BF16 weights matched across bespoke, eager-plan and graph-plan execution |
| Invalid norms | Construction and rebound dtype/shape/layout checks passed; residuals unchanged after rejected stage calls |
| Norm checks with `EXL3_GRAPHS=0` | 3 passed, 63 subtests passed |
| GLM primitive stage comparisons | Seven batch/query shapes passed exactly |
| Actual-weight DeepSeek target/draft stages | 414 specializations passed |
| Wheel verification | ZIP integrity, packaged-source identity and isolated package/extension imports passed |

The graph-policy matrix also verifies exact eager/candidate results, changing
boundary pointers and FP16/FP32 boundary dtypes, expected capture/replay counts,
and the identity of the extension imported by each subprocess. Graph and norm
regressions cover both Ada and Blackwell GPUs.

Commands, logs, source hashes, binary identity and pass markers are retained
under `native-results/v1.6.5-ir-fixes/validation`. The isolated driver is
`native-results/v1.6.5-ir-fixes/validate.py`; the build invocation is in
`native-results/v1.6.5-ir-fixes/build.bat`.

## Artifact and scope

Wheel:
`native-results/v1.6.5-ir-fixes/package/wheels/exllamav3-1.6.5+nativeplan.v2.tp1-2irfixes-cp313-cp313-win_amd64.whl`

SHA-256:
`a4cc4e2b1fc28059e3436420586fe0707153d49b10e4a341251ab3a8a09d1c88`.

The earlier integration branch and checkpoint tags remain unchanged. Existing
untracked development artifacts, including the architectural review report,
were preserved. No push, runtime configuration change, or installation was
performed. This focused correction does not claim a new full-model serving,
long-context, or throughput campaign, and does not address R2–R6.
