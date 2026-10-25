# TP cold-start correction

The `1.5.4+nativeplan.v2.tp1` wheel can fail on GLM's first prompt when Triton
has to compile kernels. The expected `shape outside existing small-row IR
domain` warning reports stock prefill execution; the actual failure is
`CPU reduce wait timeout`. Affected GPU ranks can remain blocked after the CPU
helper fails, requiring that backend process to exit before another load.

## Reproduction and validation gap

The original comparison harness explicitly called warmup. The later TabbyAPI
HTTP tests used a new server process but reused the existing compiler cache.
Neither was a genuinely cold-start gate.

The failure was reproduced using the unchanged WebUI environment: Python
3.13.12, Torch 2.9.0+cu128, triton-windows 3.5.1.post24, physical GPUs `2,0`,
GLM-5.3 Flash 3.05bpw, FP16 caches of 65536 tokens, four recurrent slots, MTP
with three draft tokens, and default 2048-token prefill chunks. Vision and MTP
components were loaded before the target, matching the WebUI loader. A synthetic
352-token first prompt reproduced the exact timeout with a private empty
`TRITON_CACHE_DIR`; no user compiler cache was deleted or modified.

Cold compilation was entering real TP collectives. The existing CPU helper has
a 50-second pass deadline, and GPU collectives and host RPCs also have finite
wait budgets. The shape warning is not a failed IR instruction or an omitted
reduction. Existing collective-free warmup before generation made the same
cold-cache case succeed.

## Fix

`1.5.4+nativeplan.v2.tp2` warms a TP target automatically when constructing its
generator, before its page table claims cache ownership. Frontends do not need
to add a manual warmup call. Single-GPU and non-TP generators do not take this
path. MTP verification length is included in the warmup request.

Warmup runs ranks serially through the existing null backend, with a 600-second
host compilation wait per rank call. It never starts the CPU reducer or real
collectives, and it synchronizes local GPU work before acknowledging completion.
Serial rank compilation also avoids the existing Windows Triton cache-file race
between identical GPUs. Normal inference timeout values and collective ordering
are unchanged; this is not a global timeout increase.

Successful warmup is remembered per cache, TP load, chunk/batch/query settings.
Recreating a generator with the same configuration does not repeat it. Reloading
TP invalidates the marker. A failed TP warmup stops instead of dispatching later
warmup families, and prevents generator construction. Synthetic batch size is
capped at eight, then further bounded by recurrent slots; the generator's
256-sequence scheduling default is not treated as a warmup batch.

This is representative-shape startup coverage, not a claim that every possible
future input can never trigger another JIT specialization. The existing IR
small-row limits and visible stock-prefill fallbacks remain unchanged.

No CUDA/C++ source changes or dependency upgrades are involved. The wheel
contains the exact Ada/Blackwell extension from tp1, SHA256
`702c0d54d5285bf7c9b2b408f3754332237ff6b3ab0da8a058290ce150b2e875`.

## Evidence

Artifacts are under `native-results/tp-cold-start`. `baseline` reproduces the
failure, `warmed` verifies the manual-warmup hypothesis, and `automatic-webui`
tests the product fix without a manual warmup call. Each cold run uses its own
initially empty compiler-cache directory. The WebUI product-fix run completed
two MTP requests, with zero native executions counted during warmup and 810
completed native block executions on each rank after generation.

`automatic-tabby` repeats that cold-start gate against the exact extracted tp2
wheel with TabbyAPI's triton-windows 3.8.0.post28. It also completed first and
repeated MTP requests, with zero warmup native executions and 810 native block
executions per rank after generation. Both environments generated 32 tokens per
request with 23 draft tokens accepted and four rejected on this synthetic input.
These are correctness/startup checks, not matched performance measurements.

The packaged wheel passed all 184 native/dispatch/attention-export regression
tests with no skips, plus the TP terminal-signal regression. Eight new regressions
cover per-call timeout isolation, serial dispatch and cleanup, load/cache marker
invalidation, bounded warmup batches, failure handling, and cache ownership order.

## Deployment

Install the complete tp2 wheel and restart the backend; replacing only the
`.pyd` cannot apply this Python startup fix. The wheel remains Windows x64 /
CPython 3.13 / Torch 2.9.0+cu128, with Ada and Blackwell images. No Triton upgrade
is needed. The first cold load can spend several minutes compiling during
generator creation; the console explicitly reports this warmup phase.

TabbyAPI was updated from tp1 with `--no-index --no-deps --force-reinstall`, after
checking the installed tp1 payload matched its rollback wheel. Its normal
configuration and all dependency versions stayed unchanged. WebUI's installed
package was not modified by this work; its isolated tests imported the candidate
code without installing it. The tp1 release and earlier Tabby backup remain
available.

The installed tp2 TabbyAPI backend also passed text completion, templated chat,
and streaming chat requests with two-GPU TP, IR and MTP, generating 64 tokens
per request. This final HTTP smoke test reused the private compiler cache from
the completed Tabby-environment cold gate; it is reported separately from the
empty-cache tests. Automatic startup warmup compiled additional kernels for
the frontend's load settings before serving requests. Runtime diagnostics
verified native execution on both ranks. The temporary server unloaded and
exited, and the normal configuration remained unchanged.

Release files are in `D:\exllamav3\dist\native-ir-tp-1.5.4-tp2`. The manifest
identifies the follow-up commit on `feature/native-ir-tensor-parallel`, which
remains checked out in the main project directory.

The tp2 wheel SHA256 is
`877738c9b80117a6d42b0c746a120f491dd96e0f7ec10425feb6d017a195a6f7`.
