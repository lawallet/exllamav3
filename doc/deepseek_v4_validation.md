# DeepSeek V4 native-plan validation after the stock fix

The original integration resumed after the shared MGEMM defect was isolated,
fixed and validated. See [the investigation](deepseek_v4_state_investigation.md)
for the separate stock bug and preserved before/after evidence, and
[the architecture map](native_deepseek_v4.md) for the unchanged IR boundary.

All runtime checks use `D:\tabbyAPI\venv\Scripts\python.exe`, checkout Python,
and the isolated extension in `native-results/deepseek-v4/extension` (SHA256
`40ca8e4f2910e72655dc42422af94d72dd629f314746c2d565840c9787c72473`).
Dependencies and the installed TabbyAPI package are unchanged. The branch is
`deepseek-v4-native-plan`. Artifact paths below are relative to
`native-results/deepseek-v4`.

## Resumed short gate

`resume-short`: 32 steps each of B1/Q1, B2/Q1, B3/Q1, B4/Q1, B1/Q6 and B2/Q6
pass exact stock/plan logits, block boundaries, exported target taps where
requested, and initialized persistent state. Each case constructs 43 plans
and executes 1,376 calls at each of begin/middle/finish, with zero fallback.

Unsupported modes, a seven-token query, and incompatible normalization reject
before residual mutation. Restoring the normalization configuration restores
eligibility. Unload clears bindings; reload reconstructs them and reproduces
the same outputs/state. Participating devices are logical CUDA0/1 in this
placement, with the transfer at layer 34. Unused logical CUDA2 records zero
Torch allocation under visibility `2,0,1`.

## Generator instrumentation

The harness records live state, logical paged ranges, block/transfer outputs,
logits, cancellations, slot/page reuse, and speculative rewinds. Counters
separate target and draft plan invocations. Timing runs omit this instrumentation.

The initial `resume-generator` run exposed an assertion error in the harness:
layer-split prefill sets the `prefill` flag only on its last block. Earlier
blocks correctly reject long prompt chunks by query shape. The assertion now
checks ordinary one-token decode separately; it does not classify all
`prefill=False` calls as ordinary decode. No production change was made for
this issue. The failed artifacts remain available.

`resume-generator-v2`: all seven ordinary-generator cases pass: B1..4,
three concurrent jobs plus a staggered replacement, cancellation plus
replacement, and B2 seeded stochastic sampling. Each compares final tokens,
logits, block/transfer boundaries, live target state, and page/slot reuse.
Every fixed-batch job crosses a 256-token page boundary. The staggered case
observes both a reused page and a reused recurrent slot. Supported ordinary
decode has zero unexpected fallback. Expected long-query/final-prefill
fallbacks are recorded separately.

## Short DSpark/MTP gate

`resume-mtp`: nine stock/plan cases pass exact tokens, logits, block/transfer
boundaries, target state, draft KV, and rewind histories. These cover B1..4,
staggered jobs, seeded sampling, and B1/B2 forced rejection. Natural B1 accepts
52 draft tokens and rejects 8; natural B2 accepts 103 and rejects 22. Forced
B1 rejects 320 drafts with 63 positive rewinds, and forced B2 rejects 640 with
126 positive rewinds. The post-rewind live state remains exact.

B1/B2 invoke both target and draft plans. Larger B/Q combinations above the
12-row plan contract fall back before native execution, while supported
shapes as jobs finish still use plans. Draft weights/cache reside on logical
CUDA0, sharing the target's head through the existing attachment protocol.
The unused GPU still has zero Torch allocation.

The original MTP cancellation case finished its short first job before the
fixed cancellation iteration. Its exact result is preserved but does not
establish live cancellation coverage. The harness now cancels only an active
job after at least three generated tokens. The dedicated
`resume-mtp-cancel` rerun passes exact parity with a live cancellation and
replacement job; both page and slot reuse are recorded where observed.

## Visibility equivalence

`resume-topology-short` repeats all six short shapes under visibility `2,0`.
Every stock and plan trace matches `resume-short` (`2,0,1`) exactly: logits,
state, block/transfer boundaries, placement, plan construction and stage/fallback
counts. `resume-topology-mtp` similarly matches natural B1 and forced-rejection
B2 against `resume-mtp`, including target/draft state and rewind histories.
The extra visible device changes none of these results and has zero allocation
in the three-device runs.

`resume-remapped` reverses physical GPU order (`0,2`) and uses validation
budgets [20,90] GiB. It loads an 8/35 layer split, transferring at layer 8,
and passes 32-step exact B1/Q1 stock/plan outputs and persistent state with
43 plans, 1,376 calls per stage and zero fallback. This is a different
placement as well as a different physical-to-logical mapping; no production
guard or dispatch setting was changed.

## Long context

`resume-long`: a 65,472-token prompt followed by 128 B1/Q1 decode steps passes
exact logits, block/transfer boundaries and live persistent state, reaching
position 65,600 across the 65,536 boundary. All 43 plans execute 5,504 calls per
stage with zero fallback. A cold stock sample is retained separately. This
exercises sparse selection with a large compressed pool, not merely the
short-context region below the selection threshold. It is a 64K validation,
not certification of the checkpoint's full 1M advertised context.

`resume-long-mtp`: the same 65,472-token prompt and 128 output tokens pass
exact natural-draft and forced-rejection stock/plan comparisons. Both target
and draft state, logits, block/transfer boundaries, acceptance/rejection and
rewind histories match across the 65,536 boundary. The forced case rejects
634 drafts (one perturbed draft happens to match the target). No tolerance
or live-state exclusion is introduced for long context.

The final native guard suite again passes all 113 tests (`guards-resumed.log`).
The previously completed 414 actual-weight site cases, 75 variable-width
kernel cases per GPU, and Qwen/GLM 128-step regressions use the same extension.

## Performance

Correctness gates above completed before timing. Both backends use the fixed
stock kernel, identical actual checkpoint/quantization, 480-token prompts,
placement [85,40] GiB, visibility `2,0,1`, greedy sampling, and 256 generated
tokens per job. Three pairs per configuration alternate stock/plan order.
Every sample is retained; no outliers are discarded.

Ordinary runs time direct decode after prefill and eight warmup tokens. They
exclude state hashing, invocation counters and prefill. MTP runs time warmed
generator requests, including prefill, with a separate 32-token warmup request
before each sample, five draft tokens and dynamic drafting disabled. Aggregate
throughput counts tokens across all jobs. Ordinary timed pairs also compare
final logits/tokens/live state; MTP pairs compare final job tokens and draft
acceptance/rejection counts.

| Configuration | Stock median tokens/s | Plan median tokens/s | Median paired change |
| --- | ---: | ---: | ---: |
| Ordinary B1 | 36.57 | 43.77 | +19.6% |
| Ordinary B4 | 104.30 | 115.98 | +11.2% |
| DSpark B1 | 85.22 | 86.91 | -1.1% |
| DSpark B2 | 73.97 | 75.66 | +1.5% |

The paired column is the median of each plan/stock pair's throughput change,
not the ratio of the two independent medians. Ordinary B1 pairs are +20.5%,
+9.3%, +19.6%; B4 pairs are +12.1%, +11.2%, +9.2%. DSpark B1 is noisy
(-1.1%, +7.2%, -2.1%), so these samples establish no clear MTP B1 benefit.
DSpark B2 pairs are +1.5%, +1.3%, +3.7%. There is no repeatable regression
above roughly 2–3% requiring another optimization investigation.

Raw results: `resume-benchmark/benchmark.json` and
`resume-benchmark-mtp/benchmark.json`. The complete computed statistics are in
`resumed-performance-summary.json`.

## Engineering result

The current IR supports both HC/norm/residual sites in all 43 target blocks
and all three DSpark blocks without any new primitive, `BC_BlockPlan` contract
change, or architecture-specific C++ wrapper. The architecture addition is
an approximately ten-line emitter, an 88-line registration/eligibility/binding
helper, and small dispatch/lifecycle/export hooks. The only C++ change is the
separately demonstrated shared stock MGEMM correctness fix.

Attention, cache ownership, token-dependent routing, expert execution and
speculative repair remain on their existing Python paths. The
[future-extension note](deepseek_v4_future_ir.md) identifies configurable
routing/external selections as the next useful optional boundary extension,
and stream reduction as another smaller opportunity. Neither is required for
correctness or implemented here.

Enable the integration with `EXL3_NATIVE_PLAN=1` using an extension built from
this source. No DeepSeek-specific native flag is added. Unsupported execution
falls back before native mutation; errors after native entry propagate.
Loaded weights are immutable; load/unload and identity/configuration changes
invalidate bindings. These results cover the tested shapes, placements and
64K context, not TP or the full advertised 1M context. No installed package,
dependency, protected Qwen/GLM reference or release tag was changed; no release
was created during validation. A subsequently packaged and checked local wheel
is documented in [the wheel note](deepseek_v4_wheel.md); it has not been installed
or published.

## Reproducing the gates

Set `CUDA_VISIBLE_DEVICES=2,0,1` and invoke the pinned Python with
`util\validate_deepseek_plan.py --extension-dir native-results\deepseek-v4\extension
--output <new-directory>` plus the arguments below. Each output directory must
be new; failed runs are never overwritten.

| Gate | Additional arguments |
| --- | --- |
| Short shapes and lifecycle | `--phase short --reload` |
| Ordinary generator | `--phase generator --context 480 --steps 64` |
| DSpark and rejection | `--phase generator --mtp --context 480 --steps 64` |
| Visibility equivalence | use `2,0`; same arguments plus `--reference-dir <matching-three-device-run>` |
| Reversed physical order | use `0,2`; `--phase short --diagnose --budgets 20 90` |
| 64K ordinary | `--phase long --diagnose --max-batch 1 --context 65472 --steps 128` |
| 64K DSpark | `--phase generator --mtp --max-batch 1 --context 65472 --steps 128 --case 0 --case 8` |
| Ordinary timing | `--phase benchmark --context 480 --steps 64 --case 0 --bench-batch 1 --bench-batch 4` |
| DSpark timing | `--phase benchmark --mtp --context 480 --steps 64 --case 0 --bench-batch 1 --bench-batch 2` |

The default draft device is a validation choice (`cuda:0`); `--draft-device`
allows another participating device. Neither validation budgets nor visibility
are used by the production guard. `util/summarize_deepseek_validation.py`
summarizes retained generator and timing JSON without loading a model.
