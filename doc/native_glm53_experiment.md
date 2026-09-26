# GLM 5.3 Flash ordinary-decode experiment

Status: native implementation and exact packaged wheel passed GLM parity,
alternating benchmarks, and Qwen exact regression with the GLM flag enabled.

Protected starting revision: `39b6a5e994b728bc35aae89963f5739a1f500a48`.
Development branch: `native-glm53-decode`. Existing Qwen artifacts and installation
are retained. GLM outputs live under `native-results/glm53/`.

## Environment and scope

Python 3.13.12; Torch 2.9.0+cu128; CUDA toolkit 12.8;
triton-windows 3.8.0.post28; flash-linear-attention 0.5.2;
installed ExLlama 1.5.1+nativeqwen.concurrent1. Checkout Python imports the existing
installed extension for the baseline. No dependencies were changed.
Verified host: Windows 11 Pro build 26200, AMD EPYC 7663 (56 cores / 112 logical
processors), 274735357952 bytes of reported physical memory (256 GB installed).

User confirmed physical GPU 2 (RTX Pro 6000 Blackwell, 300 W) followed by physical
GPU 0 (RTX 4090D 48 GB, 425 W). Physical GPU 1 is excluded. CUDA_VISIBLE_DEVICES=2,0
maps these to logical 0 and 1. Initial budgets are 90 and 44 GiB; these budgets
were chosen for this experiment, not recovered from a production configuration.
Tabby's saved config currently selects Qwen. Actual placement is recorded by the
harness after loading; budgets alone do not establish a layer boundary.

MTP disabled, one active sequence, batch 1, ordinary single-token decode, no TP.
Loaded placement: embedding on CPU; expansion and layers 0–34 on logical GPU 0
(physical Pro 6000); layers 35–44, mean collapse, final norm and head on logical
GPU 1 (physical 4090D). The sole residual-stack inter-GPU transition is immediately
before layer 35. The CPU embedding uploads before stream expansion. Cache capacity
is 1280 tokens; the stock Cache constructor's default slot capacity is 16, with
only one active sequence and no history. The saved prompt contains 32 tokens.

## Model discovered from local configuration and source

Directory: `D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw`.
Architecture: `Glm5NextForConditionalGeneration`, text component `Glm5NextModel`.
EXL3 3.05 bpw, head 6 bits, conversion version 1.4.5. Hidden width 4096,
vocabulary 154880, 45 transformer layers. There are 34 KDA layers and 11 NoPE MLA
layers at indices 3,7,11,15,19,23,27,31,35,39,43.

KDA: 64 heads of width 128, short convolution kernel 4, gate lower bound -5.
MLA: 64 query heads, query low rank 1536, KV low rank 512, NoPE head width 256,
value width 256, zero RoPE width. DSA indexer: 32 heads, width 128, top-k 2048,
key pool compression 4, always select tail, full indexers.

Every layer carries four FP32 residual streams and two mHC sites with 20 Sinkhorn
iterations. Each site collapses to FP16 before weighted RMSNorm. Final stream
collapse is an unweighted mean, followed by RMSNorm and the output projection.
First three MLPs are dense (width 12288); remaining 42 are MoE with 288 routed
experts, 8 selected, one shared expert, intermediate width 2048, sigmoid/noaux_tc
normalized routing, scale 2.5 and clamped SiLU limit 10.

The configured single MTP layer uses normalized embedding/hidden concatenation,
an input projection, a plain residual MLA decoder (no mHC), and shared output
head. It is neither loaded nor optimized here. Vision is also outside scope.

## Source execution trace

The generator invokes Model.forward. GLM prepare_inputs prepares paged attention
and recurrent slots; forward_ls iterates module/instance pairs, calls
prepare_for_device, then forward. Recurrent positions advance after the forward.
Placement and transfers remain owned by this existing layer-split mechanism.

Each transformer layer executes:

1. HyperConnection.mix: views the residual, determines chunk count, looks up four
   per-device bucketed workspaces, creates views, calls ext.hc_mix (existing fused
   CUDA partial/finalize kernels), and returns three views.
2. RMSNorm.forward: FP16 collapse view, output allocation, ext.rms_norm.
3. GatedDeltaNet.forward or MLAttention.forward.
4. HyperConnection.apply_: views and ext.hc_apply, updating residual in place.
5. A second mix and RMSNorm, followed by dense/MoE MLP and a second apply_.

Thus the source path has 45 block forwards, 90 mix calls, 90 apply calls and 90
outer RMSNorm calls per decode token before attention/MLP internals. These are
source-derived counts, not measured timing or proof of a host bottleneck.

KDA resolves the recurrent slot and layer cache in Python. Eligible decode uses
BC_GatedDeltaNetSplit.needs_configure/run_bszN with existing internal graph replay;
convolution and recurrent states remain in that cache. It allocates the output in
Python. MLA retrieves per-device block tables/cache lengths, looks up the cache
layer and dispatch cache, and attempts its existing native BC MLA/DSA path before
falling back. Compressed indexer state must be included in any eventual parity
test, as well as latent KV, convolution and recurrent tensors.

MoE routing and expert execution already have optimized extension kernels;
eligibility and buffer/routing orchestration remain in Python. Runtime profiling
must establish which concrete routing and attention branches this model takes.
No inference implementation changes are justified solely by this source trace.

## Measured baseline and chosen boundary

Three uninstrumented 1000-token runs: 37.8361, 37.3021, 37.6540 tokens/sec;
total times 26.5495, 26.9478, 26.7030 seconds. All completion hashes matched.
Separate 32-token host profile: mHC mix 8.796 ms/token, mHC apply 2.777 ms,
RMSNorm 4.515 ms, MoE 4.336 ms, KDA 1.559 ms, MLA 1.054 ms. Instrumented
model.forward averaged 26.219 ms. The CUDA trace spans 1975.268 ms and records
339.922 ms / 363.320 ms of device activity on logical devices 0 / 1; that
instrumented window includes inter-token gaps and cannot quantify Python-only
idle time. These observations motivate measuring a narrow orchestration change.

BC_GlmBlock owns per-block, per-device scratch for the two mHC sites and RMSNorm.
begin performs attention mix + norm; middle applies attention residual then MLP
mix + norm; finish applies MLP residual. Existing hc_mix, rms_norm and hc_apply
CUDA functions execute in exactly their stock order. The attention and MLP calls
remain Python-owned, including routing, expert execution, native graph replay,
cache lookup, recurrent slots and indexer state. No new graph or math kernel is
introduced. The block's CUDA device is guarded in C++, and every weight/workspace
belongs to that device. Scratch holds no persistent model/cache state.
Existing synchronization includes sampling results returned to the CPU and
blocking device-transfer behavior in to_device; its initial peer-copy probe runs
during loading/warmup. The harness synchronizes both devices at run boundaries.
The native wrapper adds no synchronization call or inter-device copy.

The flag is `EXL3_NATIVE_GLM_BLOCK=1`, independent of the Qwen flag. Guards limit
the prototype to the measured architecture, shape, normalizations and GPU mapping
with the split before layer 35. Prefill, non-single-row inputs, history/MTP state,
TP, capture/conversion, LoRA and export requests fall back. Missing extension
support also falls back. Reload invalidates native scratch.

Parity covers 256 decode steps (including a page crossing), byte hashes of logits,
tokens, every block residual, selected experts and weights, final hidden states,
both sides of every logical transfer, cache/slot positions, and six snapshots of
all convolution/recurrent tensors and valid latent/indexer/pooled-key cache rows.
It compares valid entries rather than uninitialized padding. Native execution is
required for every eligible block; failure aborts before any native benchmark.

## Baseline reproduction

Run with the existing venv from the checkout:

```powershell
$env:CUDA_VISIBLE_DEVICES='2,0'
& D:\tabbyAPI\venv\Scripts\python.exe util\profile_glm_decode.py --model D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw --split 90 44 --output native-results\glm53\baseline-exploratory --trace
```

Output directories must be new to prevent accidental result overwrites. Greedy
sampling, seed 1234, a saved short prompt, FP16 default cache, 128-token warmup,
three retained 1000-token runs, then separate 32-token dispatch and CUDA profiles.
Throughput excludes the first generated token using the generator's decode time;
total elapsed time includes generation and synchronization of both active GPUs.
Telemetry records all physical GPUs once per second with timestamps to associate
samples with runs. Instrumented host times overlap GPU execution and do not by
themselves measure GPU idle time. Native benchmarking is deliberately unavailable
until a separate exact-parity gate is implemented and passes.

## Rollback

The installed Qwen package is unchanged. Disable any experimental GLM flag and
return to `native-qwen-concurrency` / protected revision `39b6a5e` when finished.
Do not remove or overwrite the existing Qwen wheels, validation files or bundles.

## Staged validation results

256-step parity passed across all 45 native blocks, including the cache page
boundary. No tolerance was used. Full tensor-hash records are retained in
`native-results/glm53/ab-staged/parity.json`. Both modes recorded exactly one
embedding upload and one 64 KiB FP32 residual-stack inter-GPU transfer before
layer 35 per decode. The native boundary contains no copy operation.

| Pair/order | Stock t/s | Native t/s |
| --- | ---: | ---: |
| 0 A/B | 38.5027 | 46.9498 |
| 1 B/A | 39.4321 | 47.2667 |
| 2 A/B | 38.7436 | 47.1234 |
| Mean | 38.8928 | 47.1133 |
| Median | 38.7436 | 47.1234 |

Mean improvement: 21.1362%. All six 1000-token completion SHA-256 values are
`f469da222ccd55f5fb45670c8e798d823f65a8aa0ccfc0b3bf045c4af04e0bde`.
No runs were removed, including the slower initial stock-only session.

Sampled mean GPU utilization per run: Pro 6000 stock 44.26–45.74%, native
53.74–54.75%; 4090D stock 37.78–38.48%, native 36.79–39.05%. Mean power:
Pro 6000 stock 147.53–160.03 W, native 166.41–179.48 W; 4090D stock
64.42–65.22 W, native 71.32–72.40 W. Maximum observed VRAM was unchanged
at 92942 MiB / 30391 MiB. WDDM utilization is sampled device telemetry, not
kernel occupancy. Raw timestamps, power limits and all samples are retained.

Instrumented module/method calls fell from 412 to 142 per token. Native/pybind
calls fell from 528 to 303: 90 mix chunk queries, 90 mixes, 90 outer norms and
90 applies became 45 begin + 45 middle + 45 finish calls. Attention, routing,
expert and output-projection call counts stayed the same. Every decode layer
used native orchestration, with zero eligible-block fallbacks in the profile.

All eight GLM guard tests and all 29 existing Qwen guard tests passed. With the
rebuilt extension, the existing Qwen 128-step exact-parity test passed on all
48 blocks (tokens/logits/residuals/recurrent/PLE/KV/QSA), followed by matching
128-token stock/native smoke completions. These smoke rates are not a new Qwen
performance claim. Results are under `native-results/glm53/qwen-regression/`.

Remaining limitations: only the measured 35/10 split and CUDA_VISIBLE_DEVICES=2,0
mapping are enabled; other placements fall back. Independent simultaneous calls
or streams are outside the stock serialized execution contract. Short-context
validation does not establish long-context sparse top-k behavior. No MTP,
concurrency, TP, CPU offload or vision validation is claimed. The 256-step
parity check samples persistent state at six steps and compares block outputs
and logits at every step; it does not hash every persistent byte on every step.

## Build and packaged reproduction

`util/build_native_glm.bat` invokes the existing VS 2022 toolchain and venv with
MAX_JOBS=4, CUDA 12.8, SM89 and SM120, no build isolation and no dependency
installation. The emitted ninja configuration contains both architecture flags.
The wheel is `exllamav3-1.5.1+nativeglm.decode1-cp313-cp313-win_amd64.whl`.
SHA-256: `5d8beeb3be5c6b1e5ba58823eb3a386aa2811d59102ca77261b6dc63dbd8bfd7`.

After extracting the wheel into a new directory (a wheel is a ZIP archive):

```powershell
$env:CUDA_VISIBLE_DEVICES='2,0'
& D:\tabbyAPI\venv\Scripts\python.exe util\profile_glm_decode.py --model D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw --split 90 44 --output native-results\glm53\ab-wheel --package-dir native-results\glm53\wheel-stage --ab
& D:\tabbyAPI\venv\Scripts\python.exe util\summarize_glm_decode.py native-results\glm53\ab-wheel
& D:\tabbyAPI\venv\Scripts\python.exe util\check_glm_wheel_qwen.py --package-dir native-results\glm53\wheel-stage --model D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw --output native-results\glm53\qwen-wheel.json
```

Use a new output directory on subsequent runs. `--ab` always runs exact parity
before throughput and controls the GLM flag internally. Normal inference enables
the path by setting `EXL3_NATIVE_GLM_BLOCK=1` before importing ExLlama, with the
same GPU mapping and placement. The flag defaults off. The experiment did not
install the wheel into Tabby's environment. To install later, use the existing
venv's pip with `--no-deps --force-reinstall` and the preserved wheel path.
For package rollback use the preserved Qwen wheel under `native-results/wheels/`
with those same pip options; no dependency upgrade is needed.

## Exact wheel results and preserved checkpoint

The extracted wheel passed the complete 256-step GLM parity gate again. Source
hash comparisons confirmed that its GLM Python implementation matches the tested
checkout. The wheel also passed the existing Qwen 128-step exact-parity check on
all 48 blocks with the GLM flag enabled, confirming independent selection. See
`native-results/glm53/qwen-wheel.json` for imported package/extension paths.

| Pair/order | Wheel stock t/s | Wheel native t/s |
| --- | ---: | ---: |
| 0 A/B | 32.5503 | 45.6795 |
| 1 B/A | 33.0810 | 45.6206 |
| 2 A/B | 32.4188 | 45.5355 |
| Mean | 32.6833 | 45.6119 |
| Median | 32.5503 | 45.6206 |

Wheel session improvement: 39.5569%. All completion hashes match the staged
session. Stock throughput was lower than the earlier session (38.8928 mean),
while native fell less (47.1133 to 45.6119). The cause of this session variation
was not established; neither session nor any run was discarded. Treat the two
session gains (21.1% and 39.6%) as observed results, not a guaranteed speedup.

Wheel telemetry per-run means: Pro 6000 stock utilization 37.93–38.79%, native
51.75–52.65%; power 139.63–151.30 W versus 170.15–179.86 W. 4090D stock
utilization 35.54–36.39%, native 39.15–39.85%; power 58.99–59.68 W versus
69.37–70.02 W. Peak VRAM remained 92942 / 30391 MiB. GPU budgets, power limits,
prompt, cache, sampler and placement matched the staged session.

Validated tag: `glm53-native-nonmtp-v1`, on branch `native-glm53-decode`.
Release files are preserved in `native-results/glm53/release/`: the exact tested
wheel, a source ZIP, validation ZIP, Git bundle and SHA256SUMS.txt. The checksum
manifest identifies the source commit and every release artifact; its own hash
is not recursively included. The Git bundle contains the GLM branch/tag and its
history back through the protected Qwen checkpoint. Raw results, traces and
telemetry remain under `native-results/glm53/` as well.

## MTP phase: trace from the protected non-MTP checkpoint

Starting point: `glm53-native-nonmtp-v1` (`1630fd3`), new branch
`native-glm53-mtp`; outputs under `native-results/glm53-mtp/`. Python, Torch,
CUDA, Triton and FLA versions and 300/425 W power limits were rechecked unchanged.
The model now has configured `mtp_bits=4`; its text architecture configuration
matches the saved non-MTP configuration. No model/dependency conversion is done.

GLM's single MTP layer has `default_draft_size=3` (also the Tabby fallback).
This is the selected fixed window, not a claim of a hard engine limit: the
generator accepts an override. Dynamic confidence truncation is disabled here.
Observed stock shapes: target verification (1,4), draft forward (1,1), and
accepted-prefix repair prefill (1,1), (1,2), (1,3). Initial target forward and
draft prefill use the 31-token prompt prefix. A bootstrap with no carried hidden
state can use target (1,1). Target verification sets recurrent_history=True.

Stock Tabby loading order was reproduced: draft autosplit first, target 90/44 GiB
second. Actual target placement remains layers 0–34 on logical 0 (Pro 6000),
layers 35–44 and head on logical 1 (4090D). All MTP modules reside on logical 0.
attach_to binds the CPU embedding and shared target head and requests export of
the POST-final-RMSNorm target hidden state (after mean stream collapse).
The draft input separately normalizes that hidden state and the token embedding,
concatenates embedding then hidden (8192 channels), and projects to 4096.
The draft block is plain residual MLA/MoE, followed by its own final norm.

MTP adds stock hidden-state traffic: target carry/repair inputs travel from the
4090D to the Pro 6000; draft outputs travel back to the shared head on the 4090D.
Between draft steps, the returned hidden state again moves to the Pro 6000.
Draft embeddings upload from CPU. The native trunk boundary must add none of
these copies or move any module/cache. Trunk verification carries a four-row
residual through the same layer-35 boundary.

KDA cache history has four recurrent matrices per slot (current plus three
history entries), and convolution width 4+3. Verification advances four tokens;
GDNState records last_history=3. Rejection calls rewind(n), which dispatches the
existing per-device batched recurrent/convolution rewind kernels, subtracts n
from position and clears last_history. Full acceptance calls rewind(0) to
normalize convolution state. MLA latent/indexer/pool tensors are not erased on
rejection: page/sequence lengths retreat, invalid suffixes are ignored and later
overwritten. Complete pooled keys are valid only through floor(valid_length/4).

After acceptance, Generator.iterate_gen repairs draft cache positions K+1 through
K+A-1 using accepted IDs and exported target hidden states via draft.prefill.
It clones the target hidden at A-1 into job.mtp_last_hidden for the next round.
Persistent state consists of that carry, target recurrent/convolution state,
target/draft MLA planes and shared logical page positions; the draft has no KDA.

Existing GLM guards reject export_state_norm_keys, recurrent_history, nonzero
cache.max_history and any shape other than (1,1,4,4096). Native scratch and
views are hard-coded to one row. Simply dropping those guards would be unsafe.
The extension needs independently sized four-row scratch using the same stock
hc_mix/RMSNorm/hc_apply kernels and operation order, plus an attachment-aware
guard for the actual three-draft protocol. The plain residual draft stays stock.

Stock trace: 29 target q4 forwards, 87 draft q1 forwards, repair q1/q2/q3 counts
11/3/6 in the 64-token diagnostic. Instrumented target host time per forward:
mHC 10.08 ms, outer RMSNorm 3.47 ms, MoE 6.58 ms. Draft host methods total about
0.96 ms per forward, with input preparation/transfers and norms prominent;
MLA/MoE already dispatch existing native kernels. Optimizing 45 trunk mHC sites
is the evidence-based priority; no new specialized draft kernel is justified in
this phase. The first stock 1000-token MTP run measured 64.70 effective output
t/s, with 646 accepted and 416 rejected drafts. First-use compilation affected
the preceding 128-token warmup, so final benchmarks warm the full measured span.

### MTP implementation and validation contract

BC_GlmBlock now accepts a row count of 1 (the original constructor default) or 4.
Its allocations/views use that count; the three boundary calls, math kernels,
weights, normalization arguments and operation order are unchanged. Ordinary
decode retains its original eligibility logic and scratch cache. MTP dispatch
uses a separate `native_glm_mtp.py` guard and per-query-length scratch cache.
GLM attach_to records a weak reference solely for validating the trunk/draft
relationship. Loading/unloading clears the added scratch cache.

MTP eligibility requires the attached GLM text model, the precise final-norm
export key, a single local recurrent slot, history capacity 3, the validated
35/10 target placement and Pro-6000-resident draft. q4 requires history; q1
bootstrap requires no history. Other windows, batched rows, TP, exports from
individual blocks, conversion/capture, LoRA and incompatible attachment/layouts
fall back. The MTP component itself always stays stock. Missing extension MTP
support also falls back. The ordinary and MTP scratch objects are independent.

The MTP parity harness runs the real generator twice, stock then native, recording
target/draft head logits, token proposals, all block residuals, routing IDs and
weights, exported target state, draft hidden inputs/outputs, carried state,
logical page positions and device-boundary tensor hashes. Persistent snapshots
cover before/after rewind, before/after draft-cache repair and committed state.
Recurrent history is restricted to valid slots, and KV/indexer planes to the
valid logical prefix; complete pool entries stop at floor(length/4). Backing KV
storage is cleared between runs, but unused padding is not treated as model state.

An independent rewind oracle hashes the selected history matrix and convolution
window before rewind, then checks the restored current state and position. Test-
only forced mutation records genuine proposals first, then changes one rotating
proposal column. Natural tests require accepted and rejected drafts and full
acceptance; forced tests require all rejection depths 1,2,3. Any mismatch or
missing coverage aborts before throughput. Instrumentation is removed for timed
runs; stock job.draft_stats supplies acceptance distributions without extra
per-layer benchmark hooks. Native/fallback counts are also measured separately.

Natural 128-output-token validation passed: 5905 exact events, 78 accepted /
72 rejected drafts. Rewind depths 0/1/2/3 occurred 16/7/17/10 times in each mode;
all 50 restoration checks passed. All 2250 target q4 block calls used native
orchestration, and no draft call did.


### MTP measured results

The staged 128-token forced-proposal run passed 8695 byte-exact events. Both
modes accepted 54 and rejected 168 drafts (24.3243% acceptance), with rewind
0/1/2/3 counts 1/13/28/32 and 74 independent restoration checks. All 3330 target
q4 block calls were native; the draft remained stock. Original proposals and
mutated proposals are separately retained in the event logs.

After those two parity gates, each mode warmed the entire 1000-token span.
The measured order was A/B, B/A, A/B. No per-layer hooks or tensor hashing ran
during these measurements. Effective rate is new output tokens / engine generation
time (prefill excluded); synchronized total wall time and engine prefill time
are also retained. The older ordinary rate convention excludes the first output
token from its numerator, so MTP summary.json records that rate separately too.

| Run | Mode | Effective output t/s | Generation seconds | Total seconds | Prefill/first-token marker ms |
|---|---|---:|---:|---:|---:|
| 1 | Stock | 64.4950 | 15.5051 | 15.6720 | 164.56 |
| 2 | Native | 64.8275 | 15.4255 | 15.5901 | 162.33 |
| 3 | Native | 64.5595 | 15.4896 | 15.6549 | 162.75 |
| 4 | Stock | 64.4758 | 15.5097 | 15.6749 | 162.87 |
| 5 | Stock | 64.0417 | 15.6148 | 15.7837 | 166.57 |
| 6 | Native | 64.8796 | 15.4132 | 15.5795 | 164.01 |

Means: **64.3375 stock / 64.7556 native (+0.65%)**; medians 64.4758/64.8275.
This is effectively tied at the resolution of this experiment, not evidence of
another large speedup. Every run proposed 1062 drafts, accepted 646 and rejected
416 (60.8286% acceptance). Each had 354 verification windows: accepted counts
0/1/2/3 occurred 60/79/78/137 times, equivalently rollback depths 0/1/2/3 occurred
137/78/79/60 times. Each native run therefore executed 15930 native target block
calls. Its 1062 draft forward blocks, 294 repair blocks and initial target/draft
prefill stayed stock; native MTP-component calls were zero. A separate 64-token
instrumented pass confirmed 1305/1305 target q4 blocks native, with identical
stock/native call shapes. Completion SHA-256 for all six runs:
`29ed8c773f2719b4cc466684108b578fb5d27ed96c15e941400d76647287aa2f`.

| Run | Pro 6000 mean util % / W | 4090D mean util % / W |
|---|---:|---:|
| 1 stock | 50.27 / 172.96 | 38.20 / 78.41 |
| 2 native | 48.21 / 176.22 | 39.36 / 78.52 |
| 3 native | 48.07 / 177.95 | 37.79 / 78.26 |
| 4 stock | 45.71 / 180.19 | 37.00 / 78.25 |
| 5 stock | 48.50 / 182.20 | 36.79 / 77.96 |
| 6 native | 45.93 / 185.12 | 37.73 / 78.87 |

VRAM was 96572 MiB on physical GPU 2 and 30377 MiB on physical GPU 0 throughout
all six runs. Power caps remained 300 W and 425 W. Raw 1 Hz nvidia-smi samples,
per-run maxima and sample counts are retained; WDDM utilization does not establish
whether execution is GPU-bound. Physical GPU 1 was not used for inference.

The separate instrumented target-forward host time fell from 745.21/29 = 25.70 ms
to 494.82/29 = 17.06 ms; instrumentation itself disproportionately taxes Python
calls, so this is not a throughput prediction. Draft input and block host time
were nearly unchanged. The evidence supports amortization of ordinary host
boundary overhead under four-row verification, with remaining draft, transfer,
attention and generator work limiting further benefit. It does not isolate those
remaining costs well enough to justify a new draft kernel. The ordinary-path
percentage advantage does **not** stack multiplicatively with MTP in this setup.

All sessions are retained. The initial stock trace measured 64.7008 t/s; its
first-use 128-token warmup was 14.9268 t/s including compilation. Full-span warmups
and all six subsequent rates are in `ab-staged/`; no measured outlier was dropped.


### Reproducing the MTP validation

Use the existing Tabby Python environment and CUDA toolkit; build with
`util/build_native_glm_mtp.bat` (SM 8.9 + 12.0, no dependency installation or
build isolation). Extract the resulting wheel into a fresh directory before
using `--package-dir`. The benchmark harness refuses an existing output directory.
For example, in PowerShell from `D:\exllamav3`:

```powershell
$env:CUDA_VISIBLE_DEVICES = '2,0'
& D:\tabbyAPI\venv\Scripts\python.exe util/profile_glm_mtp.py `
  --model D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw `
  --package-dir native-results/glm53-mtp/wheel-stage `
  --output native-results/glm53-mtp/reproduction `
  --parity-tokens 256 --forced-tokens 128 --tokens 1000 --runs 3
& D:\tabbyAPI\venv\Scripts\python.exe util/summarize_glm_mtp.py `
  native-results/glm53-mtp/reproduction
```

`--parity-only` stops after both exact gates; `--trace-only` is deliberately a
stock-only diagnostic and does not claim a validated native benchmark. The
harness alternates the internal feature flag explicitly, regardless of its
initial environment value. Normal use selects GLM with `EXL3_NATIVE_GLM_BLOCK=1`;
`EXL3_NATIVE_QWEN_BLOCK` remains independent. Both flags are read at import time.
Use a fixed three-draft window, a single active sequence and target cache history
capacity 3. Other speculative row lengths safely use the existing stock path.
The validated placement is narrowly guarded, including draft placement on logical
GPU 0. Concurrency, TP, other splits, long contexts and alternate draft windows
are not validated by this release. q1 MTP bootstrap is guard-tested; the real
32-token-prompt MTP runs exercise q4 verification, while ordinary regression
exercises the original one-row native math.


### Transfer-counter diagnostic

The first packaged 256-token pass matched all 11481 tensor/state events but
failed the newly added all-transfer-count gate. A separate 32-token diagnostic
isolated exactly two stock-only CPU-to-GPU uploads of a 248-element int64 index
vector (one per GPU). GPU-to-GPU counts were identical. These are the process-wide
`buffered_aranges` cache: stock ran first and populated it, native reused it.
The harness now clears that index cache before each parity run, preserving the
full transfer equality assertion. No inference code or wheel changed. The failed
pass (`wheel-parity/`) and diagnostic (`transfer-diagnostic/`, including both
count dictionaries) are retained with the corrected run (`wheel-validated/`).
The failed gate launched no benchmark. Guard tests also passed: 8 original GLM,
8 MTP-specific and 29 Qwen cases (45 distinct tests).


The benchmark was run from the development checkout before the final validation
commit, so its environment.json revision identifies the protected starting
commit. `wheel-source-hashes.json` verifies every packaged Python source against
the tested checkout; the staged and packaged extension binaries also match
SHA-256 `d2229c5c63966dc39e451370842e46c713459ba2d1b154f62188ee367a0555b8`.
The final release manifest identifies the committed source and exact tested wheel.
Protected Qwen revision `39b6a5e994b728bc35aae89963f5739a1f500a48`, non-MTP tag
`1630fd380b83b05488fc56c594fdd25bcd0adcd5` and all five non-MTP release files were
reverified unchanged (`protected-checkpoints.json`). The installed Tabby package
has not been replaced by this experiment.


### Packaged wheel and regressions

The exact packaged wheel passed the corrected natural 256-token MTP test:
11481 exact events, 158 accepted / 136 rejected drafts (53.7415% acceptance),
98 restoration checks, and rewind 0/1/2/3 counts 29/24/25/20. The sequence crosses
a 256-token page boundary (two allocated pages). All 4410 target q4 block calls
were native and no draft block was native. Every transfer count matched stock,
including CPU uploads and GPU-to-GPU copies grouped by direction, shape and dtype.
In particular, the layer-35 q4 transfer occurred 98 times, shared-head draft
output transfers 294 times, and hidden carry/repair traffic matched exactly.
Raw events, statistics and placement are in `wheel-validated/natural/`.


Packaged forced rejection also passed: 64 output tokens, 4621 exact events,
25 accepted / 92 rejected drafts (21.3675%), 39 restoration checks, rewind
0/1/2/3 counts 1/5/14/19, and 1755/1755 target verification blocks native.
All transfer counts matched; the larger staged forced run above adds 128-token
coverage. Logs are in `wheel-validated/forced/`.

The packaged shared extension passed the existing **256-step ordinary GLM
byte-exact regression**, including logits, tokens, every block, routing, final
hidden, transfer boundary and valid persistent cache/recurrent/indexer state.
Fresh ordinary 1000-token A/B, B/A, A/B rates were:

| Pair | Stock t/s | Native t/s |
|---|---:|---:|
| 1 | 38.4842 | 46.8389 |
| 2 | 38.5160 | 47.3255 |
| 3 | 38.6382 | 47.0927 |
| Mean | **38.5461** | **47.0857** |

The ordinary gain is **22.15%**, consistent with the original staged experiment;
it was preserved. All six ordinary completions hash to
`f469da222ccd55f5fb45670c8e798d823f65a8aa0ccfc0b3bf045c4af04e0bde`.
Raw prefill/generation/total times, telemetry and profiles are retained under
`ordinary-regression/`. MTP and ordinary completions differ, so direct speed
ratios between the two protocols are workload-dependent; the stock/native
comparison *within* each protocol is exact and uses identical completions.


The packaged Qwen regression passed 128 steps across all 48 native blocks with
`EXL3_NATIVE_GLM_BLOCK` enabled: exact logits/tokens, block residuals, recurrent
and PLE state, KV/QSA state, and fallback checks. This used the existing
`Qwen3.8-Flash-Next-Uncensored-exl3-4bpw` model; no Qwen implementation changed.
Result: `qwen-wheel.json`.

### MTP checkpoint, artifacts and rollback

Validated branch `native-glm53-mtp`, tag **`glm53-native-mtp-v1`**. Release directory:
`native-results/glm53-mtp/release/`. It contains the exact tested
`exllamav3-1.5.1+nativeglm.mtp1-cp313-cp313-win_amd64.whl`, source ZIP, complete
validation ZIP, Git bundle, README and SHA256SUMS.txt (including source commit).
Wheel SHA-256:
`986c088532d1eeb81e5889dc7478789c92b5f628a4740d6ed5cb304467ab630c`.
The bundle preserves history back through both protected checkpoints. Validation
includes raw natural/forced state-event logs, all measured sessions, telemetry,
profiles, failed transfer diagnostic, guard logs, build log and wheel identities.

No package installation was performed. To opt in later, stop the inference process,
then install the preserved wheel using the existing Tabby venv pip with
`--no-deps --force-reinstall`; enable `EXL3_NATIVE_GLM_BLOCK=1` before restarting.
For immediate stock GLM behavior, set that flag to `0` and restart. For full
non-MTP package rollback, reinstall the preserved `nativeglm.decode1` wheel from
`native-results/glm53/release/`, disable MTP, and use tag
`glm53-native-nonmtp-v1` in a separate checkout if source rollback is needed.
The Qwen package remains preserved at
`native-results/wheels/exllamav3-1.5.1+nativeqwen.concurrent1-cp313-cp313-win_amd64.whl`.
Release README gives the exact commands without overwriting existing checkpoints.
