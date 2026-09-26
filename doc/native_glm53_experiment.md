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


## Concurrency phase: ordinary trace and boundary

Protected start `glm53-native-mtp-v1` (`e445f4d`), development branch
`native-glm53-concurrency`. Stock execution traced before implementation changes
using the protected wheel, same physical GPUs 2,0 and 90/44 GiB target budgets.
The target's 35/10 layer split remains unchanged. Raw trace is under
`native-results/glm53-concurrency/trace-ordinary.json`.

Observed ordinary decode residuals are (B,1,4,4096), B=2,3,4. Prompt prefills
remain independent B=1 calls. The generator compacts active rows only along B;
its recurrent-state list and block-table/cache-length rows follow that order.
Observed membership [0,1,2,3] -> [1,2,3] -> [2,3,4] -> [3,4], with recurrent
slots [0,1,2,3] -> [1,2,3] -> [2,3,1] -> [3,1]. A new job thus reuses a released
slot without renumbering surviving jobs. Per-row KDA slot indices select separate
convolution/recurrent matrices; per-row block tables select MLA latent, indexer
and pooled-key pages. Physical ownership is asserted disjoint for these independent
prompts, while parity compares logically ordered page contents.

The layer-35 transfer carries (B,1,4,4096) rather than introducing one transfer
per job. Batch changes alter its byte size, not its one-per-target-forward count.
The native mHC boundary has no cache or recurrent-state pointers. Its original
scratch interpreted all rows as query length, so it cannot simply accept a larger
batch. The extension now keeps explicit batch and query dimensions and allocates
R=B*Q row-local mix/norm scratch. Existing kernels flatten those two dimensions
only inside mHC/norm/residual operations, with no cross-row reduction. Returned
attention/MLP inputs retain (B,Q,4096), and stateful sublayers stay untouched.

Ordinary native B=2,3,4 requires one distinct, non-exported recurrent slot per row
from the same zero-history cache. Scratch is per block, device and batch shape;
load/unload clears it. Original B=1 and single-job MTP caches remain separate.
The generator serializes all calls; this does not support simultaneous direct
model calls from separate threads. Concurrent MTP remains guarded out pending
ordinary parity and an actual MTP trace. No scheduling or kernel math changed.


The ordinary parity harness observes each committed job state on every iteration,
all block residuals, routing, head logits/hidden state, physical ownership
invariants and logical block tables/cache lengths. It hashes RNG state per job;
the stochastic case uses DefaultSampler (min-p 0.08, temperature 0.8) with seeds
1234+i, in addition to greedy cohorts. Model prefill/forward, per-job rewind and
release wrappers assert untouched jobs retain byte-identical persistent state.
New zero-position target states are checked directly for zero current KDA matrices
and convolution windows in the reuse/final harness. Unwritten KV padding is ignored;
written speculative prefixes and valid pooled keys are compared in logical order.

A first B=2 diagnostic matched all 3984 recorded tensor/state events but failed
CPU-upload counts because stock-first execution populated the process-wide
recurrent-slot tensor cache. GPU-to-GPU counts matched. The harness now resets
`buffered_aranges`, `_slot_tensors` and rectangular `_block_tables` before both
passes. The failed diagnostic remains in `ordinary-transfer-diagnostic/`; the
corrected B=2/3/4 and stochastic tests pass transfer equality as well as exact
state parity. Staggered membership is [0,1,2,3] -> [1,2,3] -> [2,3] -> [2,3,4]
-> [3,4] -> [3], exercising 4 -> 3 -> 2 -> 3 -> 2 -> 1 and recurrent-slot reuse.
The roomy cache does not force physical page reuse; a separate four-page run does.


### Ordinary concurrency gate (first build)

| Case | Exact events | Slot reuse | Page reuse |
|---|---:|---:|---:|
| ordinary-2 | 3944 | 0 | 0 |
| ordinary-3 | 5540 | 0 | 0 |
| ordinary-4 | 7136 | 0 | 0 |
| ordinary-stagger | 8732 | 1 | 0 |
| ordinary-stochastic | 4696 | 0 | 0 |
| ordinary-page-crossing | 25752 | 0 | 0 |
| ordinary-cancel | 7980 | 1 | 0 |

All transfer groups match exactly. Every ordinary q1 target block for B=1..4
uses native orchestration when enabled; prefill remains stock. The existing
single-job 256-step regression passed all 45 blocks and persistent state checks.
In a separate 1024-token/four-page cache, cancellation also passed 7985 events,
reassigned one physical page and one recurrent slot, and checked zero initial
current KDA/convolution state for all five jobs including the replacement.
These gates precede concurrent MTP tracing or implementation changes.


### Concurrent MTP trace (after the ordinary gates)

Stock traces for two and three jobs observe target (B,4,4,4096), draft (B,1,4096),
and serial per-job repair (1,1..3,4096). For the three-job trace, target verification
changes B=3 -> 2 -> 1. The draft processes the same compact cohort jointly for
three q1 forwards, using concatenated per-job carried hidden states. Verification
then handles each job's accepted prefix separately, rewinds that job's KDA slot,
retreats its page/sequence positions, repairs its draft prefix and clones its own
accepted target hidden slice. The raw traces include independently divergent
rewinds: job 0 [2,2,0], job 1 [2,1,1,0], job 2 [1,3,0,0,0]. State isolation and
independent history-restoration assertions passed in these stock traces.

Target q4 transfer is (B,4,4,4096) across layer 35. Draft/head carry transfers
are (B,1,4096) in both GPU directions, while repair transfers remain per-job
(1,A,4096). Draft placement remains logical 0; target split remains 35/10.
The stock mHC kernels already operate independently over R=B*Q rows, with no
hard restriction at R=4; R=8 and 12 use the same row-local operations. KDA's
existing native sublayer supports B<=8, Q<=16 with slot-indexed history. Therefore
the smallest valid extension is shape-specific target scratch for B=2/3,Q=4,
not a new draft boundary or recurrent kernel. The plain residual MTP layer stays
stock. B=4 MTP and unvalidated windows will remain fallback-only.


### Concurrent MTP implementation and first gates

The native target constructor now accepts Q=1 with B=1..4 and Q=4 with B=1..3.
The attached-MTP guard separately requires the fixed three-draft protocol,
history capacity 3, matching export key, correct placement and distinct local
recurrent slots for all B rows. Scratch keys retain the original integer Q for
B=1 and use (B,Q) for B>1, preserving single-job scratch ownership. All prefills,
draft forwards and accepted-prefix draft repairs remain stock. B=4 MTP, other
verification windows, TP, unsupported exports/LoRA and incompatible layouts fall
back. Multi-job q1 bootstrap is guard-tested and shares the ordinary row math;
the runtime MTP cohorts exercise q4 verification after prompt prefill.

The harness clips draft snapshots to min(target committed position, last written
draft prefix), and separately records written speculative prefixes after each
model call. Thus unwritten verification suffixes are not mistaken for valid draft
state. Before each rewind it also computes an independent history-selection oracle,
then checks the restored current KDA matrix/convolution window and position.
Other jobs must remain byte-identical across rewind, prefill/repair and release.
At least one shared verification cycle must take different rollback depths across
jobs; different output lengths alone cannot satisfy the divergence requirement.

| Case | Exact events | Shared cycles with divergent rewinds |
|---|---:|---:|
| Two natural MTP jobs | 3000 | 9 |
| Three natural MTP jobs | 3565 | 13 |
| Three-job forced rejection + arrival | 6256 | 28 |

All three passed grouped transfer equality. The two-job case used 495 native B2q4
and 450 native B1q4 block calls. The three-job case used 405 B3q4, 225 B2q4 and
450 B1q4 native block calls. Draft native calls were zero. Forced mutation affects
only even job IDs; original proposals are recorded before mutation. Forced-run
accepted/rejected totals by job were 9/36, 26/16, 29/100 and 16/32. Every job
covered rewind depths 0,1,2,3; the late arrival reused a recurrent slot. Full raw
proposals, logits, routing, state, carry, repair and boundary tensors are retained
as exact hashes in the corresponding event logs.


Concurrent MTP boundary parity passed 4258 events and 14 divergent-rewind cycles.
It uses two independent 224-token prompts and 64/88 outputs, and explicitly
asserts that both jobs pass position 256 while still active together. A longer
short-prompt attempt was stopped after 72 stock rounds to avoid redundant
pre-boundary hashing; its metadata/telemetry are retained in
`mtp-boundary-interrupted/`. No benchmark ran in that attempt.

MTP cancellation passed 8095 events, 25 divergent cycles and one slot reassignment.
The final three-page (768-token capacity) cancellation/reuse run passed 5014 events,
18 divergent cycles, one slot reassignment and one physical page reassignment.
It checks fresh zero recurrent/convolution state and empty carried hidden state
for the replacement. Both runs assert other jobs retain byte-identical state
across release and repair; transfer groups match exactly. The tiny-cache run uses
shorter output lengths because it targets immediate reuse, not long generation.

Final wheel SHA-256:
`283885767a606dcf9a8e22a7c805b0207ca77334b66c81faba1be0eb966bc5bc`.
All packaged Python sources match the tested checkout. The first ordinary-only
build is retained separately in `ordinary-wheels/` and `ordinary-wheel-stage/`;
its identities and build log are saved. Subsequent production changes only
extended the MTP shape capability/guard, without changing ordinary row math.
Original ordinary session metadata and stagger/regression records are preserved
in `ordinary-first-build-session/` before final-wheel regression/benchmarking.
The final wheel rechecks the full changing-membership ordinary cohort (B4 through
B1) and single-job 256-step parity. Other first-build ordinary proofs identify
their tested wheel SHA explicitly.

### Final-wheel regressions and benchmark method

The final changing-membership ordinary gate passed 8737 exact events, including
fresh-state initialization checks added after the initial ordinary run. Its
native block counts were B4:1080, B3:1305, B2:990 and B1:585; 225 prefill
blocks stayed stock. Single-job ordinary passed 256 exact steps across all 45
blocks. Natural single-job MTP passed 5905 events (128 outputs, 78 accepted and
72 rejected drafts); forced MTP passed 4621 events (64 outputs, 25 accepted and
92 rejected). Both exercised rollback depths 0 through 3. Qwen passed 128 exact
steps across 48 native blocks, including residuals, recurrent/PLE and KV/QSA
state, with its fallback checks and the GLM flag enabled. Guard suites passed
19 + 16 + 29 tests (56 distinct tests after accounting for imported duplicates).

Benchmarks use the final wheel, the same 90/44 GiB placement budgets and physical
GPUs 2,0. Ordinary and MTP each load separately. Cache capacity is 5120 tokens,
with four ordinary slots/history 0 or three MTP slots/history 3. Multi-job prompts
are independent; seeds are 1234 onward. Outputs are 384/512/640/768 tokens for
ordinary cohorts, 384/512/640 for MTP, plus a 256-token late arrival in each
staggered workload. Single-job runs use the protected prompt and 1000 outputs.
No stop tokens truncate these workloads. MTP always drafts exactly three tokens.

Each workload receives a full-length stock and native diagnostic warmup, then
three alternating pairs in stock/native, native/stock, stock/native order. Timed
runs omit state hashing and per-layer/transfer hooks. Call and transfer counts
come from the separate warmups, whose shape counts, output IDs, RNG states and
accept/reject counts must match the timed runs. Every measured repeat is retained.
GPU telemetry is sampled at approximately 1 Hz on both physical devices.

Aggregate delivered throughput includes initial prefill and the complete workload
wall time. Per-job effective throughput and completion latency start at that
job's enqueue time; late arrivals therefore use their own arrival time. Maximum
delivery gaps measure intervals between generator deliveries, which can contain
several accepted MTP tokens. Arrival-induced gaps are the delivery intervals
straddling enqueue of the late job. They are not isolated prefill timings.
Diagnostic forward/draft/transfer host times overlap and may include waits for
earlier GPU work; they must not be summed or interpreted as isolated GPU costs.

This validates one serialized dynamic generator sharing a model. It does not
make the model's mutable scratch thread-safe for simultaneous forwards from
separate generators/threads, and it does not change Tabby's scheduling policy.
Long contexts, alternate placements, B>4 ordinary, B>3 MTP, other draft windows,
TP, LoRA and unsupported exports remain outside this validation/fast path.

### Measured concurrency performance

All rates below are delivered output tokens/s. Each cell listing three
runs preserves chronological order within that mode; no repeats are removed.

| Workload | Stock runs | Native runs | Stock mean | Native mean | Change |
|---|---|---|---:|---:|---:|
| ordinary-1 | 28.79, 28.00, 27.25 | 40.38, 38.70, 38.04 | 28.01 | 39.04 | +39.36% |
| ordinary-2 | 29.80, 29.45, 29.27 | 36.42, 35.78, 34.26 | 29.51 | 35.49 | +20.25% |
| ordinary-3 | 39.13, 39.62, 40.42 | 45.95, 48.49, 46.37 | 39.73 | 46.94 | +18.15% |
| ordinary-4 | 49.05, 50.34, 50.37 | 56.71, 56.92, 57.27 | 49.92 | 56.96 | +14.11% |
| ordinary-4-stagger | 49.42, 50.00, 47.21 | 57.69, 57.86, 59.14 | 48.88 | 58.23 | +19.14% |
| mtp-1 | 63.82, 64.02, 64.02 | 64.07, 64.05, 64.06 | 63.95 | 64.06 | +0.17% |
| mtp-2 | 68.57, 69.27, 69.39 | 70.85, 70.52, 71.28 | 69.08 | 70.88 | +2.61% |
| mtp-3 | 47.38, 48.58, 48.42 | 49.20, 50.40, 49.95 | 48.13 | 49.85 | +3.58% |
| mtp-3-stagger | 45.79, 43.56, 47.02 | 48.38, 47.90, 47.15 | 45.45 | 47.81 | +5.18% |

Per-job values below are three-run means, written **stock / native**.
Latencies and gaps are seconds. Job IDs correspond to the independent
prompts in `util/check_glm_concurrency.py`; the last ID in staggered
workloads is the new arrival.

| Workload/job | Tokens | Effective t/s | First token | Completion | Maximum gap | Arrival gap |
|---|---:|---:|---:|---:|---:|---:|
| ordinary-1/0 | 1000 | 28.011 / 39.037 | 0.239 / 0.224 | 35.719 / 25.633 | 0.043 / 0.030 | 0.000 / 0.000 |
| ordinary-2/0 | 384 | 14.969 / 17.707 | 0.472 / 0.406 | 25.655 / 21.710 | 0.083 / 0.071 | 0.000 / 0.000 |
| ordinary-2/1 | 512 | 16.863 / 20.278 | 0.472 / 0.406 | 30.364 / 25.266 | 0.083 / 0.071 | 0.000 / 0.000 |
| ordinary-3/0 | 384 | 14.703 / 16.759 | 0.675 / 0.699 | 26.131 / 22.938 | 0.094 / 0.082 | 0.000 / 0.000 |
| ordinary-3/1 | 512 | 15.122 / 17.330 | 0.675 / 0.699 | 33.866 / 29.577 | 0.094 / 0.082 | 0.000 / 0.000 |
| ordinary-3/2 | 640 | 16.553 / 19.558 | 0.675 / 0.699 | 38.671 / 32.742 | 0.094 / 0.082 | 0.000 / 0.000 |
| ordinary-4/0 | 384 | 14.770 / 16.114 | 0.806 / 0.681 | 26.014 / 23.831 | 0.155 / 0.140 | 0.000 / 0.000 |
| ordinary-4/1 | 512 | 15.115 / 16.526 | 0.806 / 0.681 | 33.899 / 30.983 | 0.094 / 0.084 | 0.000 / 0.000 |
| ordinary-4/2 | 640 | 15.328 / 17.179 | 0.806 / 0.681 | 41.761 / 37.256 | 0.094 / 0.084 | 0.000 / 0.000 |
| ordinary-4/3 | 768 | 16.641 / 18.988 | 0.806 / 0.681 | 46.159 / 40.447 | 0.094 / 0.084 | 0.000 / 0.000 |
| ordinary-4-stagger/0 | 384 | 14.482 / 16.422 | 0.791 / 0.749 | 26.519 / 23.401 | 0.161 / 0.137 | 0.000 / 0.000 |
| ordinary-4-stagger/1 | 512 | 14.852 / 16.890 | 0.791 / 0.749 | 34.477 / 30.330 | 0.095 / 0.084 | 0.000 / 0.000 |
| ordinary-4-stagger/2 | 640 | 14.891 / 17.055 | 0.791 / 0.749 | 42.983 / 37.537 | 0.273 / 0.284 | 0.273 / 0.284 |
| ordinary-4-stagger/3 | 768 | 14.688 / 17.499 | 0.791 / 0.749 | 52.320 / 43.894 | 0.364 / 0.284 | 0.273 / 0.284 |
| ordinary-4-stagger/4 | 256 | 14.515 / 18.957 | 0.273 / 0.284 | 17.752 / 13.510 | 0.229 / 0.069 | 0.000 / 0.000 |
| mtp-1/0 | 1000 | 63.955 / 64.062 | 0.214 / 0.210 | 15.636 / 15.610 | 0.085 / 0.086 | 0.000 / 0.000 |
| mtp-2/0 | 384 | 34.133 / 35.164 | 0.408 / 0.398 | 11.250 / 10.921 | 0.096 / 0.092 | 0.000 / 0.000 |
| mtp-2/1 | 512 | 39.474 / 40.506 | 0.408 / 0.398 | 12.971 / 12.640 | 0.096 / 0.092 | 0.000 / 0.000 |
| mtp-3/0 | 384 | 16.857 / 17.614 | 0.625 / 0.623 | 22.787 / 21.805 | 0.220 / 0.196 | 0.000 / 0.000 |
| mtp-3/1 | 512 | 19.367 / 20.189 | 0.625 / 0.623 | 26.442 / 25.364 | 0.220 / 0.196 | 0.000 / 0.000 |
| mtp-3/2 | 640 | 20.053 / 20.771 | 0.625 / 0.623 | 31.920 / 30.816 | 0.220 / 0.196 | 0.000 / 0.000 |
| mtp-3-stagger/0 | 384 | 16.382 / 17.498 | 0.661 / 0.648 | 23.474 / 21.951 | 0.228 / 0.199 | 0.000 / 0.000 |
| mtp-3-stagger/1 | 512 | 16.545 / 17.470 | 0.661 / 0.648 | 30.985 / 29.313 | 0.356 / 0.357 | 0.356 / 0.357 |
| mtp-3-stagger/2 | 640 | 16.234 / 17.075 | 0.661 / 0.648 | 39.463 / 37.486 | 0.356 / 0.357 | 0.356 / 0.357 |
| mtp-3-stagger/3 | 256 | 17.723 / 18.251 | 0.356 / 0.357 | 14.458 / 14.027 | 0.196 / 0.181 | 0.000 / 0.000 |

MTP proposal outcomes are identical in all six measured repeats per
workload and in matching diagnostic runs. Counts below are per run.

| Workload/job | Proposed | Accepted | Rejected | Acceptance |
|---|---:|---:|---:|---:|
| mtp-1/0 | 1062 | 646 | 416 | 60.83% |
| mtp-2/0 | 387 | 255 | 132 | 65.89% |
| mtp-2/1 | 501 | 345 | 156 | 68.86% |
| mtp-3/0 | 375 | 259 | 116 | 69.07% |
| mtp-3/1 | 501 | 345 | 156 | 68.86% |
| mtp-3/2 | 873 | 349 | 524 | 39.98% |
| mtp-3-stagger/0 | 375 | 259 | 116 | 69.07% |
| mtp-3-stagger/1 | 504 | 344 | 160 | 68.25% |
| mtp-3-stagger/2 | 834 | 362 | 472 | 43.41% |
| mtp-3-stagger/3 | 369 | 133 | 236 | 36.04% |

Native target calls by actual batch/query shape follow. Each count is
from one matching diagnostic cohort. Fallback includes prefill and,
for MTP, draft generation/repair. All stock blocks are fallback.

| Workload | Native target calls by shape | Native total | Fallback total |
|---|---|---:|---:|
| ordinary-1 | b1/q1:45000 | 45000 | 45 |
| ordinary-2 | b2/q1:17280, b1/q1:5760 | 23040 | 90 |
| ordinary-3 | b3/q1:17280, b2/q1:5760, b1/q1:5760 | 28800 | 135 |
| ordinary-4 | b4/q1:17280, b3/q1:5760, b2/q1:5760, b1/q1:5760 | 34560 | 180 |
| ordinary-4-stagger | b4/q1:17280, b3/q1:11385, b2/q1:5895, b1/q1:135 | 34695 | 225 |
| mtp-1 | b1/q4:15930 | 15930 | 1402 |
| mtp-2 | b2/q4:5805, b1/q4:1710 | 7515 | 859 |
| mtp-3 | b3/q4:5625, b2/q4:1890, b1/q4:5580 | 13095 | 1472 |
| mtp-3-stagger | b3/q4:7425, b2/q4:3870, b1/q4:1215 | 12510 | 1563 |

Telemetry below gives stock/native mean utilization and power; VRAM is
the maximum observed MiB across either mode. Device 2 is the Pro 6000
(300 W cap), device 0 the 4090D (425 W cap). Means average the three
per-run means, avoiding weighting slower runs more heavily.

| Workload | GPU | Utilization % stock/native | Power W stock/native | Peak VRAM MiB |
|---|---:|---:|---:|---:|
| ordinary-1 | 2 | 33.7 / 46.0 | 139.3 / 165.7 | 92764 |
| ordinary-1 | 0 | 34.2 / 37.4 | 56.2 / 65.3 | 30401 |
| ordinary-2 | 2 | 26.2 / 30.2 | 123.0 / 134.5 | 92764 |
| ordinary-2 | 0 | 34.2 / 32.9 | 51.9 / 55.1 | 30401 |
| ordinary-3 | 2 | 30.1 / 34.6 | 137.2 / 149.5 | 92764 |
| ordinary-3 | 0 | 32.8 / 35.0 | 55.4 / 59.0 | 30401 |
| ordinary-4 | 2 | 32.5 / 36.1 | 148.9 / 159.4 | 92764 |
| ordinary-4 | 0 | 34.2 / 34.1 | 58.5 / 61.8 | 30401 |
| ordinary-4-stagger | 2 | 30.9 / 37.0 | 143.1 / 158.4 | 92764 |
| ordinary-4-stagger | 0 | 33.5 / 35.4 | 57.3 / 62.5 | 30401 |
| mtp-1 | 2 | 47.4 / 48.2 | 187.4 / 187.9 | 96686 |
| mtp-1 | 0 | 35.2 / 36.7 | 78.0 / 78.2 | 30509 |
| mtp-2 | 2 | 37.9 / 36.8 | 168.0 / 170.3 | 96692 |
| mtp-2 | 0 | 34.2 / 32.6 | 67.1 / 67.6 | 30515 |
| mtp-3 | 2 | 30.4 / 31.6 | 142.0 / 145.6 | 96706 |
| mtp-3 | 0 | 35.9 / 33.3 | 58.0 / 58.8 | 30523 |
| mtp-3-stagger | 2 | 29.8 / 28.5 | 135.8 / 140.3 | 96706 |
| mtp-3-stagger | 0 | 33.5 / 35.3 | 55.4 / 56.6 | 30523 |

Inclusive diagnostic host seconds (stock/native) are from separate
instrumented warmups, not the throughput runs. Nested categories overlap;
transfer calls may wait for earlier GPU work. These are not GPU profiles.

| Workload | Target forward | Draft forward | Generator verify | Generator draft | GPU transfer calls |
|---|---:|---:|---:|---:|---:|
| ordinary-1 | 28.88 / 18.81 | 0.00 / 0.00 | 36.96 / 25.85 | 0.00 / 0.00 | 0.27 / 0.27 |
| ordinary-2 | 25.04 / 21.15 | 0.00 / 0.00 | 29.59 / 25.50 | 0.00 / 0.00 | 0.13 / 0.14 |
| ordinary-3 | 33.31 / 26.63 | 0.00 / 0.00 | 40.01 / 32.53 | 0.00 / 0.00 | 0.15 / 0.15 |
| ordinary-4 | 39.30 / 33.76 | 0.00 / 0.00 | 46.87 / 41.21 | 0.00 / 0.00 | 0.16 / 0.18 |
| ordinary-4-stagger | 44.45 / 37.41 | 0.00 / 0.00 | 51.85 / 44.71 | 0.00 / 0.00 | 0.15 / 0.18 |
| mtp-1 | 8.51 / 5.79 | 1.09 / 1.05 | 12.92 / 12.98 | 2.58 / 2.58 | 0.42 / 0.41 |
| mtp-2 | 9.45 / 8.79 | 1.10 / 1.08 | 10.87 / 10.44 | 2.04 / 1.99 | 0.21 / 0.20 |
| mtp-3 | 25.79 / 23.32 | 1.95 / 1.93 | 29.28 / 27.45 | 3.82 / 3.68 | 0.41 / 0.39 |
| mtp-3-stagger | 32.15 / 30.63 | 2.34 / 2.39 | 34.75 / 33.27 | 4.35 / 4.24 | 0.41 / 0.41 |

### Interpretation, fairness and retained variation

Ordinary batching retains a useful native advantage: +20.25% at B2, +18.15%
at B3, +14.11% at B4 and +19.14% with arrivals. Every job's mean effective
throughput improved. Initial cohorts finished in output-length order in all 54
timed runs, and every arrival advanced and completed. No starvation or native
latency spike was observed in these workloads. The maximum individual delivery
gap across ordinary runs was 0.525 s stock versus 0.302 s native; MTP maxima were
0.373 s versus 0.367 s. The 0.525 s stock gap and slower stock staggered repeats
are retained, not excluded as outliers. Mean arrival gaps were 0.273/0.284 s
ordinary and 0.356/0.357 s MTP (stock/native), so native does not remove the
prefill interruption. Ordinary B3 first-token latency was also slightly higher
native (0.699 versus 0.675 s), despite better completion times.

Concurrent MTP gains are small: +2.61%, +3.58% and +5.18% for B2, B3 and
staggered workloads. Single-job MTP is effectively unchanged (+0.17%). These
are three-pair observations, not a statistical guarantee. The retained
43.56--47.02 stock staggered MTP spread is material relative to the measured
gain. All q4 target shapes remain native; fallback is not the explanation for
the small improvement. Diagnostic B3 generator verification takes 29.28/27.45 s
versus 3.82/3.68 s for draft generation. Verification remains the larger measured
host region. Single-job target-forward host time falls from 8.51 to 5.79 s,
yet total verification stays near 12.9 s: asynchronous work/waits and the rest
of the generator limit what this orchestration change achieves. GPU transfer
host-call time is small and counts are identical, but these measurements do
not establish isolated copy cost or a particular GPU-kernel bottleneck.

Do not interpret the cohorts as ideal scaling or compare ordinary/MTP rates
without their prompts and histories. The third MTP job accepts only 39.98% of
proposals and continues for 124 B1q4 rounds after the first two finish; that
tail contributes to B3's lower aggregate rate than B2. Cross-cohort outputs are
not asserted equal: stock batching can change numerical behavior and speculative
histories. Exact native/stock comparisons use identical cohort membership.
Historical single-job ordinary rates also used a different harness/cache setup;
the fresh 28.01/39.04 t/s pair is the concurrency-harness smoke, not a direct
speed comparison against historical runs. Protected 1000-token completion hashes
remain `f469da222ccd55f5fb45670c8e798d823f65a8aa0ccfc0b3bf045c4af04e0bde`
ordinary and `29ed8c773f2719b4cc466684108b578fb5d27ed96c15e941400d76647287aa2f`
MTP. All 30 ordinary and 24 MTP benchmark runs passed output/RNG parity.

### Reproduction and checkpoint

Run from `D:\exllamav3`, with `CUDA_VISIBLE_DEVICES=2,0`, using the existing
`D:\tabbyAPI\venv\Scripts\python.exe`. Build with
`util/build_native_glm_concurrency.bat`, extract the resulting wheel to a fresh
staging directory, and pass it via `--package-dir`. The harness switches the
native flag internally for paired validation. Engine configuration and GPU
placement are recorded in each results directory. Representative commands:

```powershell
$env:CUDA_VISIBLE_DEVICES = '2,0'
$py = 'D:\tabbyAPI\venv\Scripts\python.exe'
$stage = 'native-results/glm53-concurrency/wheel-stage'
& $py util/profile_glm_concurrency.py --suite ordinary --package-dir $stage --output native-results/glm53-concurrency-repro/ordinary --regressions --benchmark
& $py util/profile_glm_concurrency.py --suite ordinary --package-dir $stage --output native-results/glm53-concurrency-repro/reuse --cache-tokens 1024 --case ordinary-page-reuse --case ordinary-cancel
& $py util/profile_glm_concurrency.py --suite mtp --package-dir $stage --output native-results/glm53-concurrency-repro/mtp --benchmark
& $py util/profile_glm_concurrency.py --suite mtp --package-dir $stage --output native-results/glm53-concurrency-repro/mtp-reuse --cache-tokens 768 --case mtp-cancel
& $py util/summarize_glm_concurrency.py native-results/glm53-concurrency-repro/ordinary
& $py util/summarize_glm_concurrency.py native-results/glm53-concurrency-repro/mtp
```

The validated checkpoint is `glm53-native-concurrency-v1` on branch
`native-glm53-concurrency`. Release files live in
`native-results/glm53-concurrency/release`: exact tested wheel, source ZIP,
validation ZIP, Git bundle, README and SHA-256 manifest. The source checkpoint
includes later harness/documentation improvements; production Python hashes
still match the wheel, and the compiled production source was frozen at build.
Validation archives identify the initial ordinary build separately and retain
its full proofs, the final-wheel rechecks, all MTP proofs, warmups, all measured
repeats, telemetry, protected-artifact audit and build logs. MTP parity-session
telemetry was saved separately before the benchmark session reused its output
directory. Interrupted/failed diagnostic records are explicitly labeled.

No installation or Tabby configuration change was performed. This is engine
dynamic-generator validation, not an HTTP connection/load test. Enable via
`EXL3_NATIVE_GLM_BLOCK=1` before launching a process using the new wheel. For
stock fallback set it to `0` and restart. For full rollback, stop inference and
reinstall the preserved `nativeglm.mtp1` wheel from
`native-results/glm53-mtp/release` without changing dependencies, or create a
fresh worktree at `glm53-native-mtp-v1`. Exact commands are in the release README.
Both protected GLM releases were rehashed unchanged and protected GLM/Qwen
revisions were rechecked before packaging; Qwen artifacts were not modified.


## 64K deployment closure

This phase starts at protected `glm53-native-concurrency-v1`
(`c486ed70bd9e1ae4428eb07ed9231aed937d12b4`) on a new
`native-glm53-longctx` branch. The concurrency release manifest, wheel, source
archive and production Python hashes were verified before harness changes.
Earlier results remain untouched. The exact wheel SHA-256 remains
`283885767a606dcf9a8e22a7c805b0207ca77334b66c81faba1be0eb966bc5bc`;
no production inference source change or rebuild is used for this phase.

`util/check_glm_longctx.py` reuses the real dynamic-generator concurrency
harness with one active sequence and the original stock/native switch. It
constructs a deterministic token stream from numbered measurement records,
retains the model chat prefix/assistant suffix, and saves the exact token IDs.
The prompt is 65,216 tokens, leaving space below the 65,536-token deployment
ceiling for decode and the fixed-three-draft speculative window. Cache capacity
is 65,536 tokens / 256 pages, with the existing fp16 MLA/indexer/pool format.
The same physical GPUs 2,0 and 90/44 GiB budgets are used; loading asserts target
layers 0--34 on logical 0 and 35--44 plus head on logical 1. One recurrent slot
is provisioned, with history 0 ordinary and 3 MTP. Draft placement is asserted
on logical 0. No alternative split is accepted if loading changes placement.

Intermediate stock prefill block hashes and growing cache-prefix hashes are
omitted. The complete final valid prefix is hashed, followed by complete valid
state throughout decode. Existing residual, routing, hidden/logit, recurrent,
convolution, logical cache/page, RNG and transfer checks are retained. The
sparse observer additionally records the ordered selected-token index hash for
every MLA layer and query row, visible length, selected count and incomplete
pool-tail count, and asserts causal bounds. Thus equality of stored indexer
keys alone cannot hide different sparse selections. Helpers reset the established
arange, recurrent-slot and block-table caches for transfer-count comparisons.

For MTP, optional harness-only deep snapshots add complete state immediately
before verification and after draft-prefix repair, plus explicit draft hidden
inputs. Existing before/after-rewind snapshots, the independent history-selection
oracle, committed state, written speculative prefixes and carried hidden state
remain in use. The new option defaults off for existing short-context harness
users. Logical pooled validity is floor(position / 4); invalid suffix/padding is
excluded. Sparse selection statics are transient and shared between serial
layers, so they are captured immediately at each attention return.

The first load proved the 64K cache fits at the required placement, then stopped
before generation on JSON serialization of a NumPy int64 storage-size value.
The harness now converts it to a Python int. That diagnostic is retained in
`native-results/glm53-longctx/metadata-diagnostic*`; it was not an inference or
parity failure.

### 64K ordinary result

Passed 27,133 exact event records and 2,816 sparse-selection records (11 MLA
layers x 256 decode steps). Prompt 65,216 + 256 outputs = 65,472 sequence tokens;
the final committed cache/recurrent position is 65,471 because the last emitted
token has not yet been fed back. Decode crosses position 65,280, growing from
255 to 256 logical pages. All 34 KDA layers and all 11 complete valid MLA latent,
indexer and pooled-key planes match, including current recurrent matrices and
valid convolution windows. Sparse widths are 2,048/2,049/2,050/2,051, with each
tail size observed 704 times, and their ordered selection hashes match exactly.

All 11,520 q1 target block calls are native. The 1,485 prefill block calls remain
stock (1,395 q2048, 45 q1536, 45 q191). The real generator's default prefill
chunk is 2048; model loading used its existing 256-token measuring chunk.
Reported B0 iterations are generator bookkeeping while prefill is incomplete,
not model forwards. Every transfer group matches by source/destination, shape
and dtype. A memory sample during the gate was 94,078 MiB on physical GPU 2 and
32,073 MiB on physical GPU 0; the original split was unchanged.

One post-parity stock/native smoke pair, without state hashing, produced
42.058 / 50.528 decode tokens/s (+20.14%). These are observational single samples,
not another performance campaign; both completions equal the parity run. Prompt
ingestion is excluded from the reported job decode rate. The short-context
concurrency campaign remains the primary performance evidence.

The first MTP observation attempt was deliberately interrupted during stock
prefill after confirming that the generator uses target `forward` (to export
draft hidden states), not target `prefill`, for this phase. The observer initially
treated those calls as decode. Both entry points now mark prompt ingestion for
observation purposes; inference calls and inputs are unchanged. The interrupted
log, metadata, placement and prompt IDs are retained in `mtp-prefill-diagnostic*`.
This was an observer-coverage correction, not an observed stock/native mismatch.

The optional two-job long-context test is omitted. The validated cache has 256
pages; two independent 65,216-token prompts alone would require 510 pages.
MTP already uses about 97,076 MiB on the 97,887 MiB Pro 6000 in a decode-state
sample, leaving limited headroom. This phase does not expand cache provisioning
or alter placement to engineer optional concurrency. Short-context concurrency
remains protected; concurrent independent near-64K sequences are not certified.

### 64K MTP result

Passed 8,775 exact events and 595 sparse-selection events comprising 1,968 query
rows across target and draft. The same 65,216-token prompt plus 128 generated
tokens gives 65,344 sequence tokens / final committed position 65,343, crossing
65,280 and occupying 256 logical pages. All 34 target KDA layers, 11 target
MLA/indexer/pool layers and the draft MLA/indexer/pool layer match in their
complete logically valid ranges. Transient written speculative prefixes and
subsequent valid overwrites also match. Sparse selections include all pool-tail
sizes, and target/draft hidden inputs, exports, logits and carried state agree.

There are 40 real q4 verification rounds, 120 proposals, 88 accepted drafts and
32 rejected drafts (73.33% acceptance). Rollback depths 0/1/2/3 occur
25/5/6/4 times respectively. All 40 before/after-rewind comparisons and independent
history-selection oracle checks pass. There are 35 accepted-prefix draft repairs
(q1:6, q2:5, q3:24), each followed by the added full-state snapshot, plus 33
initial draft-prefill calls. The 72 before-verification/commit events include 32
prefill-only generator iterations; all 40 actual verification rounds receive
those snapshots. Natural rejection already covers every depth, so no additional
forced long-context run is necessary.

All 1,800 q4 target block calls are native. Fallback totals 1,673: 1,485 target
prefill blocks plus 188 draft blocks (31 q2048, one q1536, one q191, 126 q1,
five q2 and 24 q3). Draft q1 counts include 120 proposal forwards and six repairs.
All transfer groups match exactly, including both GPU directions and CPU uploads;
the native path introduces no extra transfers. Target/draft cache capacity and
placement remain unchanged throughout rejection and repair.

One uninstrumented decode smoke pair produced 75.093 / 74.917 tokens/s
stock/native (-0.24%, effectively tied). Output IDs, completion hashes, RNG and
proposal/accept/reject counts match the exact runs. This shows no catastrophic
native regression at the tested context; it is not evidence of a meaningful
MTP speed difference. Prefill timing is excluded from this decode metric.

Transfer counts are identical in 18 ordinary and 31 MTP shape/dtype groups.
Aggregated only for orientation, ordinary counts are CPU->logical0:871,
CPU->logical1:582 and logical0->logical1:289; MTP counts are 787,150,193 and
logical1->logical0:188 respectively. The raw parity files retain the actual
shape/dtype grouping, including legitimate target residual and draft/carry
transfers, rather than relying on these aggregate counts alone.

Reproduce from the repository root using the existing environment and extracted
unchanged concurrency wheel (each output directory should be fresh):

```powershell
$env:CUDA_VISIBLE_DEVICES = '2,0'
$py = 'D:\tabbyAPI\venv\Scripts\python.exe'
& $py util/check_glm_longctx.py --suite ordinary --prompt-tokens 65216 --tokens 256 --output native-results/glm53-longctx-repro/ordinary --smoke
& $py util/check_glm_longctx.py --suite mtp --prompt-tokens 65216 --tokens 128 --output native-results/glm53-longctx-repro/mtp --smoke
```

Both commands default to the verified concurrency `wheel-stage` directory;
`--package-dir` can select a fresh extraction of that same wheel. Native flags
are toggled internally for comparison. `--forced` is available only as a
test-proposal mutation if a different natural prompt lacks rollback coverage.

### Final regression and deployment verdict

Focused exact regressions against the unchanged wheel passed after both long
gates: ordinary GLM 256 steps / 45 blocks; changing-membership B4/B3/B2/B1
ordinary, 8,737 events and slot reuse; natural MTP 128 outputs / 5,905 events;
forced MTP 64 outputs / 4,621 events; and Qwen 128 steps / 48 blocks, including
its recurrent/PLE, KV/QSA and fallback checks with the GLM flag enabled. No
historical benchmark campaign was repeated. Smoke output IDs, completion hashes,
RNG and speculation counts were also checked against their long exact runs.

The final protected checkpoint is `glm53-native-longctx-v1` on
`native-glm53-longctx`. Its release directory is
`native-results/glm53-longctx/release`, containing a byte-identical copy of the
concurrency wheel, source ZIP with the new harness/report, validation ZIP, Git
bundle, README/rollback instructions and SHA-256 manifest. Production source
and the prior GLM/Qwen checkpoint/release identities are rechecked during
packaging. The wheel is reused, not rebuilt under a different filename.

For the specified single-job 64K ceiling, ordinary q1 and fixed-three-draft MTP
q4 target verification are exact and fully native. KDA and long-context
MLA/DSA/indexer/pool state remain exact, including sparse selections and
speculative retreat/repair. No production inference code, dependencies, GPU
placement, installed package or Tabby configuration changed. Existing strict
feature/placement guards remain. Near-64K concurrency is outside this closure,
and the sampled MTP memory headroom is about 811 MiB on the Pro 6000. The
requested deployment validation is complete; no further optimization is pursued.
