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
