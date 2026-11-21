# WDDM autosplit validation — 2026-11-21

The 2.02 bpw DeepSeek V4.1 Flash checkpoint loaded and generated successfully
through TabbyAPI's installed backend in both requested configurations. The
original `60,94` production split was sufficient. The fix is **opt-in**:

```powershell
$env:CUDA_VISIBLE_DEVICES = "1,2"
$env:EXL3_WDDM_MEMORY = "1"
```

Unset `EXL3_WDDM_MEMORY` or set it to `0` to compare against the original CUDA
accounting. No global environment variables or saved TabbyAPI configuration were
changed. Validation processes unloaded their models and exited.

## Git and source protection

- Branch: `fix/wddm-autosplit-accounting`.
- Base: `b52f32e38d8fa9237eb5300085679f40d0abea95`.
- Final commit and status: `native-results/wddm-autosplit/final-provenance.json`.
- All 90 preexisting modified/untracked files were backed up under
  `native-results/wddm-autosplit/preexisting`; their SHA-256 hashes were checked
  unchanged after validation. They are excluded from this change's commit.
- Product changes: `exllamav3/util/memory.py`, new `util/wddm_memory.py` inside
  that package, `exllamav3/model/model_ls.py` (diagnostic only), and `version.py`.
- Supporting files: `tests/test_wddm_memory.py`, `util/package_wddm_wheel.py`,
  `util/validate_wddm_autosplit.py`, and the two `doc/wddm_autosplit*.md` files.
- No native IR, TP, MoE, graph, architecture, MTP, quantization, or streaming
  implementation changes. No merge, rebase, push, or remote changes.

## Confirmed mechanism and correction

The current source capped additional allocations at CUDA-reported free memory,
then used that same source for subsequent transient/physical headroom checks.
During the successful 8K load, CUDA reported **39.719 GiB** free on logical GPU 1
when its cap was established. The corrected cumulative cap was **90.021 GiB**.
For 65K/MTP4, the corresponding figures were **62.426 GiB** and **94.021 GiB**.
The model subsequently reserved more than those old CUDA-derived limits and
generated successfully. This independently confirms the artificial cap problem;
the earlier standalone 80+80 GiB allocation experiment was not repeated.

With the switch enabled, the shared helper uses fresh NVML v2 physical free
memory on UUID-verified WDDM GPUs, bounded by CUDA-addressable capacity, driver
reservations, NVML used memory, and current PyTorch reservations. The allocator's
cumulative limit still adds current reservations to an additional per-load
allowance. Unused allocator reservations remain reusable in transient checks.

The same helper serves both initial caps and later autosplit headroom checks.
The 256 MiB margin, maximum measured transients, worst-case measuring forwards,
real OOM handling, and explicit user caps remain intact. The observed largest
transients with a 4096-token chunk were 1196/1193 MiB, higher than the historical
prompt's measurements; they were retained in the placement checks.

There is no mandatory Python dependency. NVML loads from Windows System32.
Missing APIs, failed queries, bad identities/counters, capacity mismatches, or
non-WDDM devices fall back to CUDA reporting. Linux and integrated GPU behavior
is unchanged. No per-token memory queries were added to the library.

## Environment and device mapping

Python: `D:\tabbyAPI\venv\Scripts\python.exe`, version 3.13.12.
PyTorch: `2.9.0+cu128`. Installed package location:
`D:\tabbyAPI\venv\Lib\site-packages\exllamav3`.

| CUDA logical device | NVIDIA-SMI index | PCI | GPU UUID |
|---|---:|---|---|
| 0 | 0 | 0000:01:00.0 | GPU-37ad53ed-68fd-d439-a896-958b21102656 |
| 1 | 2 | 0000:c5:00.0 | GPU-011f3e6c-9c4c-3e77-d37a-20a00a57bfdc |

Both are RTX PRO 6000 Blackwell Max-Q Workstation Edition, approximately 95.59
GiB CUDA-addressable each. NVML confirmed WDDM. The RTX 4090D (SMI index 1) was
never CUDA-visible to validation. Mapping uses UUIDs, never ordinal equivalence.

The existing `D:\tabbyAPI\models\DeepSeek-V4.1-Flash-2.1bpw` junction was stale.
Its previous target, `D:\text-generation-webui\user_data\models\DeepSeek-V4.1-Flash-2bpw`,
did not exist. Header-only inspection verified the matching renamed checkpoint:
`D:\text-generation-webui\user_data\models\DeepSeek-V4.1-Flash-1017-Heretic-EXL3-2bpw`.
It has 2.02 bpw metadata, 132.833343 GiB backbone, 6.868075 GiB MTP,
188.833133 GiB streamed table, and 0.872573 GiB vision/aligner tensors.
The junction now targets that directory; the old link was preserved under
`native-results/wddm-autosplit/original-model-junction`. No checkpoint file was
modified or copied. Engram disk streaming remained enabled.

## Tests and model results

| Validation | Result |
|---|---|
| Focused accounting/loader safety tests | 13 passed |
| Existing native IR/TP/graph unittest suite | 230 passed; no skips/failures |
| Existing MoE assignment/guards, config parsing, device-copy pytest tests | 157 passed |
| Small real allocations (0.25/0.5 GiB) and 32 MiB additional caps | Passed; 128 MiB requests rejected on both GPUs |
| 8K, FP16, no MTP/vision, `90,90` | Loaded in 23.30 s; two identical 32-token greedy completions |
| 65K, FP16, native DSpark MTP4, `60,94` | Loaded in 25.43 s; two identical 32-token speculative completions |

The full run used max batch 4, chunk 4096, dynamic draft disabled,
`EXL3_DSPARK_CONF=0.0`, `EXL3_NATIVE_PLAN=1`, the default enabled V4.1 native graph
policy, and `EXL3_AUTOSPLIT_WORSTCASE=1`. It populated 43 native plan cache entries
and executed target verification with shape `(1,5)`. Each request accepted 23/36
drafted tokens. The MTP component is DSpark, dispatched through the generator's
DFlash/block path; the generic AR-MTP boolean is correctly false.

All four completed requests produced identical text beginning “The capital of
France is Paris.” The raw completion continued with a correct Germany/Berlin
example because the test requested 32 tokens. No transformer layer was placed
on CPU; the normal CPU embedding module remained there. The native MoE assignment
extension is architecture-gated and correctly fell back for this checkpoint;
its execution/guard tests passed independently.

These are TabbyAPI **backend** load/generation tests against the installed wheel,
including its streaming request lifecycle; they are not HTTP endpoint tests.
Early harness attempts exposed a stale junction, missing request context,
an observation-wrapper keyword mismatch, and an incorrect AR-MTP assertion.
Those harness issues were corrected; their evidence was retained separately.
No product change was needed after the focused memory fix.

The environment did not include pytest; it was installed in the task-local
`native-results/wddm-autosplit/test-deps` directory, without changing TabbyAPI's
dependency set. An initial pip replacement met a native DLL held by the running
regression process. Installation succeeded once it exited; pip's temporary
rollback remnants were preserved in the task backup directory.

## Physical memory after generation

All figures are GiB, measured while the models were still loaded.

| Run | GPU | PyTorch allocated | PyTorch reserved | NVML used | NVML free |
|---|---:|---:|---:|---:|---:|
| 8K/no MTP | 0 | 87.536 | 90.602 | 91.273 | 3.882 |
| 8K/no MTP | 1 | 45.992 | 48.018 | 48.713 | 46.441 |
| 65K/MTP4/native IR | 0 | 65.797 | 67.824 | 68.499 | 26.655 |
| 65K/MTP4/native IR | 1 | 75.122 | 77.738 | 78.420 | 16.734 |

NVML additionally reserves about 0.438 GiB per GPU for driver/firmware use.
Minimum sampled physical free memory during the full run was 25.753/16.638 GiB.
Two-second sampling is not an exhaustive peak measurement; the loader retained
its own allocation/transient checks. After validation processes exited, SMI
reported the Blackwells back at 19/46 MiB used and 97420/97393 MiB free.

## Wheel and rollback

Installed version: `1.6.6+nativeplan.v2.tp.moe1.ds41.1.omp1.pipe1.attn1.wddm1`.

Wheel:
`D:\exllamav3\native-results\wddm-autosplit\package\wheels\exllamav3-1.6.6+nativeplan.v2.tp.moe1.ds41.1.omp1.pipe1.attn1.wddm1-cp313-cp313-win_amd64.whl`

SHA-256: `dea4f09306a79e049427e986f419d423667cd2867e1ad65523cb6213aaf5cb28`.

The wheel uses the established packaging approach for Python-only changes,
reusing the verified native binary. Every native source matched the prior
wheel; every installed Python/native payload matched the new wheel. Extension
SHA-256 remains `6b46a5ced9b406a664e9b67476de6d01df452c205afde8b970bb6a8908016604`.

Previous version: `1.6.6+nativeplan.v2.tp.moe1.ds41.1.omp1.pipe1.attn1`.
Its wheel and hash are recorded in `native-results/wddm-autosplit/package/rollback.json`.
Rollback uses the same TabbyAPI Python with `-m pip install --force-reinstall
--no-deps` and that recorded wheel, after stopping processes holding its DLL.

Raw evidence is under `native-results/wddm-autosplit`: `memory.json`,
`minimal.json`, `full-native.json`, per-run stdout/stderr and memory JSONL,
`unit-tests.log`, `regressions.json`, `additional-regressions.log`,
`checkpoint.json`, and `package/installed-proof.json`.

## Limits and follow-up

The underlying WDDM/driver reason for CUDA's cross-device free-memory figure
remains unresolved. NVML provides snapshots, not a guarantee against concurrent
allocations or Windows residency changes. Fallback remains conservative.
Linux, TCC, single-GPU, asymmetric capacities, and unavailable NVML were covered
by mocked policy tests where applicable, not additional physical machines.
No exhaustive 65K prompt or throughput stress test was performed.
Keep the opt-in switch for controlled comparisons; no broader memory-policy
or native execution changes are needed for this checkpoint's tested workload.
