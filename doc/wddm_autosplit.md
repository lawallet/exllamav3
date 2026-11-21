# Optional WDDM autosplit accounting

Set `EXL3_WDDM_MEMORY=1` **before starting TabbyAPI** to use physical framebuffer
headroom for loading on Windows WDDM NVIDIA GPUs:

```powershell
$env:CUDA_VISIBLE_DEVICES = "1,2"
$env:EXL3_WDDM_MEMORY = "1"
```

Unset `EXL3_WDDM_MEMORY`, or set it to `0`, for the original CUDA accounting.
Only the exact value `1` enables the new policy. The environment setting is
process-local; this change does not update TabbyAPI's configuration or launcher.
Verify the devices exposed by the CUDA mask on your machine before loading.

The initial allocator budget and subsequent layer-split headroom checks share
`device_mem_info()`. Some WDDM configurations report less CUDA free memory on one
GPU as another GPU fills, imposing an unnecessarily small allocator cap. With
the switch enabled, discrete WDDM devices use the driver's NVML v2 physical free
memory instead. CUDA UUIDs identify NVML devices; enumeration order is never used.
The handle's returned UUID and active WDDM driver mode are checked.

Physical free memory excludes other applications, this process's allocations,
and driver/firmware reservations. The result is additionally bounded by the
CUDA-addressable capacity minus driver reserves and the greater of NVML used
memory and PyTorch's reservation. PyTorch's unused reserved blocks are counted
as reusable only in the existing transient-headroom checks. `use_per_device`
remains an additional allowance on top of current allocator reservations.

Missing DLLs/functions, failed queries, missing UUIDs, non-WDDM devices, mismatched
capacities, and invalid counters retain CUDA accounting. Linux and integrated
GPU behavior is unchanged. NVML is loaded from Windows System32 without requiring
`pynvml`. Only the library and verified handles are cached; memory readings are
fresh. No queries are added to token generation.

Real allocation failures, worst-case measuring forwards, maximum transients,
explicit memory budgets, and the existing 256 MiB physical margin remain intact.
NVML is a physical-memory snapshot, not a guarantee of Windows residency or
future availability; concurrent applications can still exhaust memory between
measurement and allocation. Unsupported reporting conservatively falls back.

Focused tests: `python -m unittest discover -s tests -p test_wddm_memory.py -v`.
`util/validate_wddm_autosplit.py` exercises the installed wheel through TabbyAPI's
backend, recording device identities, caps, live memory, loading and generation.
Run it with the specified TabbyAPI Python and `-I`, using `memory`, `minimal`, or
`full`. For `full`, also set `EXL3_DSPARK_CONF=0.0`; set `EXL3_NATIVE_PLAN=1` to
validate the custom native IR and its default graph policy. The full validation
uses the checkpoint's DSpark MTP component (the generator's block-drafting path).
`util/package_wddm_wheel.py` preserves the tested native binary and records
source payload hashes and rollback provenance.

API references: [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html),
[NVML v2 memory counters](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlMemory__v2__t.html).
