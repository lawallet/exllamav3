"""Optional WDDM framebuffer accounting through the driver-supplied NVML DLL.

No Python NVML dependency, ordinal mapping, or cached memory readings. Failure to
identify a physical WDDM device or obtain sane v2 counters leaves CUDA in charge.
"""

import ctypes as ct
from functools import lru_cache
import sys
from uuid import UUID


class _MemoryV2(ct.Structure):
    _fields_ = [
        ("version", ct.c_uint),
        ("total", ct.c_ulonglong),
        ("reserved", ct.c_ulonglong),
        ("free", ct.c_ulonglong),
        ("used", ct.c_ulonglong),
    ]


def _check(result):
    if result != 0:
        raise OSError(f"NVML query failed ({result})")


class _NVML:
    def __init__(self):
        # LOAD_LIBRARY_SEARCH_SYSTEM32: never load a DLL from the working directory.
        self.lib = ct.CDLL("nvml.dll", winmode = 0x800)
        signatures = {
            "nvmlInit_v2": [],
            "nvmlDeviceGetHandleByUUID": [ct.c_char_p, ct.POINTER(ct.c_void_p)],
            "nvmlDeviceGetUUID": [ct.c_void_p, ct.c_char_p, ct.c_uint],
            "nvmlDeviceGetDriverModel": [ct.c_void_p, ct.POINTER(ct.c_uint), ct.POINTER(ct.c_uint)],
            "nvmlDeviceGetMemoryInfo_v2": [ct.c_void_p, ct.POINTER(_MemoryV2)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ct.c_int
        _check(self.lib.nvmlInit_v2())
        # Retain one NVML initialization for the process lifetime.

    @lru_cache(maxsize = 32)
    def handle(self, uuid):
        handle = ct.c_void_p()
        _check(self.lib.nvmlDeviceGetHandleByUUID(uuid, ct.byref(handle)))
        actual = ct.create_string_buffer(96)
        _check(self.lib.nvmlDeviceGetUUID(handle, actual, len(actual)))
        if actual.value.lower() != uuid.lower():
            raise ValueError("CUDA/NVML UUID mismatch")
        current, pending = ct.c_uint(), ct.c_uint()
        _check(self.lib.nvmlDeviceGetDriverModel(handle, ct.byref(current), ct.byref(pending)))
        if current.value != 0:  # NVML_DRIVER_WDDM
            raise ValueError("Not a WDDM device")
        return handle

    def memory(self, uuid):
        memory = _MemoryV2()
        memory.version = ct.sizeof(_MemoryV2) | (2 << 24)
        _check(self.lib.nvmlDeviceGetMemoryInfo_v2(self.handle(uuid), ct.byref(memory)))
        return memory


@lru_cache(maxsize = 1)
def _nvml():
    try:
        return _NVML()
    except (OSError, AttributeError):
        return None


def physical_free_memory(properties, cuda_total, torch_reserved):
    """Return conservative physical free bytes, or None to retain CUDA reporting.

    NVML free excludes allocations of all processes and driver/firmware reserves.
    Also bound it by CUDA-addressable capacity minus system reserves and at least
    this process's allocator reservation (including its unused cached blocks).
    That latter bound protects against lagging NVML allocation counters. Cached
    blocks are added back only by the caller's transient/reusable-memory check.
    """
    if sys.platform != "win32" or properties.is_integrated:
        return None
    try:
        # PyTorch exposes a CUDA UUID, not an NVML ordinal. Reject MIG and missing
        # identities rather than accidentally querying their parent physical GPU.
        uuid = ("GPU-" + str(UUID(str(properties.uuid).removeprefix("GPU-")))).encode("ascii")
        nvml = _nvml()
        if nvml is None:
            return None
        memory = nvml.memory(uuid)
        total = min(cuda_total, properties.total_memory)
        if total <= 0 or abs(memory.total - total) > total // 100:
            return None
        if any(value < 0 or value > memory.total for value in (memory.free, memory.used, memory.reserved)):
            return None
        if memory.free + memory.used + memory.reserved > memory.total:
            return None
        return max(0, min(memory.free, total - memory.reserved - max(memory.used, torch_reserved)))
    except (OSError, AttributeError, ValueError, TypeError):
        return None
