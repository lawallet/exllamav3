"""Opt-in OpenMP worker wait policy, applied once on each inference thread.

The kmp setting belongs to the calling thread and the teams it forms, not all
threads in the process. Never load another OpenMP runtime to satisfy this option.
"""
import ctypes
import os
import sys
import threading
import warnings


def _parse_blocktime(value):
    if value is None:
        return None
    text = value.strip()
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError("EXL3_OMP_BLOCKTIME_MS must be a nonnegative integer number of milliseconds")
    milliseconds = int(text)
    if milliseconds > 2**31 - 1:
        raise ValueError("EXL3_OMP_BLOCKTIME_MS exceeds the OpenMP signed 32-bit integer range")
    return milliseconds


def _loaded_runtime():
    """Resolve an already loaded kmp API; no library search or second runtime."""
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        handle = kernel.GetModuleHandleW("libiomp5md.dll")
        if not handle:
            return None
        runtime = ctypes.CDLL("libiomp5md.dll", handle=handle)
    else:
        # Only symbols already exported by the process. GNU OpenMP and locally
        # scoped libraries without these symbols are safely unsupported.
        runtime = ctypes.CDLL(None)
    try:
        setter, getter = runtime.kmp_set_blocktime, runtime.kmp_get_blocktime
    except AttributeError:
        return None
    setter.argtypes, setter.restype = [ctypes.c_int], None
    getter.argtypes, getter.restype = [], ctypes.c_int
    return runtime


class _OpenMPPolicy:
    def __init__(self, value, discover=_loaded_runtime):
        self.requested = _parse_blocktime(value)
        self._discover = discover
        self._runtime = None
        self._resolved = False
        self._reason = None
        self._lock = threading.Lock()
        self._local = threading.local()

    def _resolve(self):
        if not self._resolved:
            with self._lock:
                if not self._resolved:
                    try:
                        self._runtime = self._discover()
                    except (OSError, AttributeError) as error:
                        self._reason = str(error)
                    if self._runtime is None:
                        self._reason = self._reason or "no loaded OpenMP runtime exporting the kmp block-time API"
                    self._resolved = True
        return self._runtime

    def apply(self):
        # Unset means no discovery, setter, getter or warning. The thread-local
        # fast path makes repeated forward/iterate calls avoid native API calls.
        if self.requested is None:
            return False
        if getattr(self._local, "applied", False):
            return self._local.success
        runtime = self._resolve()
        if runtime is None:
            with self._lock:
                if not getattr(self, "_warned", False):
                    warnings.warn("EXL3_OMP_BLOCKTIME_MS ignored: " + self._reason, RuntimeWarning, stacklevel=2)
                    self._warned = True
            self._local.success = False
        else:
            before = runtime.kmp_get_blocktime()
            runtime.kmp_set_blocktime(self.requested)
            actual = runtime.kmp_get_blocktime()
            self._local.before = before
            self._local.actual = actual
            self._local.success = actual == self.requested
            if not self._local.success:
                # A runtime that rejects the setting must retain its prior policy.
                runtime.kmp_set_blocktime(before)
                warnings.warn(f"EXL3_OMP_BLOCKTIME_MS={self.requested} was not accepted (read back {actual}); "
                              "restored the previous policy", RuntimeWarning, stacklevel=2)
        self._local.applied = True
        return self._local.success

    def status(self):
        """Diagnostic readback only; does not apply or change the policy."""
        runtime = self._resolve()
        return dict(requested_ms=self.requested, runtime_available=runtime is not None,
                    actual_ms=runtime.kmp_get_blocktime() if runtime is not None else None,
                    applied_on_thread=getattr(self._local, "applied", False),
                    successful=getattr(self._local, "success", False),
                    previous_ms=getattr(self._local, "before", None),
                    pid=os.getpid(), thread_id=threading.get_native_id(), reason=self._reason)

    def _after_fork(self):
        # Spawn imports a new policy. For CPU-only fork users, inherited thread
        # identities/locks must not suppress initialization or deadlock the child.
        self._local = threading.local()
        self._lock = threading.Lock()


_policy = _OpenMPPolicy(os.environ.get("EXL3_OMP_BLOCKTIME_MS"))
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_policy._after_fork)


def configure_openmp_for_current_thread():
    """Apply the import-time EXL3_OMP_BLOCKTIME_MS option once on this thread.

    Useful for custom inference/CPU workers. Does not change Torch's thread count,
    affinity, GPU settings or another thread's OpenMP policy. Returns whether the
    requested value was applied; an unset or unsupported option returns False.
    """
    return _policy.apply()


def openmp_status():
    """Read the actual policy on the current thread without changing it."""
    return _policy.status()
