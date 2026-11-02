# Optional OpenMP worker wait policy

`EXL3_OMP_BLOCKTIME_MS` requests a nonnegative integer worker wait time in
milliseconds through the loaded runtime's `kmp_set_blocktime` API. Set it before
importing ExLlamaV3. Leave it unset to preserve the runtime's existing behavior.
This option does not change Torch's thread count, CPU affinity or numerical
operations, and works with stock execution as well as native IR.

For a single PowerShell-launched process, without changing persistent settings:

```powershell
$previous = $env:EXL3_OMP_BLOCKTIME_MS
try {
    $env:EXL3_OMP_BLOCKTIME_MS = '0'
    # Launch the existing application here, using its normal arguments.
    & 'D:/tabbyAPI/venv/Scripts/python.exe' main.py
} finally {
    if ($null -eq $previous) { Remove-Item Env:EXL3_OMP_BLOCKTIME_MS -ErrorAction SilentlyContinue }
    else { $env:EXL3_OMP_BLOCKTIME_MS = $previous }
}
```

The integer range is 0 through 2147483647, matching the runtime's signed
32-bit argument. Malformed or out-of-range values raise `ValueError`; an empty
value is not equivalent to an unset variable. This library option is read once
at import. It uses the runtime API even if OpenMP was initialized earlier;
it does not assume `KMP_BLOCKTIME` has the same initialization behavior.

Block time applies to the calling thread and the teams it forms. Model loading,
direct forward/prefill, generator iteration, and tensor-parallel workers ensure
the policy is applied on their actual execution thread. Each thread calls the
setter once and verifies the getter; subsequent iterations take a cached path.
This matters when an application loads a model in a thread pool and generates
on an event loop. Spawned TP processes inherit the environment selection.

For a custom CPU/inference worker outside these entry points:

```python
from exllamav3.util.openmp import configure_openmp_for_current_thread, openmp_status

configure_openmp_for_current_thread()
print(openmp_status())
```

`openmp_status()` reads the actual current-thread value without setting it.
The result identifies the requested/actual milliseconds, previous value,
thread/process IDs, runtime availability, and whether configuration succeeded.
An external runtime setter called later is not continually overridden by the
library. Select a different process environment for a different application
policy; changing the environment after import does not reconfigure live workers.

On Windows, the implementation only uses an already loaded `libiomp5md.dll`.
On other platforms it only uses `kmp_set_blocktime`/`kmp_get_blocktime` symbols
already exported by the process. It never searches for or loads another OpenMP
runtime. GNU OpenMP or a runtime without these symbols remains unchanged, with
one warning when the option was explicitly requested. A rejected readback causes
the previous value to be restored. No option is silently substituted with a
single-threaded Torch setting.

The measured recommendation and workload limitations are maintained in
[the DeepSeek investigation](deepseek_v41_ir_investigation.md). Zero waiting can
reduce CPU contention in GPU-heavy inference, but repeated CPU parallel regions
may prefer keeping workers awake. This is an explicit per-process choice, not
an unconditional global default. CPU expert workers with their own native pools
are not automatically reconfigured by this option.

On the tested EPYC 7663 system, `0` is the low-CPU choice for GPU-heavy DeepSeek
V4.1 inference. `1` is worth selecting for applications that also perform many
short parallel CPU operations: the CPU control found a substantial wakeup-cost
regression with `0` that `1` avoided. `50` retained much of the worker spinning
in the DeepSeek workload. These measurements do not justify changing the unset
default for every model, concurrent workload, operating system or CPU runtime.

API semantics: [Intel OpenMP extension routines](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/optimization-and-programming/openmp-support/openmp-library-support/intel-compiler-extension-routines-to-openmp.html).
