"""Runtime policy tests that do not import Torch or require a GPU/OpenMP install."""
import importlib.util
from pathlib import Path
import threading
import sys
import unittest
import warnings
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('tested_openmp', Path(__file__).resolve().parents[1]/'exllamav3/util/openmp.py')
omp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(omp)


class Runtime:
    def __init__(self):
        self.local = threading.local()
        self.sets = []
        self.gets = 0

    def kmp_get_blocktime(self):
        self.gets += 1
        return getattr(self.local, 'value', 200)

    def kmp_set_blocktime(self, value):
        self.local.value = value
        self.sets.append((threading.get_ident(), value))


class TestOpenMPPolicy(unittest.TestCase):
    def test_unset_has_no_runtime_side_effects(self):
        def forbidden():
            self.fail('unset policy discovered a runtime')
        p = omp._OpenMPPolicy(None, forbidden)
        for _ in range(5):
            self.assertFalse(p.apply())

    def test_explicit_values_and_cached_hot_path(self):
        for value in (0, 1, 50, 200, 2**31-1):
            with self.subTest(value=value):
                runtime = Runtime()
                p = omp._OpenMPPolicy(str(value), lambda: runtime)
                self.assertTrue(p.apply())
                for _ in range(100):
                    self.assertTrue(p.apply())
                self.assertEqual([v for _,v in runtime.sets], [value])
                self.assertEqual(runtime.gets, 2)
                status = p.status()
                self.assertEqual(status['actual_ms'], value)
                self.assertEqual(status['previous_ms'], 200)

    def test_readback_does_not_apply(self):
        runtime = Runtime()
        p = omp._OpenMPPolicy('0', lambda: runtime)
        self.assertEqual(p.status()['actual_ms'], 200)
        self.assertFalse(p.status()['applied_on_thread'])
        self.assertEqual(runtime.sets, [])

    def test_each_execution_thread_is_initialized(self):
        runtime = Runtime()
        p = omp._OpenMPPolicy('1', lambda: runtime)
        self.assertTrue(p.apply())
        results = []
        def worker():
            results.append((p.apply(), p.status()['actual_ms']))
            p.apply()
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(results, [(True,1)]*3)
        self.assertEqual(len(runtime.sets), 4)

    def test_unsupported_warns_once_and_never_sets(self):
        discoveries = []
        p = omp._OpenMPPolicy('50', lambda: discoveries.append(True))
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            self.assertFalse(p.apply())
            t = threading.Thread(target=p.apply);t.start();t.join()
        self.assertEqual(len(discoveries), 1)
        self.assertEqual(len(seen), 1)
        self.assertIn('ignored', str(seen[0].message))

    def test_loading_error_preserves_behavior(self):
        def unavailable():raise OSError('missing runtime')
        p = omp._OpenMPPolicy('0', unavailable)
        with self.assertWarnsRegex(RuntimeWarning, 'missing runtime'):
            self.assertFalse(p.apply())

    def test_rejected_value_restores_previous_setting(self):
        runtime = Runtime()
        def clamp(value):
            runtime.sets.append((0,value))
            runtime.local.value = 200 if value==200 else 2
        runtime.kmp_set_blocktime = clamp
        p = omp._OpenMPPolicy('1', lambda: runtime)
        with self.assertWarnsRegex(RuntimeWarning, 'restored'):
            self.assertFalse(p.apply())
        self.assertEqual(runtime.kmp_get_blocktime(), 200)
        self.assertEqual([v for _,v in runtime.sets], [1,200])

    def test_invalid_configuration_is_explicit(self):
        for value in ('', '-1', '+1', '1.5', 'nan', '0x1', '2147483648', '１'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                omp._OpenMPPolicy(value)
        self.assertEqual(omp._parse_blocktime(' 50 '), 50)

    def test_fork_resets_thread_marker_and_lock(self):
        runtime = Runtime()
        p = omp._OpenMPPolicy('0', lambda: runtime)
        p.apply();p._after_fork();p.apply()
        self.assertEqual(len(runtime.sets), 2)

    def test_unix_missing_kmp_symbols_is_unsupported(self):
        with patch.object(omp.sys,'platform','linux'), patch.object(omp.ctypes,'CDLL',return_value=object()) as load:
            self.assertIsNone(omp._loaded_runtime())
        load.assert_called_once_with(None)

    def test_windows_does_not_load_absent_runtime(self):
        class Function:
            def __call__(self, name):return None
        class Kernel:
            GetModuleHandleW=Function()
        with patch.object(omp.sys,'platform','win32'), patch.object(omp.ctypes,'WinDLL',return_value=Kernel(),create=True), \
             patch.object(omp.ctypes,'CDLL') as load:
            self.assertIsNone(omp._loaded_runtime())
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
