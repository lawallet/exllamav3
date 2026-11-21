"""CPU-only accounting tests, including the actual layer-loader rollback path."""
import ctypes as ct
from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from exllamav3.util import memory, wddm_memory as wddm
from exllamav3.model import model_ls

G = 1 << 30
UUIDS = ["37ad53ed-68fd-d439-a896-958b21102656", "011f3e6c-9c4c-3e77-d37a-20a00a57bfdc"]


def props(i=0, total=96*G):
    return NS(uuid=UUIDS[i], total_memory=total, is_integrated=False)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(wddm.sys, "platform", "win32"))
        self.stack.enter_context(patch.dict('os.environ', {'EXL3_WDDM_MEMORY': '1'}))
        self.nvml = Mock()
        self.stack.enter_context(patch.object(wddm, "_nvml", return_value=self.nvml))
        self.nvml.memory.return_value = NS(total=96*G, free=90*G, used=5*G, reserved=G)

    def test_identity_and_asymmetric_order(self):
        readings = {("GPU-"+UUIDS[1]).encode(): NS(total=48*G, free=10*G, used=37*G, reserved=G),
                    ("GPU-"+UUIDS[0]).encode(): NS(total=96*G, free=80*G, used=15*G, reserved=G)}
        self.nvml.memory.side_effect = readings.__getitem__
        self.assertEqual(wddm.physical_free_memory(props(1,48*G),48*G,0),10*G)
        self.assertEqual(wddm.physical_free_memory(props(0),96*G,0),80*G)

    def test_driver_and_uuid_checked_without_ordinals(self):
        backend = object.__new__(wddm._NVML)
        backend.lib = Mock()
        backend.lib.nvmlDeviceGetHandleByUUID.return_value = 0
        def uuid_query(handle, buffer, size):
            buffer.value = ("GPU-"+UUIDS[1]).encode()
            return 0
        backend.lib.nvmlDeviceGetUUID.side_effect = uuid_query
        backend.lib.nvmlDeviceGetDriverModel.return_value = 0
        u = ("GPU-"+UUIDS[1]).encode()
        backend.handle(u)
        backend.lib.nvmlDeviceGetHandleByUUID.assert_called_once()
        backend.lib.nvmlDeviceGetHandleByIndex.assert_not_called()
        with self.assertRaises(ValueError):
            backend.handle(("GPU-"+UUIDS[0]).encode())
        backend.handle.cache_clear()
        def tcc(handle,current,pending):
            ct.cast(current,ct.POINTER(ct.c_uint))[0] = 1
            return 0
        backend.lib.nvmlDeviceGetDriverModel.side_effect = tcc
        with self.assertRaises(ValueError):
            backend.handle(u)

    def test_lagging_counter_bounded_by_torch_reservation(self):
        self.assertEqual(wddm.physical_free_memory(props(),96*G,20*G),75*G)

    def test_cuda_capacity_is_upper_bound(self):
        self.assertEqual(wddm.physical_free_memory(props(total=95*G+G//2),95*G+G//2,0),89*G+G//2)

    def test_bad_identity_capacity_counters_and_failures(self):
        for p in (NS(is_integrated=False), NS(is_integrated=False,uuid='MIG-bad'),props(total=48*G)):
            self.assertIsNone(wddm.physical_free_memory(p,96*G,0))
        for reading in (NS(total=96*G,free=97*G,used=0,reserved=0),
                        NS(total=96*G,free=90*G,used=10*G,reserved=G)):
            self.nvml.memory.return_value=reading
            self.assertIsNone(wddm.physical_free_memory(props(),96*G,0))
        self.nvml.memory.side_effect=OSError('query failed')
        self.assertIsNone(wddm.physical_free_memory(props(),96*G,0))
        with patch.object(wddm,'_nvml',return_value=None):
            self.assertIsNone(wddm.physical_free_memory(props(),96*G,0))

    def test_non_windows_and_integrated_unchanged(self):
        with patch.object(wddm.sys,'platform','linux'):
            self.assertIsNone(wddm.physical_free_memory(props(),96*G,0))
        p=props();p.is_integrated=True
        self.assertIsNone(wddm.physical_free_memory(p,96*G,0))
        self.nvml.memory.assert_not_called()

    def test_caps_and_cross_device_discrepancy(self):
        with patch.object(torch.cuda,'mem_get_info',return_value=(40*G,96*G)), \
             patch.object(torch.cuda,'get_device_properties',return_value=props()), \
             patch.object(torch.cuda,'memory_reserved',return_value=4*G), \
             patch.object(torch.cuda,'set_per_process_memory_fraction') as cap, \
             patch.object(memory,'touch_device'):
            self.assertEqual(memory.device_mem_info(1),(90*G,96*G))
            self.assertEqual(memory.set_memory_fraction_use(90*G,1),94*G)
            self.assertEqual(memory.set_memory_fraction_use(2*G,1),6*G)
            self.assertEqual(memory.set_memory_fraction_use(120*G,1),94*G)
            self.assertEqual(memory.set_memory_fraction_reserve(2*G,1),92*G)
            self.nvml.memory.side_effect=OSError('unavailable')
            self.assertEqual(memory.set_memory_fraction_use(90*G,1),44*G)
            cap.assert_called_with(44/96,device=1)

    def test_initialization_failure_is_optional(self):
        with patch.object(ct,'CDLL',side_effect=OSError('missing DLL')):
            # The original cached factory is patched in setUp; exercise its class.
            with self.assertRaises(OSError):
                wddm._NVML()

    def test_environment_switch_defaults_to_original_cuda_method(self):
        with patch.object(torch.cuda,'mem_get_info',return_value=(40*G,96*G)), \
             patch.object(torch.cuda,'get_device_properties',return_value=props()):
            for value in ('0','', 'true'):
                with patch.dict('os.environ',{'EXL3_WDDM_MEMORY':value}):
                    self.assertEqual(memory.device_mem_info(0),(40*G,96*G))
            with patch.dict('os.environ',{},clear=True):
                self.assertEqual(memory.device_mem_info(0),(40*G,96*G))
            self.nvml.memory.assert_not_called()


class LoaderTests(unittest.TestCase):
    def run_loader(self, free, budget=10*G, transient=G, allocation=4*G, fail_forward=False):
        class Loader(model_ls.Model_LSMixin):
            caps={}
            def get_layer_instances(self,idx): return ()
            def __iter__(self): return iter(())
        class Module:
            caps={};key='layers.0';modules=[]
            def __iter__(self): return iter(())
            def can_defer_load(self): return False
            def load(self,*a,**kw): pass
            def prepare_for_device(self,x,params): return x
            def forward(self,x,params):
                if fail_forward: raise torch.cuda.OutOfMemoryError('real CUDA OOM')
                return x
            unload=Mock()
        config=NS(stc=Mock(),infer_params=NS(vision_pinned=False))
        zeros=torch.zeros
        with ExitStack() as stack:
            stack.enter_context(patch.dict('os.environ',{'EXL3_AUTOSPLIT_MARGIN_MB':'256'}))
            stack.enter_context(patch.object(torch,'zeros',side_effect=lambda *a,**kw: zeros(*a,**{**kw,'device':'cpu'})))
            for name,value in [('memory_allocated',allocation),('memory_reserved',allocation+G//2),
                               ('max_memory_allocated',allocation+transient),('reset_peak_memory_stats',None)]:
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            for name,value in [('device_mem_info',(free,96*G)),('set_memory_fraction_use',budget),
                               ('unset_memory_fraction',None),('free_mem',None)]:
                stack.enter_context(patch.object(model_ls,name,return_value=value))
            list(Loader()._load_autosplit(False,None,[budget],[0],1,1,1,None,False,config,
                                         [Module()],False,1,{},False))

    def test_reusable_pool_and_margin_boundary(self):
        self.run_loader(3*G//4)  # .75 free + .5 cached == 1 transient + .25 margin
        with self.assertRaisesRegex(RuntimeError,'Insufficient VRAM'):
            self.run_loader(3*G//4-1)

    def test_insufficient_budget_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Insufficient VRAM'):
            self.run_loader(90*G,budget=4*G)

    def test_largest_transient_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Insufficient VRAM'):
            self.run_loader(90*G,transient=7*G)

    def test_genuine_oom_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Insufficient VRAM'):
            self.run_loader(90*G,fail_forward=True)


if __name__ == '__main__':
    unittest.main()
