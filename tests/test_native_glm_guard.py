"""GLM eligibility and invalidation tests; no GPU allocations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import weakref
import torch
from exllamav3.modules import native_glm as n


class Owner:
    pass


class Tensor:
    dtype = torch.float32
    is_cuda = True
    def __init__(self, shape, device='cuda:0'):
        self.shape, self.device = shape, torch.device(device)
    def is_contiguous(self):
        return True
    def numel(self):
        import math
        return math.prod(self.shape)
    def half(self):
        t = Tensor(self.shape, self.device)
        t.dtype = torch.float16
        return t


class TestGuard(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.owner.loaded_tp = False
        self.owner.config = NS(arch_string='Glm5NextForConditionalGeneration', num_hidden_layers=45)
        self.owner.caps = {}
        def site():
            h = n.HyperConnection.__new__(n.HyperConnection)
            h.hc_mult, h.hidden_size, h.sinkhorn_iters = 4, 4096, 20
            h.hc_eps, h.rms_eps = 1e-6, 1e-5
            h.fn, h.base, h.scale = Tensor((24, 16384)), Tensor((24,)), Tensor((3,))
            h.fn_h = h.fn.half()
            norm = n.RMSNorm.__new__(n.RMSNorm)
            norm.rms_norm_eps, norm.constant_bias, norm.constant_scale = 1e-5, 0, 1
            norm.groups, norm.span_heads, norm.unweighted = 1, False, False
            norm.weight = Tensor((4096,)).half()
            return h, norm
        a, an = site()
        m, mn = site()
        self.block = NS(_native_glm_owner=weakref.ref(self.owner), device='cuda:0', layer_idx=0,
            attn_hc=a, mlp_hc=m, attn_norm=an, mlp_norm=mn,
            attn_resid_scalar=None, mlp_resid_scalar=None, attn_post_norm=None, mlp_post_norm=None)
        self.owner.modules = [self.block]
        self.cache = NS(max_history=0)
        self.params = dict(cache=self.cache, recurrent_states=[NS(cache=self.cache, exported=False)])
        self.x = Tensor((1, 1, 4, 4096))
        self.patches = [patch.object(n.ext, 'BC_GlmBlock', create=True),
                        patch.object(torch.cuda, 'device_count', return_value=2),
                        patch.object(torch.cuda, 'get_device_name', side_effect=lambda i: 'RTX PRO 6000' if i == 0 else '4090 D')]
        self.constructor = self.patches[0].start()
        for p in self.patches[1:]: p.start()
        for p in self.patches: self.addCleanup(p.stop)

    def get(self, **params):
        return n.get_native_block(self.block, self.x, dict(self.params, **params))

    def test_supported_and_reuse(self):
        self.assertIsNotNone(self.get())
        self.assertIsNotNone(self.get())
        self.assertEqual(self.constructor.call_count, 1)

    def test_protocol_fallbacks(self):
        for k, value in [('capture', None), ('loras', []), ('export_state_norm_keys', set()),
                         ('backend', None), ('prefill', True), ('recurrent_history', True),
                         ('autosplit_measure', True), ('layer_instance', 1)]:
            with self.subTest(k=k):
                self.assertIsNone(self.get(**{k: value}))

    def test_carried_pre_rejects_cached_plan(self):
        self.assertIsNotNone(self.get())
        self.block.attn_hc.carry_pre = True
        self.assertIsNone(self.get())

    def test_norm_change_invalidates(self):
        self.get()
        self.block.attn_norm.constant_scale = 2
        self.assertIsNone(self.get())

    def test_norm_dtype_and_shape_checked_before_cached_plan(self):
        for dtype in (torch.float16, torch.bfloat16):
            self.block.mlp_norm.weight = Tensor((4096,)).half()
            self.block.mlp_norm.weight.dtype = dtype
            self.assertIsNotNone(self.get())
        calls = self.constructor.call_count
        # Keep tensor identity unchanged, as with weight.data rebinding.
        self.block.mlp_norm.weight.dtype = torch.float32
        self.assertIsNone(self.get())
        self.block.mlp_norm.weight.dtype = torch.float16
        self.block.mlp_norm.weight.shape = (1,4096)
        self.assertIsNone(self.get())
        self.assertEqual(self.constructor.call_count, calls)

    def test_reload_rebuilds(self):
        self.get()
        self.block.attn_norm.weight = Tensor((4096,)).half()
        self.get()
        self.assertEqual(self.constructor.call_count, 2)

    def test_wrong_device(self):
        self.block.device = 'cuda:1'
        self.assertIsNone(self.get())

    def test_shapes(self):
        for shape in ((2, 1, 4, 4096), (1, 2, 4, 4096), (1, 1, 4, 2560)):
            self.x.shape = shape
            self.assertIsNone(self.get())

    def test_tp_and_other_architecture(self):
        self.owner.loaded_tp = True
        self.assertIsNone(self.get())
        self.owner.loaded_tp = False
        self.owner.config.arch_string = 'Qwen4ExpForConditionalGeneration'
        self.assertIsNone(self.get())

    def test_history_and_slots(self):
        self.cache.max_history = 4
        self.assertIsNone(self.get())
        self.cache.max_history = 0
        self.assertIsNone(self.get(recurrent_states=[]))
        self.assertIsNone(self.get(recurrent_states=[NS(cache=self.cache, exported=True)]))


if __name__ == '__main__':
    unittest.main()
