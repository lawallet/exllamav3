"""V2 selection, binding lifetime and single-stream boundary regressions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from exllamav3.modules import native_devstral as n, native_plan
from exllamav3.modules.attn import Attention
from exllamav3.modules.mlp import GatedMLP
from test_native_glm_guard import Tensor, Owner


class Weight(Tensor):
    def data_ptr(self): return id(self)


class TestDevstral(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.owner.loaded_tp = False
        dev = torch.device('cuda:7')
        attn, mlp = Attention.__new__(Attention), GatedMLP.__new__(GatedMLP)
        attn.device = mlp.device = dev
        attn.out_dtype, attn.tp_mode, attn.o_proj = torch.float32, False, NS(out_dtype=torch.float32)
        mlp.out_dtype, mlp.tp_reduce, mlp.hidden_size = torch.float32, False, 12288
        def norm():
            m = n.RMSNorm.__new__(n.RMSNorm)
            m.device, m.groups, m.span_heads, m.unweighted = dev, 1, False, False
            m.rms_norm_eps, m.constant_bias, m.constant_scale = 1e-6, 0, 1
            m.weight = Weight((12288,), dev)
            m.weight.dtype = torch.bfloat16
            return m
        self.block = NS(_native_plan_owner=weakref.ref(self.owner), device=dev, attn=attn, mlp=mlp,
            attn_norm=norm(), attn_post_norm=None, mlp_norm=norm(), mlp_post_norm=None,
            attn_hc=None, mlp_hc=None, attn_resid_scalar=None, mlp_resid_scalar=None, layer_scalar_f=None)
        self.x = Tensor((1, 1, 12288), dev)
        self.x.ndim = 3
        for name in ('enabled', 'v2_residual_available'):
            p = patch.object(native_plan, name, True); p.start(); self.addCleanup(p.stop)
        p = patch.object(native_plan.ext, 'BC_BlockPlanV2', create=True)
        self.constructor = p.start(); self.addCleanup(p.stop)

    def get(self, **kw): return n.get_native_block(self.block, self.x, kw)

    def test_emission_and_reuse(self):
        self.assertIsNotNone(self.get()); self.get()
        self.assertEqual(self.constructor.call_count, 1)
        self.assertEqual(self.constructor.call_args.args[3],
            [[(3, 0, 3, 0)], [(7, 1, 4, 1)], [(8, 2, 0, -1)]])

    def test_shapes_topology(self):
        with patch.object(torch.cuda, 'device_count', side_effect=AssertionError), \
             patch.object(torch.cuda, 'get_device_name', side_effect=AssertionError):
            for b, q in ((1, 1), (2, 1), (3, 1), (4, 1), (1, 5), (2, 6)):
                self.x.shape = (b, q, 12288)
                self.assertIsNotNone(self.get())
            for b, q in ((0, 1), (5, 1), (1, 7), (3, 6)):
                self.x.shape = (b, q, 12288)
                self.assertIsNone(self.get())
        self.x.shape = (1, 1, 12288)
        self.block.mlp.device = torch.device('cuda:3')
        self.assertIsNone(self.get())

    def test_modes_and_old_extension(self):
        for key, value in (('capture', None), ('loras', []), ('backend', None),
                           ('quant_preserve', False), ('prefill', True), ('layer_instance', 1)):
            self.assertIsNone(self.get(**{key: value}))
        with patch.object(native_plan, 'v2_residual_available', False): self.assertIsNone(self.get())
        self.owner.loaded_tp = True
        self.assertIsNone(self.get())

    def test_binding_and_config_changes(self):
        self.get()
        self.block.attn_norm.weight = Weight((12288,), self.x.device)
        self.block.attn_norm.weight.dtype = torch.float16
        self.get()
        self.assertEqual(self.constructor.call_count, 2)
        self.block.attn_norm.constant_bias = 0.5
        self.get()
        self.assertEqual(self.constructor.call_count, 3)
        self.block.attn_norm.weight.dtype = torch.float32
        self.assertIsNone(self.get())

    def test_error_propagates(self):
        self.constructor.side_effect = RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError, 'native failure'): self.get()

    def test_export_before_scale_and_residual_param(self):
        from exllamav3.modules import TransformerBlock, transformer
        b = TransformerBlock.__new__(TransformerBlock)
        b.layer_idx, b.out_dtype, b.layer_scalar_f = 0, None, 0.5
        b.attn = NS(forward=lambda x, p: x)
        def mlp(x, p):
            self.assertIs(p['residual'], residual)
            return x
        b.mlp = NS(forward=mlp)
        stages = NS(begin=lambda x: x.half(), middle=lambda x,y: y, finish=lambda x,y: x.add_(1))
        b._native_plan_get = lambda *args: stages
        residual = torch.zeros(1, 1, 8)
        params = {'export_state_layers': {0}}
        with patch.object(transformer, '_native_qwen_enabled', False), \
             patch.object(transformer, '_native_glm_enabled', False):
            self.assertTrue(torch.equal(b.forward(residual, params), torch.full_like(residual, 0.5)))
            self.assertTrue(torch.equal(params['export_states'][0], torch.ones_like(residual).half()))
            stages.finish = lambda *args: (_ for _ in ()).throw(RuntimeError('stage failure'))
            with self.assertRaisesRegex(RuntimeError, 'stage failure'): b.forward(residual, {})

    def test_load_unload_clears(self):
        from exllamav3.modules import TransformerBlock, Module
        b = TransformerBlock.__new__(TransformerBlock)
        b.key_layer_scalar = b.key_attn_resid_scalar = b.key_mlp_resid_scalar = None
        b._native_devstral = {1: object()}
        with patch.object(Module, 'load'): b.load(self.x.device)
        self.assertEqual(b._native_devstral, {})
        b._native_devstral = {1: object()}
        with patch.object(Module, 'unload'): b.unload()
        self.assertEqual(b._native_devstral, {})


if __name__ == '__main__': unittest.main()
