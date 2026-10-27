"""Conservative guard and emitter contracts without GPU allocations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import weakref
import torch
from exllamav3.modules import native_deepseek as n, native_plan
from exllamav3.modules.dsv4 import DSV4Attention
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from test_native_glm_guard import Tensor, Owner


class TestDeepseek(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.owner.loaded_tp = False
        def site():
            h = n.HyperConnection.__new__(n.HyperConnection)
            h.hc_mult, h.hidden_size, h.sinkhorn_iters = 4, 4096, 20
            h.hc_eps = h.rms_eps = 1e-6
            h.device = torch.device('cuda:7')
            h.fn, h.base, h.scale = [Tensor(s, h.device) for s in ((24,16384),(24,),(3,))]
            h.fn_h = None
            norm = n.RMSNorm.__new__(n.RMSNorm)
            norm.device = h.device
            norm.rms_norm_eps, norm.constant_bias, norm.constant_scale = 1e-6, 0, 1
            norm.groups, norm.span_heads, norm.unweighted = 1, False, False
            norm.weight = Tensor((4096,), h.device).half()
            return h,norm
        a,an = site()
        m,mn = site()
        attn = DSV4Attention.__new__(DSV4Attention)
        attn.tp_mode, attn.out_dtype, attn.device = False, torch.float32, a.device
        mlp = BlockSparseMLP.__new__(BlockSparseMLP)
        mlp.out_dtype, mlp.device = torch.float32, a.device
        self.block = NS(_native_plan_owner=weakref.ref(self.owner), device=a.device, attn=attn, mlp=mlp,
            attn_hc=a, mlp_hc=m, attn_norm=an, mlp_norm=mn, attn_post_norm=None, mlp_post_norm=None,
            attn_resid_scalar=None, mlp_resid_scalar=None, layer_scalar_f=None)
        self.x = Tensor((1,1,4,4096), a.device)
        self.x.ndim = 4
        for name,value in (('enabled',True),('available',True)):
            p = patch.object(native_plan,name,value); p.start(); self.addCleanup(p.stop)
        p = patch.object(native_plan.ext,'BC_BlockPlan',create=True)
        self.constructor = p.start(); self.addCleanup(p.stop)

    def get(self, **params): return n.get_native_block(self.block,self.x,params)

    def test_construct_reuse_and_emission(self):
        self.assertIsNotNone(self.get())
        self.get()
        self.assertEqual(self.constructor.call_count,1)
        args = self.constructor.call_args.args
        self.assertEqual(args[2], [1e-6,1e-6])
        self.assertEqual(args[5], [[(1,0),(3,0)],[(2,0),(1,1),(3,1)],[(2,1)]])
        self.assertEqual(args[6:8],([],None))

    def test_topology_is_participant_only(self):
        with patch.object(torch.cuda,'device_count',side_effect=AssertionError), \
             patch.object(torch.cuda,'get_device_name',side_effect=AssertionError):
            self.assertIsNotNone(self.get())
        self.block.mlp.device = torch.device('cuda:3')
        self.assertIsNone(self.get())

    def test_shapes_and_specializations(self):
        for b,q in ((1,1),(2,1),(3,1),(4,1),(1,6),(2,6),(3,4)):
            self.x.shape=(b,q,4,4096)
            self.assertIsNotNone(self.get())
        self.assertEqual(self.constructor.call_count,7)
        for b,q in ((0,1),(5,1),(1,7),(3,6)):
            self.x.shape=(b,q,4,4096)
            self.assertIsNone(self.get())

    def test_unsupported_modes(self):
        for k,v in (('capture',None),('quant_preserve',False),('backend',None),('loras',[]),
                    ('prefill',True),('autosplit_measure',True),('layer_instance',1),('export_state_norm_keys',set())):
            self.assertIsNone(self.get(**{k:v}),k)
        self.assertIsNotNone(self.get(export_state_layers={40,41,42},recurrent_history=True))
        self.owner.loaded_tp=True
        self.assertIsNone(self.get())

    def test_missing_binding_disabled(self):
        with patch.object(native_plan,'available',False): self.assertIsNone(self.get())
        with patch.object(native_plan,'enabled',False): self.assertIsNone(self.get())

    def test_invalidation_and_bad_norm(self):
        self.get()
        self.block.attn_norm.weight=Tensor((4096,),self.x.device).half()
        self.get()
        self.assertEqual(self.constructor.call_count,2)
        self.block.attn_norm.rms_norm_eps=1e-5
        self.assertIsNone(self.get())
        self.assertEqual(self.constructor.call_count,2)

    def test_native_error_propagates(self):
        self.constructor.side_effect=RuntimeError('native contract')
        with self.assertRaisesRegex(RuntimeError,'native contract'): self.get()

    def test_norm_dtypes_and_rebinding(self):
        for dtype in (torch.float16, torch.bfloat16):
            self.block.mlp_norm.weight = Tensor((4096,), self.x.device)
            self.block.mlp_norm.weight.dtype = dtype
            self.assertIsNotNone(self.get())
        calls = self.constructor.call_count
        self.block.mlp_norm.weight.dtype = torch.float32
        self.assertIsNone(self.get())
        self.block.mlp_norm.weight.dtype = torch.float16
        self.block.mlp_norm.weight.shape = (1,4096)
        self.assertIsNone(self.get())
        self.assertEqual(self.constructor.call_count, calls)

    def test_registered_dispatch_keeps_exports_and_propagates_stage_errors(self):
        from exllamav3.modules import transformer, TransformerBlock
        b=TransformerBlock.__new__(TransformerBlock)
        b.layer_idx,b.out_dtype=40,None
        b.attn=NS(forward=lambda x,p:x)
        b.mlp=NS(forward=lambda x,p:x)
        class Stages:
            def begin(self,x): return x.mean(2).half()
            def middle(self,x,y): return y
            def finish(self,x,y): x.add_(1)
        stage=Stages()
        b._native_plan_get=lambda *args:stage
        x=torch.randn(1,1,4,4096)
        params={'export_state_layers':{40,41,42}}
        with patch.object(transformer,'_native_qwen_enabled',False), \
             patch.object(transformer,'_native_glm_enabled',False):
            result=b.forward(x,params)
            self.assertIs(result,x)
            self.assertTrue(torch.equal(params['export_states'][0],x.mean(2).half().clamp(-65504,65504)))
            with patch.object(stage,'finish',side_effect=RuntimeError('stage failure')):
                with self.assertRaisesRegex(RuntimeError,'stage failure'): b.forward(x,{})

    def test_lifecycle_clears_bindings(self):
        from exllamav3.modules import TransformerBlock, Module
        b=TransformerBlock.__new__(TransformerBlock)
        b.key_layer_scalar=b.key_attn_resid_scalar=b.key_mlp_resid_scalar=None
        b._native_deepseek={1:object()}
        with patch.object(Module,'load'):
            b.load(torch.device('cuda:7'))
        self.assertEqual(b._native_deepseek,{})
        b._native_deepseek={1:object()}
        with patch.object(Module,'unload'): b.unload()
        self.assertEqual(b._native_deepseek,{})


if __name__ == '__main__': unittest.main()
