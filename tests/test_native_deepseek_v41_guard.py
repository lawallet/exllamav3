"""V4.1 eligibility, stock fallbacks and cached binding invalidation."""
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from exllamav3.modules import native_deepseek_v41 as n, native_plan
from exllamav3.modules.dsv4 import DSV4Attention
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from test_native_glm_guard import Tensor, Owner


class TestCarryGuard(unittest.TestCase):
    def setUp(self):
        self.owner=Owner();self.owner.loaded_tp=False
        def tensor(shape,dtype=torch.float):
            t=Tensor(shape);t.dtype=dtype
            return t
        def site():
            h=n.HyperConnection.__new__(n.HyperConnection)
            h.hc_mult,h.hidden_size,h.sinkhorn_iters=4,5120,20
            h.hc_eps,h.rms_eps,h.carry_pre=1e-6,1e-20,True
            h.device=torch.device('cuda:0')
            h.fn=tensor((24,20480));h.fn_h=tensor((24,20480),torch.half)
            h.base=tensor((24,));h.scale=tensor((3,))
            norm=n.RMSNorm.__new__(n.RMSNorm)
            norm.rms_norm_eps,norm.constant_bias,norm.constant_scale=1e-20,0,1
            norm.groups,norm.span_heads,norm.unweighted=1,False,False
            norm.device=h.device;norm.weight=tensor((5120,),torch.half)
            return h,norm
        a,an=site();m,mn=site()
        attn=DSV4Attention.__new__(DSV4Attention)
        attn.tp_mode,attn.out_dtype,attn.device=False,torch.float,torch.device('cuda:1')
        mlp=BlockSparseMLP.__new__(BlockSparseMLP);mlp.device=a.device
        self.block=NS(attn_hc=a,attn_norm=an,mlp_hc=m,mlp_norm=mn,attn=attn,mlp=mlp,
            device=a.device,_native_plan_owner=weakref.ref(self.owner),_native_deepseek={},
            attn_post_norm=None,mlp_post_norm=None,attn_resid_scalar=None,mlp_resid_scalar=None,layer_scalar_f=None)
        self.x=tensor((1,4,4,5120));self.x.ndim=4
        for obj,key,value in ((n,'available',True),(native_plan,'enabled',True),(Tensor,'data_ptr',lambda s:id(s))):
            p=patch.object(obj,key,value,create=True);p.start();self.addCleanup(p.stop)
        p=patch.object(native_plan,'deepseek_carry');self.emit=p.start();self.addCleanup(p.stop)

    def get(self,**params):return n.get_native_block(self.block,self.x,params)

    def test_split_attention_and_cache(self):
        self.assertIsNotNone(self.get());self.get();self.assertEqual(self.emit.call_count,1)
        self.block.attn_norm.weight=self.block.attn_norm.weight.half()
        self.get();self.assertEqual(self.emit.call_count,2)
        self.block.mlp.device=torch.device('cuda:1');self.assertIsNone(self.get())

    def test_modes_shapes_and_missing_abi(self):
        for q in (1,2,3,4,5):
            self.x.shape=(1,q,4,5120);self.assertIsNotNone(self.get())
        for shape in ((2,1,4,5120),(1,6,4,5120),(1,1,4,4096)):
            self.x.shape=shape;self.assertIsNone(self.get())
        self.x.shape=(1,4,4,5120)
        for key,value in (('capture',None),('quant_preserve',False),('backend',None),('loras',[]),
                          ('prefill',True),('autosplit_measure',True),('export_state_norm_keys',set()),('layer_instance',1)):
            self.assertIsNone(self.get(**{key:value}))
        self.assertIsNotNone(self.get(export_state_layers={36,37,38}))
        with patch.object(n,'available',False):self.assertIsNone(self.get())
        self.owner.loaded_tp=True;self.assertIsNone(self.get())

    def test_bad_carry_and_norm_before_cache(self):
        self.get();calls=self.emit.call_count
        for prev in (Tensor((1,4,5)),Tensor((1,3,4))):self.assertIsNone(self.get(hc_pre=prev))
        self.block.attn_hc.carry_pre=False;self.assertIsNone(self.get())
        self.block.attn_hc.carry_pre=True
        self.block.mlp_norm.rms_norm_eps=1e-5;self.assertIsNone(self.get())
        self.assertEqual(self.emit.call_count,calls)

    def test_error_propagates(self):
        self.emit.side_effect=RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError,'native failure'):self.get()


if __name__=='__main__':unittest.main()
