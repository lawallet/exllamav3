"""Exact stateless projection graphs, changing bindings, stream and fallback gates."""
import unittest
import os
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import native_projection as policy, native_plan
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.rmsnorm import RMSNorm
from exllamav3.modules.dsv4 import DSV4Attention


def attention(device):
    a=DSV4Attention.__new__(DSV4Attention)
    a.config=NS(arch_string='DeepseekV41ForCausalLM');a.q_head_norm=False;a.tp_mode=False
    a._native_projection_plans={}
    for name,din,dout in (('q_a',5120,1280),('q_b',1280,32768),('wkv',5120,512)):
        lin=Linear(None,name,din,dout,out_dtype=torch.half)
        lin.inner=LinearEXL3(None,din,dout,
            suh=torch.ones(din,device=device,dtype=torch.half),
            svh=torch.ones(dout,device=device,dtype=torch.half)*0.1,
            trellis=torch.randint(-32768,32767,(din//16,dout//16,48),device=device,dtype=torch.int16),
            mul1=torch.ones((),device=device),out_dtype=torch.half)
        setattr(a,name,lin)
    a.q_norm=RMSNorm(None,'q_norm',1e-6,out_dtype=torch.half)
    a.q_norm.weight=torch.ones(1280,device=device,dtype=torch.half)
    return a


def weights(a):
    return [t for lin in (a.q_a,a.q_b,a.wkv) for t in
            (lin.inner.trellis,lin.inner.suh,lin.inner.svh)]+[a.q_norm.weight]


def reference(a,x):
    qr=a.q_norm.forward(a.q_a.forward(x,{}),{},out_dtype=torch.half)
    return qr,a.q_b.forward(qr,{}),a.wkv.forward(x,{})


@unittest.skipUnless(torch.cuda.is_available() and getattr(ext,'native_projection_plan_abi',None)==1,
                     'requires projection plan extension and CUDA')
class TestNativeProjection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1841)

    def assert_outputs(self,expected,actual):
        for e,a in zip(expected,actual):
            self.assertEqual(e.dtype,a.dtype);self.assertEqual(e.shape,a.shape)
            self.assertTrue(torch.equal(e,a));self.assertTrue(torch.isfinite(a).all())

    def test_exact_all_query_shapes_and_changing_input_pointers(self):
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                a=attention(device)
                for q in range(1,6):
                    plan=native_plan.projections(weights(a),1e-6,q)
                    retained=[]
                    for scale in (0.,0.13,1.,17.):
                        x=torch.randn(1,q,5120,device=device,dtype=torch.half)*scale
                        retained.append(x)
                        self.assert_outputs(reference(a,x),plan.run(x))
                    self.assertEqual(plan.graph_captures,1)
                    self.assertEqual(plan.graph_replays,4)
                    plan.set_graphs(False)
                    self.assert_outputs(reference(a,x),plan.run(x))
                    self.assertEqual(plan.graph_replays,4)

    def test_binding_and_input_rejection_before_mutation(self):
        a=attention(0);w=weights(a);p=native_plan.projections(w,1e-6,1)
        x=torch.randn(1,1,5120,device=0,dtype=torch.half)
        out=p.run(x);before=[t.clone() for t in out]
        for bad in (x.float(),x.cpu(),x.expand(2,1,5120),x[...,::2]):
            with self.assertRaises(RuntimeError):p.run(bad)
            self.assert_outputs(before,out)
        w[9].resize_(2,640)
        with self.assertRaisesRegex(RuntimeError,'changed fixed binding'):p.run(x)
        self.assert_outputs(before,out)

    def test_stream_specific_plans(self):
        a=attention(0);x=torch.randn(1,1,5120,device=0,dtype=torch.half)
        p=native_plan.projections(weights(a),1e-6,1);p.run(x)
        stream=torch.cuda.Stream(0);stream.wait_stream(torch.cuda.current_stream(0))
        with torch.cuda.stream(stream):
            with self.assertRaisesRegex(RuntimeError,'changed execution stream'):p.run(x)
            other=native_plan.projections(weights(a),1e-6,1)
            self.assert_outputs(reference(a,x),other.run(x))
        torch.cuda.current_stream(0).wait_stream(stream)

    def test_bfloat_norm_and_output_alias_rejection(self):
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                a=attention(device);a.q_norm.weight=a.q_norm.weight.bfloat16()
                p=native_plan.projections(weights(a),1e-6,1)
                x=torch.randn(1,1,5120,device=device,dtype=torch.half)
                out=p.run(x);self.assert_outputs(reference(a,x),out)
                alias=out[1][...,:5120]
                self.assertTrue(alias.is_contiguous())
                before=[t.clone() for t in out]
                with self.assertRaises(RuntimeError):p.run(alias)
                self.assert_outputs(before,out)

    def test_rejects_invalid_specializations(self):
        a=attention(0);w=weights(a)
        for q in (0,6):
            with self.assertRaises(RuntimeError):native_plan.projections(w,1e-6,q)
        for eps in (0.,float('nan'),float('inf')):
            with self.assertRaises(RuntimeError):native_plan.projections(w,eps,1)
        for bad in (w[:-1],w[:9]+[w[9].float()],w[:9]+[w[9][::2]]):
            with self.assertRaises(RuntimeError):native_plan.projections(bad,1e-6,1)

    def test_dispatch_policy_and_global_graph_optout(self):
        code='''
import os,torch
from test_native_projection import attention,weights,reference
from exllamav3.modules import native_plan
for device in range(torch.cuda.device_count()):
    with torch.cuda.device(device):
        a=attention(device)
        for q in (1,2,4):
            p=native_plan.projections(weights(a),1e-6,q)
            for i in range(2):
                x=torch.randn(1,q,5120,device=device,dtype=torch.half)
                assert all(torch.equal(u,v) for u,v in zip(reference(a,x),p.run(x)))
            assert p.graph_captures==(0 if os.environ['EXL3_GRAPHS']=='0' else 1)
'''
        for mode,graphs in (('2','0'),('0','1'),('1','1')):
            result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,
                EXL3_INT8_GEMV=mode,EXL3_GRAPHS=graphs),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_policy_fallbacks_and_rebinding(self):
        a=attention(0);x=torch.ones(1,1,5120,device=0,dtype=torch.half)
        with patch.object(policy,'enabled',True),patch.object(policy,'graphs_enabled',True),patch.object(native_plan,'enabled',True):
            plan=policy.get_plan(a,x,{})
            self.assertIsNotNone(plan);self.assertIs(plan,policy.get_plan(a,x,{}))
            for params in ({'prefill':True},{'capture':{}},{'loras':[]},{'reconstruct':False},
                           {'export_state_norm_keys':['q_norm']},{'backend':None}):
                self.assertIsNone(policy.get_plan(a,x,params))
            for bad in (x.expand(2,1,5120),x.expand(1,6,5120),x.float(),x.cpu()):
                self.assertIsNone(policy.get_plan(a,bad,{}))
            for obj,key,value in ((a,'q_head_norm',True),(a,'tp_mode',True),(a.q_a,'pre_scale',2.),
                                  (a.q_b.inner,'bias',torch.zeros(32768,device=0)),
                                  (a.q_norm,'groups',2),(a.q_norm,'unweighted',True)):
                with patch.object(obj,key,value):self.assertIsNone(policy.get_plan(a,x,{}))
            for obj,key in ((policy,'enabled'),(policy,'available'),(policy,'graphs_enabled'),(native_plan,'enabled')):
                with patch.object(obj,key,False):self.assertIsNone(policy.get_plan(a,x,{}))
            a.q_norm.weight=a.q_norm.weight.clone()*0.9
            changed=policy.get_plan(a,x,{})
            self.assertIsNot(plan,changed)
            self.assert_outputs(reference(a,x),changed.run(x))

if __name__=='__main__':unittest.main()
