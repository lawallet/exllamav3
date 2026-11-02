"""Exact carried-HC stage parity, FP32 overflow protection and rejection gates."""
import unittest
from unittest.mock import patch
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import native_plan
from exllamav3.modules.hyperconnections import HyperConnection
from exllamav3.modules.rmsnorm import RMSNorm


@unittest.skipUnless(torch.cuda.is_available() and getattr(ext,'native_block_plan_carry_abi',None)==1,
                     'requires carry-capable extension and CUDA')
class TestCarryStages(unittest.TestCase):
    def sites(self,device,dtype):
        sites=[]
        for i in range(2):
            h=HyperConnection(None,f'carry{i}',4,5120,20,1e-6,1e-20,carry_pre=True)
            h.fn=torch.randn(24,20480,device=device)*0.01
            h.fn_h=h.fn.half()
            h.base=torch.randn(24,device=device)*0.1
            h.scale=torch.ones(3,device=device)
            n=RMSNorm(None,f'norm{i}',1e-20)
            n.weight=(torch.randn(5120,device=device)*0.01+1).to(dtype)
            sites.append((h,n))
        return sites

    def test_exact_stock_stages(self):
        torch.manual_seed(941)
        from exllamav3.modules import hyperconnections
        with patch.object(hyperconnections,'_hc_fold',False):
            for device in range(torch.cuda.device_count()):
                for dtype in (torch.float16,torch.bfloat16):
                    sites=self.sites(device,dtype)
                    for q in (1,2,3,4,5):
                        plan=native_plan.deepseek_carry(sites,1,q)
                        for scale in (1.,100000.):
                            for has_pre in (False,True):
                                with self.subTest(device=device,dtype=dtype,query=q,scale=scale,pre=has_pre):
                                    x=torch.randn(1,q,4,5120,device=device)*scale
                                    actual=x.clone()
                                    p=torch.rand(1,q,4,device=device) if has_pre else None
                                    params={'hc_pre':p}
                                    h,n=sites[0]
                                    post,comb,y,_=h.mix_norm(x,params,n)
                                    expected=n.forward(y,params,out_dtype=torch.half)
                                    out=plan.begin_carry(actual,p.view(q,4) if p is not None else None)
                                    self.assertTrue(torch.equal(out,expected))
                                    self.assertTrue(torch.isfinite(out).all())
                                    self.assertTrue(torch.equal(plan.carry_output,params['hc_pre'].view(q,4)))
                                    boundary=torch.randn(1,q,5120,device=device,dtype=dtype).float()
                                    h.apply_(x,boundary,post,comb,params)
                                    h,n=sites[1]
                                    post,comb,y,_=h.mix_norm(x,params,n)
                                    expected=n.forward(y,params,out_dtype=torch.half)
                                    out=plan.middle(actual,boundary)
                                    self.assertTrue(torch.equal(out,expected))
                                    self.assertTrue(torch.equal(actual,x))
                                    self.assertTrue(torch.equal(plan.carry_output,params['hc_pre'].view(q,4)))
                                    boundary=torch.randn_like(boundary)
                                    h.apply_(x,boundary,post,comb,params)
                                    plan.finish(actual,boundary)
                                    self.assertTrue(torch.equal(actual,x))

    def test_invalid_carry_does_not_mutate(self):
        sites=self.sites(0,torch.float16)
        plan=native_plan.deepseek_carry(sites,1,4)
        x=torch.randn(1,4,4,5120,device=0);before=x.clone()
        for pre in (torch.ones(4,4),torch.ones(4,4,device=0,dtype=torch.half),
                    torch.ones(4,5,device=0),torch.ones(4,4,device=0).t()):
            with self.assertRaises(RuntimeError):plan.begin_carry(x,pre)
            self.assertTrue(torch.equal(x,before))
        with self.assertRaises(RuntimeError):plan.begin(x)
        self.assertTrue(torch.equal(x,before))


if __name__=='__main__': unittest.main()
