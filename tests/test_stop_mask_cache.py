"""Mask reuse preserves boundaries, dynamic fallbacks, sampler inputs and ownership."""
import types
import unittest
from unittest.mock import patch
import torch
from exllamav3.generator.job import Job
from exllamav3.generator import stop_mask_cache as policy
from exllamav3.util import draft_seed
from exllamav3.generator.sampler import ArgmaxSampler, CategoricalSampler, CustomSampler, SS_Base, SS_Argmax


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class StopMaskCacheTests(unittest.TestCase):
    def setUp(self):
        self.flag=patch.object(policy,'enabled',True);self.flag.start()
        self.device=torch.device('cuda:0')

    def tearDown(self):
        torch.cuda.synchronize(self.device);self.flag.stop()

    def job(self, **kwargs):
        j=Job(torch.tensor([[1]]),max_new_tokens=20,min_new_tokens=7,
              stop_conditions=[2,5],sampler=ArgmaxSampler(),**kwargs)
        j.generator=types.SimpleNamespace(padded_vocab_size=64)
        j.logits_device=self.device
        return j

    def mask(self,j):
        j.prepare_logit_mask()
        return j.device_logit_mask

    def test_reuse_and_exact_contents(self):
        j=self.job();a=self.mask(j)
        expected=torch.zeros(1,64,dtype=torch.half);expected[:,[2,5]]=-float('inf')
        for n in range(1,7):
            j.new_tokens=n
            self.assertIs(a,self.mask(j))
            self.assertTrue(torch.equal(a.cpu(),expected))

    def test_boundary_and_rewind(self):
        j=self.job();a=self.mask(j)
        j.new_tokens=7;self.assertIsNone(self.mask(j));self.assertIsNone(j._stop_mask_cache)
        j.new_tokens=3;b=self.mask(j)
        self.assertIsNot(a,b);self.assertTrue(torch.equal(a,b))

    def test_stop_set_changes(self):
        j=self.job();a=self.mask(j);j.stop_tokens_list=[3]
        b=self.mask(j);self.assertIsNot(a,b)
        self.assertEqual(b[0,2].item(),0);self.assertEqual(b[0,3].item(),-float('inf'))
        j.stop_tokens_list=[];self.assertIsNone(self.mask(j))

    def test_disable_is_original_path(self):
        j=self.job();a=self.mask(j)
        with patch.object(policy,'enabled',False):
            b=self.mask(j);c=self.mask(j)
        self.assertIsNot(b,c);self.assertTrue(torch.equal(a,c))

    def test_job_ownership(self):
        a=self.job();b=self.job();x=self.mask(a);y=self.mask(b)
        self.assertIsNot(x,y);self.assertTrue(torch.equal(x,y))
        a.stop_tokens_list=[3];self.mask(a)
        self.assertEqual(self.mask(b)[0,3].item(),0)

    def test_stream_change_rebuilds(self):
        j=self.job();a=self.mask(j)
        stream=torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            b=self.mask(j);self.assertIsNot(a,b);self.assertIs(b,self.mask(j))
        stream.synchronize()
        c=self.mask(j);self.assertIsNot(b,c);self.assertTrue(torch.equal(a,c))

    def test_cpu_and_unspecified_device_fallback(self):
        for device in (torch.device('cpu'),None):
            j=self.job();j.logits_device=device
            self.mask(j);self.assertIsNone(j._stop_mask_cache)

    def test_checkpoint_and_banned_string_fallback(self):
        j=self.job();a=self.mask(j)
        j.checkpoint={'offset':0,'explored_tokens':[3]}
        b=self.mask(j);self.assertIsNone(j._stop_mask_cache)
        self.assertEqual(b[0,3].item(),-float('inf'))
        j.checkpoint=None;j.banned_strings=['abc'];self.mask(j)
        self.assertIsNone(j._stop_mask_cache)

    def test_dormant_filter_and_healing_fallback(self):
        j=self.job();self.mask(j)
        j.filters=[types.SimpleNamespace(is_active=False)]
        self.mask(j);self.assertIsNone(j._stop_mask_cache)
        j.filters=[];j.new_tokens=-1
        self.mask(j);self.assertIsNone(j._stop_mask_cache)

    def test_unknown_sampler_fallback(self):
        class Other(ArgmaxSampler):
            def forward(self,*a,**kw):return super().forward(*a,**kw)
        j=self.job();j.sampler=Other();self.mask(j);self.assertIsNone(j._stop_mask_cache)
        class Extra(SS_Base):
            pass
        j.sampler=CustomSampler([SS_Argmax()]);j.sampler.steps.append(Extra())
        self.mask(j);self.assertIsNone(j._stop_mask_cache)
        j.sampler=None;self.mask(j);self.assertIsNone(j._stop_mask_cache)

    def test_seeded_sampling_exact(self):
        j=self.job();j.sampler=CategoricalSampler(temperature=0.8)
        logits=torch.arange(64,device=self.device,dtype=torch.half).view(1,1,64)/16
        for n in range(10):
            j.new_tokens=n
            with patch.object(policy,'enabled',False):
                serial=self.mask(j)
                want=j.sampler.forward(logits.clone(),rand_u32=1234+n,logit_mask=serial)
            got_mask=self.mask(j)
            got=j.sampler.forward(logits.clone(),rand_u32=1234+n,logit_mask=got_mask)
            self.assertTrue(torch.equal(want,got))

    def test_async_seed_dispatch_and_exact_copy(self):
        pinned=torch.tensor([[2,5,8]],dtype=torch.long,pin_memory=True)
        pageable=torch.tensor([[2,5,8]],dtype=torch.long)
        for enabled in (False,True):
            for source in (pinned,pageable,pinned.float()):
                with patch.object(draft_seed,'enabled',enabled), patch.object(draft_seed,'to_device',wraps=draft_seed.to_device) as move:
                    copied=draft_seed.upload(source,self.device)
                    expected=enabled and source.is_pinned() and source.dtype==torch.long
                    self.assertEqual(move.call_args.kwargs['non_blocking'],expected)
                    self.assertTrue(torch.equal(copied.cpu(),source))

    def test_async_seed_device_identity_and_stream_order(self):
        source=torch.tensor([[17]],dtype=torch.long,pin_memory=True)
        stream=torch.cuda.Stream(device=self.device)
        with patch.object(draft_seed,'enabled',True),torch.cuda.stream(stream):
            copied=draft_seed.upload(source,self.device)
            self.assertIs(draft_seed.upload(copied,self.device),copied)
            result=copied+4
        stream.synchronize()
        self.assertEqual(result.item(),21)


if __name__=='__main__':unittest.main()
