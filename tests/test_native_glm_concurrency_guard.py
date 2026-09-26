"""Ordinary compact-batch ownership and shape-specific scratch guards."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from test_native_glm_guard import TestGuard
from exllamav3.modules import native_glm as n

class TestConcurrencyGuard(unittest.TestCase):
    def setUp(self):
        TestGuard.setUp(self)
        p=patch.object(n.ext,'native_glm_max_batch',4,create=True)
        p.start(); self.addCleanup(p.stop)
    def batch(self,b):
        self.x.shape=(b,1,4,4096)
        self.params['recurrent_states']=[NS(cache=self.cache,exported=False,slot=i) for i in range(b)]
    def get(self): return n.get_native_block(self.block,self.x,self.params)
    def test_batches_and_separate_scratch(self):
        self.get()
        for b in (2,3,4,2): self.batch(b); self.assertIsNotNone(self.get())
        self.assertEqual(set(self.block._native_glm_batch),{2,3,4})
        self.assertEqual(self.constructor.call_count,4)
        self.assertIsNotNone(self.block._native_glm)
    def test_duplicate_slots(self):
        self.batch(2); self.params['recurrent_states'][1].slot=0
        self.assertIsNone(self.get())
    def test_wrong_cache(self):
        self.batch(2); self.params['recurrent_states'][1].cache=object()
        self.assertIsNone(self.get())
    def test_mismatched_state_count(self):
        self.batch(3); self.params['recurrent_states'].pop()
        self.assertIsNone(self.get())
    def test_unsupported_shape(self):
        self.batch(5); self.assertIsNone(self.get())
        self.batch(2); self.x.shape=(2,4,4,4096); self.assertIsNone(self.get())
    def test_old_extension(self):
        self.batch(2)
        with patch.object(n.ext,'native_glm_max_batch',0): self.assertIsNone(self.get())

class TestMTPConcurrencyGuard(unittest.TestCase):
    def setUp(self):
        from test_native_glm_mtp_guard import TestMTPGuard
        TestMTPGuard.setUp(self)
        p=patch.object(n.ext,'native_glm_mtp_max_batch',3,create=True)
        p.start(); self.addCleanup(p.stop)
    def batch(self,b,q=4):
        self.x.shape=(b,q,4,4096)
        self.params['recurrent_history']=q==4
        self.params['recurrent_states']=[NS(cache=self.cache,exported=False,slot=i) for i in range(b)]
    def get(self): return n.get_native_block(self.block,self.x,self.params)
    def test_mtp_shapes_have_separate_scratch(self):
        for b in (1,2,3,2): self.batch(b); self.assertIsNotNone(self.get())
        self.assertEqual(set(self.block._native_glm_mtp),{4,(2,4),(3,4)})
        self.assertEqual(self.constructor.call_count,3)
    def test_mtp_duplicate_slots(self):
        self.batch(2); self.params['recurrent_states'][1].slot=0
        self.assertIsNone(self.get())
    def test_mtp_four_jobs_fall_back(self):
        self.batch(4); self.assertIsNone(self.get())
    def test_mtp_old_extension(self):
        self.batch(2)
        with patch.object(n.ext,'native_glm_mtp_max_batch',0): self.assertIsNone(self.get())
    def test_mtp_bootstrap(self):
        self.batch(3,1); self.assertIsNotNone(self.get())

if __name__=='__main__': unittest.main()
