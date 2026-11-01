"""Attachment, window, history, placement and scratch-isolation checks."""
import unittest
from unittest.mock import patch
import weakref
from types import SimpleNamespace as NS
from test_native_glm_guard import TestGuard,Owner
from exllamav3.modules import native_glm as n


class TestMTPGuard(unittest.TestCase):
    def setUp(self):
        TestGuard.setUp(self)
        self.x.ndim=4
        self.owner.config.num_mtp_layers=1
        self.owner.logit_layer_idx=2
        self.owner.modules.append(NS(key='target.norm'))
        self.draft=Owner()
        self.draft.loaded_tp=False
        self.draft.caps={'mtp_draft':True}
        self.draft.modules=[NS(device='cuda:0')]
        self.draft.attached_model=weakref.ref(self.owner)
        self.owner._native_glm_mtp_draft=weakref.ref(self.draft)
        self.params['export_state_norm_keys']={'target.norm'}
        self.cache.max_history=3
        p=patch.object(n.ext,'native_glm_mtp_max_q_len',4,create=True)
        p.start(); self.addCleanup(p.stop)

    def get(self,**kw): return n.get_native_block(self.block,self.x,dict(self.params,**kw))

    def test_bootstrap_and_verify_separate_scratch(self):
        self.assertIsNotNone(self.get())
        self.x.shape=(1,4,4,4096)
        self.assertIsNotNone(self.get(recurrent_history=True))
        self.assertEqual(set(self.block._native_glm_mtp),{1,4})
        self.assertFalse(hasattr(self.block,'_native_glm'))

    def test_history_required_for_verification(self):
        self.x.shape=(1,4,4,4096)
        self.assertIsNone(self.get())

    def test_wrong_export(self):
        self.assertIsNone(self.get(export_state_norm_keys={'other'}))

    def test_wrong_window(self):
        for q in (2,3,5):
            self.x.shape=(1,q,4,4096)
            self.assertIsNone(self.get(recurrent_history=True))

    def test_norm_dtype_and_shape_checked_before_cached_plan(self):
        TestGuard.test_norm_dtype_and_shape_checked_before_cached_plan(self)

    def test_carried_pre_rejects_cached_plan(self):
        TestGuard.test_carried_pre_rejects_cached_plan(self)

    def test_wrong_attachment(self):
        self.draft.attached_model=lambda:None
        self.assertIsNone(self.get())

    def test_history_capacity(self):
        self.cache.max_history=4
        self.assertIsNone(self.get())

    def test_draft_placement(self):
        self.draft.modules[0].device='cuda:1'
        self.assertIsNone(self.get())

    def test_prefill_and_tp(self):
        self.assertIsNone(self.get(prefill=True))
        self.draft.loaded_tp=True
        self.assertIsNone(self.get())


if __name__=='__main__': unittest.main()
