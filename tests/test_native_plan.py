"""CPU guard regressions; GPU exact parity is covered by existing harnesses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from exllamav3.modules import native_plan
from test_native_glm_guard import TestGuard as GlmGuard
from test_native_qwen_guard import TestGuard as QwenGuard


class TestGlmTopology(GlmGuard):
    def test_unused_visible_gpu(self):
        # Never enumerate global topology, including when three GPUs are visible.
        for count in (2, 3):
            self.block._native_glm = None
            with patch.object(torch.cuda, 'device_count', return_value=count) as visible:
                self.assertIsNotNone(self.get())
                visible.assert_not_called()

    def test_plan_emission_and_missing_extension(self):
        with patch.object(native_plan, 'enabled', True), \
             patch.object(native_plan, 'available', True), \
             patch.object(native_plan, 'glm_placement', return_value=True), \
             patch.object(native_plan.ext, 'BC_BlockPlan', create=True) as constructor:
            self.assertIsNotNone(self.get())
            args = constructor.call_args.args
            self.assertEqual(args[5], [[(1, 0), (3, 0)], [(2, 0), (1, 1), (3, 1)], [(2, 1)]])
            self.assertEqual(args[-2:], (1, 1))
        with patch.object(native_plan, 'enabled', True), patch.object(native_plan, 'available', False):
            self.assertIsNone(self.get())


class TestQwenTopology(QwenGuard):
    def test_unused_visible_gpu(self):
        for count in (2, 3):
            self.block._native_qwen = None
            with patch.object(torch.cuda, 'device_count', return_value=count) as visible:
                self.assertIs(self.get(), self.sentinel)
                visible.assert_not_called()


class TestPlacement(unittest.TestCase):
    def test_validation_backend_restores_and_invalidates_both_models(self):
        # tests/util.py is a legacy standalone module which shadows the utility
        # namespace during unittest discovery. Load this test helper explicitly.
        import importlib.util
        spec = importlib.util.spec_from_file_location('native_plan_validation',
            Path(__file__).resolve().parents[1]/'util/native_plan_validation.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        backend = helper.backend
        blocks = [NS(_native_qwen=object(), _native_qwen_mtp={1: object()}),
                  NS(_native_glm=object(), _native_glm_mtp={4: object()}, _native_glm_batch={2: object()})]
        previous = native_plan.enabled
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with backend([NS(modules=[b]) for b in blocks], not previous):
                self.assertEqual(native_plan.enabled, not previous)
                self.assertIsNone(blocks[0]._native_qwen)
                self.assertEqual(blocks[1]._native_glm_mtp, {})
                blocks[0]._native_qwen = object()
                raise RuntimeError('test failure')
        self.assertEqual(native_plan.enabled, previous)
        self.assertIsNone(blocks[0]._native_qwen)

    def test_qwen_emission_keeps_two_boundaries(self):
        weights = list(range(13))
        with patch.object(native_plan, 'enabled', True), \
             patch.object(native_plan.ext, 'BC_BlockPlan', create=True) as constructor:
            native_plan.qwen(weights, object(), 1e-5, 1e-5, 1, 5)
            args = constructor.call_args.args
            self.assertEqual(args[0], [weights[:3], weights[3:6]])
            self.assertEqual(args[5], [[(0, 0)], [], [(2, 0), (0, 1), (4, 1), (5, 1), (2, 1)]])
            self.assertEqual(args[-2:], (1, 5))

    def test_remapped_participants_and_head(self):
        for first, second in ((0, 1), (2, 0)):
            modules = []
            for i in range(45):
                device = f'cuda:{first if i < 35 else second}'
                child = NS(device=device)
                modules.append(NS(_native_glm_owner=object(), layer_idx=i, device=device,
                    attn=child, mlp=child, attn_hc=child, mlp_hc=child,
                    attn_norm=child, mlp_norm=child))
            modules.extend([NS(device=f'cuda:{second}'), NS(device=f'cuda:{second}')])
            owner = NS(modules=modules, logit_layer_idx=46)
            with patch.object(torch.cuda, 'device_count', side_effect=AssertionError('global topology queried')), \
                 patch.object(torch.cuda, 'get_device_name', side_effect=lambda i: 'RTX PRO 6000' if i == first else '4090 D'):
                self.assertTrue(native_plan.glm_placement(owner))
                self.assertFalse(native_plan.glm_placement(owner, NS(modules=[NS(device=f'cuda:{second}')])) )
                modules[-1].device = f'cuda:{first}'
                self.assertFalse(native_plan.glm_placement(owner))


if __name__ == '__main__':
    unittest.main()
