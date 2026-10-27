"""A stock block's deferred v1.6.5 residual must reach the next native block."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from exllamav3.modules import transformer, native_plan, native_qwen, native_glm
from exllamav3.modules.transformer import TransformerBlock


class TestDeferredNativeInput(unittest.TestCase):
    def block(self):
        b = TransformerBlock(None, 'block', layer_idx=0)
        b.attn = NS(forward=lambda x, p: torch.ones_like(x))
        b.mlp = NS(forward=lambda x, p: torch.full_like(x, 2))
        return b

    def test_stock_predecessor_then_native_consumes_residual_once(self):
        for path in ('plan', 'qwen', 'glm'):
            with self.subTest(path=path), ExitStack() as stack:
                stack.enter_context(patch.object(transformer, '_native_qwen_enabled', False))
                stack.enter_context(patch.object(transformer, '_native_glm_enabled', False))
                stack.enter_context(patch.object(native_plan, 'enabled', True))
                stock, native = self.block(), self.block()
                x = torch.zeros(1, 1, 8)
                params = {'resid_defer': True}
                stock.forward(x, params)
                self.assertTrue(torch.equal(x, torch.ones_like(x)))
                self.assertIn('resid_pending', params)
                seen = []
                def begin(value):
                    seen.append(value.clone())
                    return value.half()
                def middle(value, y):
                    value.add_(y)
                    return value.half()
                def finish(value, y):
                    value.add_(y)
                plan = NS(begin=begin, middle=middle, finish=finish)
                if path == 'plan':
                    native._native_plan_get = lambda *a: plan
                else:
                    stack.enter_context(patch.object(transformer, '_native_'+path+'_enabled', True))
                    stack.enter_context(patch.object(native_qwen if path == 'qwen' else native_glm,
                                                    'get_native_block', return_value=plan))
                native.forward(x, params)
                self.assertTrue(torch.equal(seen[0], torch.full_like(x, 3)))
                self.assertNotIn('resid_pending', params)
                native.forward(x, params)
                self.assertTrue(torch.equal(seen[1], torch.full_like(x, 4 if path == 'qwen' else 6)))

    def test_pending_hyperconnection_applied_before_each_native_entry(self):
        for path in ('plan', 'qwen', 'glm'):
            with self.subTest(path=path), ExitStack() as stack:
                stack.enter_context(patch.object(native_plan, 'enabled', True))
                stack.enter_context(patch.object(transformer, '_native_qwen_enabled', path == 'qwen'))
                stack.enter_context(patch.object(transformer, '_native_glm_enabled', path == 'glm'))
                b = self.block()
                x = torch.zeros(1, 1, 4, 8)
                params = {'hc_pending': (x,)}
                events = []
                def flush(p):
                    p.pop('hc_pending')[0].add_(3)
                    events.append('apply')
                def begin(value):
                    self.assertTrue(torch.equal(value, torch.full_like(value, 3)))
                    events.append('begin')
                    return value
                plan = NS(begin=begin, middle=lambda x,y: y, finish=lambda x,y: None)
                b._native_plan_get = lambda *a: plan
                stack.enter_context(patch.object(transformer, 'hc_flush', side_effect=flush))
                stack.enter_context(patch.object(native_qwen, 'get_native_block', return_value=plan))
                stack.enter_context(patch.object(native_glm, 'get_native_block', return_value=plan))
                b.forward(x, params)
                self.assertEqual(events, ['apply', 'begin'])
                self.assertNotIn('hc_pending', params)

    def test_ineligible_native_preserves_stock_defer_protocol(self):
        b = self.block()
        b._native_plan_get = lambda *a: None
        x = torch.zeros(1, 1, 8)
        params = {'resid_pending': torch.full_like(x, 3), 'resid_defer': True}
        with patch.object(native_plan, 'enabled', True), \
             patch.object(transformer, '_native_qwen_enabled', False), \
             patch.object(transformer, '_native_glm_enabled', False):
            b.forward(x, params)
        self.assertTrue(torch.equal(x, torch.full_like(x, 4)))
        self.assertTrue(torch.equal(params['resid_pending'], torch.full_like(x, 2)))

    def test_tp_snapshot_reads_committed_gdn_parity_row(self):
        from exllamav3.modules.gated_delta_net import GDNLayerState
        from util.validate_native_tp import snapshot, digest
        layer = GDNLayerState.__new__(GDNLayerState)
        layer.conv_state = torch.arange(16).reshape(2, 8)
        layer.recurrent_state = torch.arange(32).reshape(4, 8)
        context = dict(kv_modules=[], recurrent_modules=[NS(key='gdn', tp_recurrent_lookup={7:layer})])
        result = snapshot(context, 7, 1, 4, [(1, 0, 3)], [4])
        self.assertEqual(result['gdn:recurrent:0:0'], digest(layer.conv_state[1]))
        self.assertEqual(result['gdn:recurrent:1:0'], digest(layer.recurrent_state[3]))
        layer.recurrent_state[1].add_(100)
        self.assertEqual(snapshot(context, 7, 1, 4, [(1, 0, 3)], [4]), result)


if __name__ == '__main__':
    unittest.main()
