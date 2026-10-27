"""HC graph selection is construction-only, capability-gated, and optional."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from exllamav3.modules import native_plan, native_tp


class TestHCGraphPolicy(unittest.TestCase):
    def test_supported_hc_plan_enables_graphs_at_construction(self):
        plan = SimpleNamespace(set_graphs=Mock())
        sites = object()
        with patch.object(native_plan, 'deepseek', return_value=plan) as emit, \
             patch.object(native_tp, '_hc_graphs_available', True), \
             patch.object(native_tp, 'hc_graphs_enabled', True):
            self.assertIs(native_tp._hc_plan(sites, 2, 6, kind='glm'), plan)
        emit.assert_called_once_with(sites, 2, 6)
        plan.set_graphs.assert_called_once_with(True)

    def test_operator_opt_out_keeps_eager_plan(self):
        plan = SimpleNamespace(set_graphs=Mock())
        with patch.object(native_plan, 'deepseek', return_value=plan), \
             patch.object(native_tp, '_hc_graphs_available', True), \
             patch.object(native_tp, 'hc_graphs_enabled', False):
            self.assertIs(native_tp._hc_plan([], 1, 4, kind='glm'), plan)
        plan.set_graphs.assert_not_called()

    def test_old_extension_keeps_eager_plan_without_new_api(self):
        plan = object()
        with patch.object(native_plan, 'deepseek', return_value=plan), \
             patch.object(native_tp, '_hc_graphs_available', False), \
             patch.object(native_tp, 'hc_graphs_enabled', True):
            self.assertIs(native_tp._hc_plan([], 1, 4, kind='glm'), plan)

    def test_deepseek_keeps_eager_stages(self):
        plan = SimpleNamespace(set_graphs=Mock())
        with patch.object(native_plan, 'deepseek', return_value=plan), \
             patch.object(native_tp, '_hc_graphs_available', True), \
             patch.object(native_tp, 'hc_graphs_enabled', True):
            self.assertIs(native_tp._hc_plan([], 1, 6, kind='deepseek'), plan)
        plan.set_graphs.assert_not_called()


if __name__ == '__main__':
    unittest.main()
