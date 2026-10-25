"""TP registration, collective boundaries and binding lifetime regressions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import warnings
import weakref
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from exllamav3.modules import native_tp, native_plan, TransformerBlock, transformer
from exllamav3.modules.native_devstral import get_native_block as devstral_get


class Owner:
    caps = {'supports_tp': True}


class TestRegistration(unittest.TestCase):
    def test_requires_existing_registration_and_upstream_capability(self):
        owner = Owner()
        block = NS(_native_plan_owner=weakref.ref(owner), _native_plan_get=devstral_get)
        self.assertEqual(native_tp.export_kind(block), 'devstral')
        owner.caps = {'supports_tp': False}
        self.assertIsNone(native_tp.export_kind(block))
        owner.caps = {'supports_tp': True}
        block._native_plan_get = lambda *args: None
        self.assertIsNone(native_tp.export_kind(block))
        del owner
        self.assertIsNone(native_tp.export_kind(block))
        self.assertIsNone(native_tp.export_kind(NS()))

    def test_export_import_only_transfers_registration(self):
        owner = Owner()
        block = TransformerBlock(None, 'block', layer_idx=1)
        block.device = torch.device('cpu')
        block._native_plan_owner = weakref.ref(owner)
        block._native_plan_get = devstral_get
        block._native_devstral = {'parent_pointer': object()}
        exported = block.tp_export({}, NS(send=lambda x: x))
        self.assertEqual(exported['native_plan_kind'], 'devstral')
        self.assertNotIn('_native_devstral', exported)
        backend = object()
        context = dict(device=7, rank=1, world_size=2, backend=backend,
                       consumer=NS(recv=lambda x, **kw: x))
        imported = TransformerBlock.tp_import(context, exported, {})
        self.assertIs(imported._native_plan_get, native_tp.get_native_block)
        self.assertEqual(imported._native_tp_device, torch.device('cuda:7'))
        self.assertIs(imported._native_tp_backend, backend)
        self.assertEqual(imported._native_tp_plans, {})
        self.assertFalse(hasattr(imported, '_native_plan_owner'))

    def test_unknown_registration_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unknown native TP'):
            native_tp.bind(NS(), 'new_architecture', {})

    def test_laguna_excluded_without_changing_single_gpu_registration(self):
        from exllamav3.modules.native_laguna import get_native_block
        owner = Owner()
        block = NS(_native_plan_owner=weakref.ref(owner), _native_plan_get=get_native_block)
        self.assertIsNone(native_tp.export_kind(block))
        self.assertIs(block._native_plan_get, get_native_block)

    def test_inference_tensor_binding_and_replacement(self):
        with torch.inference_mode():
            t = torch.ones(4)
            first = native_tp._tensor_key(t)
            self.assertEqual(first, native_tp._tensor_key(t))
            self.assertNotEqual(first, native_tp._tensor_key(t.clone()))
        t = torch.ones(4)
        first = native_tp._tensor_key(t)
        t.add_(1)
        self.assertNotEqual(first, native_tp._tensor_key(t))


class TestBoundaries(unittest.TestCase):
    def test_native_stages_keep_both_upstream_reductions(self):
        calls = []
        block = TransformerBlock(None, 'block', layer_idx=0)
        block._native_tp_kind = 'qwen'
        native_tp.clear(block)
        def stage(name, result=None):
            def run(*args):
                calls.append(name)
                return result
            return run
        x = torch.zeros(1, 1, 4, 8)
        y = torch.ones(1, 1, 8)
        plan = NS(begin=stage('begin', y), middle=stage('middle', y), finish=stage('finish'))
        block._native_plan_get = lambda *args: plan
        block.attn = NS(forward=stage('attention and upstream reduction', y))
        block.mlp = NS(forward=stage('experts and upstream reduction', y))
        params = {'gr_weighted': object()}
        with patch.object(native_plan, 'enabled', True), \
             patch.object(transformer, '_native_qwen_enabled', False), \
             patch.object(transformer, '_native_glm_enabled', False):
            self.assertIs(block.forward(x, params), x)
        self.assertEqual(calls, ['begin', 'attention and upstream reduction', 'middle',
                                 'experts and upstream reduction', 'finish'])
        self.assertNotIn('gr_weighted', params)
        self.assertEqual(block._native_tp_stats['executions'], 1)
        self.assertEqual(block._native_tp_stats['operations'], 4)

    def test_no_execution_credit_for_failure(self):
        block = TransformerBlock(None, 'block')
        block._native_tp_kind = 'qwen'
        native_tp.clear(block)
        block._native_plan_get = lambda *a: NS(begin=lambda x: (_ for _ in ()).throw(RuntimeError('kernel failed')))
        block.attn = NS(forward=lambda x,p: x)
        with patch.object(native_plan, 'enabled', True), \
             patch.object(transformer, '_native_qwen_enabled', False), \
             patch.object(transformer, '_native_glm_enabled', False):
            with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                block.forward(torch.zeros(1,1,4,8), {})
        self.assertEqual(block._native_tp_stats['executions'], 0)

    def test_dense_operation_counts_match_emitted_instructions(self):
        n = NS(weight=torch.ones(8), rms_norm_eps=1e-6, constant_bias=0., constant_scale=1.)
        for kind, count in (('devstral',2), ('gemma',4)):
            with patch.object(native_plan.ext, 'BC_BlockPlanV2') as builder:
                getattr(native_plan, kind)([n]*count, 1, 1)
                stages = builder.call_args.args[3]
            self.assertEqual(native_tp._OP_COUNTS[kind], sum(map(len, stages)))

    def test_cache_rebuild_and_clear(self):
        b = NS()
        native_tp.clear(b)
        x = torch.zeros(1,1,8)
        first = native_tp._cached(b, x, ('weight_a',), object)
        self.assertIs(native_tp._cached(b, x, ('weight_a',), object), first)
        self.assertIsNot(native_tp._cached(b, x, ('weight_b',), object), first)
        self.assertEqual(b._native_tp_stats['builds'], 2)
        native_tp.clear(b)
        self.assertEqual(b._native_tp_plans, {})
        self.assertEqual(b._native_tp_stats['builds'], 0)

    def test_backend_mismatch_is_visible_and_warmup_is_excluded(self):
        b = NS()
        native_tp.bind(b, 'devstral', dict(device=5, rank=0, world_size=2, backend=object()))
        with patch.object(native_plan, 'enabled', True):
            self.assertIsNone(native_tp.get_native_block(b, None, {'tp_warmup': True}))
            self.assertEqual(b._native_tp_stats['fallbacks'], {})
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always')
                for _ in range(2):
                    self.assertIsNone(native_tp.get_native_block(b, None, {'backend': object()}))
                self.assertEqual(len(messages), 1)
                self.assertIn('stock fallback', str(messages[0].message))
            self.assertEqual(b._native_tp_stats['fallbacks']['missing or changed upstream TP backend'], 2)

    def test_shape_and_old_extension_fallback_before_plan_mutation(self):
        b = NS(attn=NS(tp_reduce=True), mlp=NS(tp_reduce=True), attn_resid_scalar=None, mlp_resid_scalar=None)
        backend = object()
        native_tp.bind(b, 'devstral', dict(device=5, rank=0, world_size=2, backend=backend))
        x = NS(device=torch.device('cuda:5'), dtype=torch.float32, ndim=3,
               shape=(1,7,16), is_contiguous=lambda: True)
        with patch.object(native_plan, 'enabled', True), warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            self.assertIsNone(native_tp.get_native_block(b, x, {'backend':backend}))
            x.shape = (1,1,16)
            with patch.object(native_plan, 'v2_available', False):
                self.assertIsNone(native_tp.get_native_block(b, x, {'backend':backend}))
            self.assertEqual(len(notices), 2)
        self.assertEqual(b._native_tp_plans, {})
        self.assertEqual(b._native_tp_stats['executions'], 0)

    def test_diagnostics_snapshot_and_dispatch_order(self):
        from exllamav3.model.model_tp import Model_TPMixin
        model = Model_TPMixin()
        with self.assertRaisesRegex(RuntimeError, 'loaded TP'):
            model.native_plan_tp_diagnostics()
        model.loaded_tp, model.active_devices = True, [7,3]
        with patch.object(model, 'tp_worker_dispatch_wait_multi', return_value=['rank0','rank1']) as dispatch:
            self.assertEqual(model.native_plan_tp_diagnostics(), ['rank0','rank1'])
            dispatch.assert_called_once_with([7,3], native_tp.diagnostics, ())
        b = NS(key='b0', _native_tp_kind='devstral', _native_tp_plans={},
               _native_tp_stats=dict(executions=1, operations=4, builds=1, fallbacks=Counter(shape=2)),
               attn=NS(num_q_heads=2), mlp=NS(intermediate_size=128))
        context = dict(device=7, rank=0, world_size=2, backend=object(), modules=[b])
        with patch.object(torch.cuda, 'synchronize') as sync, \
             patch.object(torch.cuda, 'get_device_name', return_value='test GPU'), \
             patch.object(torch.cuda, 'get_device_properties', return_value=NS(uuid='test UUID')), \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=64):
            result = native_tp.diagnostics(context)
            sync.assert_called_once_with(7)
        b._native_tp_stats['fallbacks']['shape'] += 1
        self.assertEqual(result['blocks'][0]['fallbacks']['shape'], 2)
        self.assertEqual(result['blocks'][0]['operations'], 4)

    def test_block_norm_exports_fallback_but_final_mtp_norm_does_not(self):
        b = NS(attn=NS(tp_reduce=True), mlp=NS(tp_reduce=True),
               attn_norm=NS(key='block.norm'), attn_resid_scalar=None, mlp_resid_scalar=None)
        backend = object()
        native_tp.bind(b, 'devstral', dict(device=5, rank=0, world_size=2, backend=backend))
        x = NS(device=torch.device('cuda:5'), dtype=torch.float32, ndim=3,
               shape=(1,1,16), is_contiguous=lambda: True)
        with patch.object(native_plan, 'enabled', True), \
             patch.object(native_plan, 'v2_available', True), \
             patch.object(native_plan, 'v2_residual_available', True), \
             patch.object(native_tp, '_dense', return_value='plan') as dense:
            with self.assertWarnsRegex(RuntimeWarning, 'block-local norm state export'):
                self.assertIsNone(native_tp.get_native_block(b, x,
                    dict(backend=backend, export_state_norm_keys={'block.norm'})))
            dense.assert_not_called()
            self.assertEqual(native_tp.get_native_block(b, x,
                dict(backend=backend, export_state_norm_keys={'final.norm'})), 'plan')

    def test_qwen_batched_verification_is_reported_but_batched_decode_is_native(self):
        b = NS(attn=NS(tp_reduce=True), mlp=NS(tp_reduce=True), attn_resid_scalar=None,
               mlp_resid_scalar=None, attn_post_norm=None, mlp_post_norm=None, layer_scalar_f=None)
        backend = object()
        native_tp.bind(b, 'qwen', dict(device=5, rank=0, world_size=2, backend=backend))
        x = NS(device=torch.device('cuda:5'), dtype=torch.float32, ndim=4,
               shape=(2,5,4,4096), is_contiguous=lambda: True)
        with patch.object(native_plan, 'enabled', True), patch.object(native_plan, 'available', True), \
             patch.object(native_tp, '_streams', return_value='plan') as streams:
            with self.assertWarnsRegex(RuntimeWarning, 'small-row IR domain'):
                self.assertIsNone(native_tp.get_native_block(b, x, {'backend':backend}))
            streams.assert_not_called()
            x.shape = (4,1,4,4096)
            self.assertEqual(native_tp.get_native_block(b, x, {'backend':backend}), 'plan')


if __name__ == '__main__':
    unittest.main()
