"""Cold TP startup must finish before a generator can claim cache pages."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from exllamav3.model.model_tp import Model_TPMixin, DISPATCH_TIMEOUT, WARMUP_TIMEOUT, mp_model_forward
from exllamav3.generator.generator import Generator


class TestWarmupDispatch(unittest.TestCase):
    def make_model(self, fail=False):
        events=[]
        model=Model_TPMixin()
        model.active_devices=[1,0]
        model.tp_output_device=0
        model.tp_drain_acks=lambda: events.append('drain')
        model.prepare_inputs_for_tp=lambda x,p: ('shared', {'saved':True})
        model.restore_tp_params=lambda p,r: events.append(('restore',r))
        def dispatch(device,fn,args,*,timeout):
            events.append((device,fn,args,timeout))
            if fail: raise TimeoutError('compile timed out')
            return 'output' if device == 0 else None
        model.tp_worker_dispatch_single=dispatch
        return model,events

    def test_serial_rank_calls_without_cpu_reducer_and_restore(self):
        model,events=self.make_model()
        params={'tp_warmup':True}
        self.assertEqual(model.forward_tp('input',params,7,[]),'output')
        self.assertEqual(events,['drain',
            (1,mp_model_forward,('shared',params,7,False,None),WARMUP_TIMEOUT),
            (0,mp_model_forward,('shared',params,7,False,None),WARMUP_TIMEOUT),
            ('restore',{'saved':True})])
        self.assertEqual(model.tp_pending_acks,[])
        self.assertIsNone(model.tp_pending_refs)

    def test_failure_restores_params_without_dispatching_next_rank(self):
        model,events=self.make_model(fail=True)
        with self.assertRaisesRegex(TimeoutError,'compile timed out'):
            model.forward_tp('input',{'tp_warmup':True},7,[])
        self.assertEqual(len(events),3)
        self.assertEqual(events[-1],('restore',{'saved':True}))

    def test_compile_budget_does_not_change_normal_rpc_budget(self):
        model=Model_TPMixin()
        conn=Mock()
        conn.poll.return_value=True
        conn.recv.return_value='ok'
        model.mp_parent_conn=[conn]
        for timeout in (None,WARMUP_TIMEOUT,None):
            self.assertEqual(model.tp_worker_dispatch_single(0,'fn',(),timeout=timeout),'ok')
        self.assertEqual([c.args[0] for c in conn.poll.call_args_list],
                         [DISPATCH_TIMEOUT,WARMUP_TIMEOUT,DISPATCH_TIMEOUT])


class TestGeneratorWarmup(unittest.TestCase):
    def make_model(self):
        model=Model_TPMixin()
        model.loaded_tp=True
        model.tp_warmup_token=object()
        model.warmup=Mock(return_value=[])
        return model,NS()

    def test_once_per_cache_workload_and_reload(self):
        model,cache=self.make_model()
        for _ in range(2): model.warmup_tp_for_generator(cache,2048,4,8)
        self.assertEqual(model.warmup.call_count,1)
        model.warmup_tp_for_generator(cache,256,4,8)
        self.assertEqual(model.warmup.call_count,2)
        model.tp_warmup_token=object()
        model.warmup_tp_for_generator(cache,256,4,8)
        self.assertEqual(model.warmup.call_count,3)
        model.warmup_tp_for_generator(NS(),256,4,8)
        self.assertEqual(model.warmup.call_count,4)

    def test_no_warmup_for_non_tp(self):
        model,cache=self.make_model()
        model.loaded_tp=False
        model.warmup_tp_for_generator(cache,2048,4,8)
        model.warmup.assert_not_called()

    def test_large_scheduler_capacity_does_not_create_huge_warmup_batch(self):
        model,cache=self.make_model()
        model.warmup_tp_for_generator(cache,2048,256,8)
        self.assertEqual(model.warmup.call_args.kwargs['max_batch_size'],8)

    def test_failed_warmup_is_not_cached(self):
        model,cache=self.make_model()
        model.warmup.return_value=['chunk 2048']
        with self.assertRaisesRegex(RuntimeError,'generation was not started'):
            model.warmup_tp_for_generator(cache,2048,4,8)
        self.assertFalse(hasattr(cache,'_tp_warmup_signature'))

    def test_generator_warms_before_cache_ownership_and_includes_draft_span(self):
        model=NS(config=NS(vocab_size=128),loaded_tp=True,
                 warmup_tp_for_generator=Mock(side_effect=RuntimeError('stop after warmup')))
        cache=NS(max_num_tokens=65536)
        draft=NS(caps={'default_draft_size':15})
        draft_cache=NS(max_num_tokens=65536)
        with patch('exllamav3.generator.generator.PageTable') as pages:
            with self.assertRaisesRegex(RuntimeError,'stop after warmup'):
                Generator(model,cache,None,max_batch_size=4,draft_model=draft,draft_cache=draft_cache)
            pages.assert_not_called()
        model.warmup_tp_for_generator.assert_called_once_with(cache,2048,4,16)
        self.assertFalse(hasattr(cache,'owner_serial'))


if __name__ == '__main__': unittest.main()
