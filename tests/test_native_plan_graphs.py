"""Exact HC stage replay with changing boundary storage and dtypes."""
import unittest

import torch
from exllamav3.ext import exllamav3_ext as ext


@unittest.skipUnless(torch.cuda.is_available() and hasattr(getattr(ext, 'BC_BlockPlan', object), 'set_graphs'),
                     'requires the experimental graph-capable extension and CUDA')
class TestPlanGraphs(unittest.TestCase):
    def weights(self, device):
        return [[torch.randn(24, 16384, device=device, dtype=torch.float16)*0.01,
                 torch.randn(24, device=device, dtype=torch.float32)*0.1,
                 torch.ones(3, device=device, dtype=torch.float32),
                 torch.randn(4096, device=device, dtype=torch.bfloat16)*0.01+1]
                for _ in range(2)]

    def plan(self, weights, b=1, q=4):
        return ext.BC_BlockPlan(weights, [1,1], [1e-5,1e-5], [1e-6,1e-6], [20,20],
            [[(1,0),(3,0)], [(2,0),(1,1),(3,1)], [(2,1)]], [], None, b, q)

    def test_replay_matches_eager_at_every_boundary(self):
        torch.manual_seed(731)
        for device in range(torch.cuda.device_count()):
            weights = self.weights(device)
            for b,q in ((b,q) for b in range(1,5) for q in range(1,7) if b*q<=12):
                with self.subTest(device=device, batch=b, query=q):
                    eager, graphed = self.plan(weights,b,q), self.plan(weights,b,q)
                    graphed.set_graphs(True)
                    # Retain old boundaries so the allocator cannot reuse their
                    # addresses and accidentally hide a missing graph pointer patch.
                    retained = []
                    for dtype in (torch.float16, torch.float32, torch.float16):
                        x = torch.randn(b,q,4,4096,device=device)
                        reference, actual = x.clone(), x.clone()
                        y = torch.randn(b,q,4096,device=device,dtype=dtype)
                        z = torch.randn(b,q,4096,device=device,dtype=torch.float32)
                        retained.extend((reference,actual,y,z))
                        self.assertTrue(torch.equal(eager.begin(reference),graphed.begin(actual)))
                        self.assertTrue(torch.equal(eager.middle(reference,y),graphed.middle(actual,y)))
                        self.assertTrue(torch.equal(reference,actual))
                        eager.finish(reference,z)
                        graphed.finish(actual,z)
                        self.assertTrue(torch.equal(reference,actual))
                    self.assertEqual(graphed.graph_captures,3)
                    self.assertEqual(graphed.graph_replays,6)
                    graphed.set_graphs(False)
                    x = torch.randn(b,q,4,4096,device=device)
                    self.assertTrue(torch.equal(eager.begin(x.clone()),graphed.begin(x.clone())))
                    self.assertEqual(graphed.graph_replays,6)
                    graphed.set_graphs(True)
                    self.assertTrue(torch.equal(eager.begin(x.clone()),graphed.begin(x.clone())))
                    self.assertEqual(graphed.graph_captures,3)

    def test_invalid_dynamic_boundary_does_not_mutate(self):
        weights = self.weights(0)
        plan = self.plan(weights)
        plan.set_graphs(True)
        x = torch.randn(1,4,4,4096,device=0)
        plan.begin(x)
        original = x.clone()
        with self.assertRaises(RuntimeError):
            plan.middle(x, torch.zeros(1,4,4097,device=0))
        self.assertTrue(torch.equal(x,original))
        self.assertEqual(plan.graph_captures,1)

    def test_rebound_fixed_storage_rejects_before_mutation(self):
        weights = self.weights(0)
        plan = self.plan(weights)
        plan.set_graphs(True)
        x = torch.randn(1,4,4,4096,device=0)
        plan.begin(x)
        original = x.clone()
        # Late-stage binding must be checked before the attention APPLY writes x.
        weights[1][0].set_(torch.zeros_like(weights[1][0]))
        with self.assertRaisesRegex(RuntimeError,'changed fixed binding'):
            plan.middle(x, torch.zeros(1,4,4096,device=0))
        self.assertTrue(torch.equal(x,original))

    def test_changed_fixed_layout_rejects_before_mutation(self):
        weights = self.weights(0)
        plan = self.plan(weights)
        plan.set_graphs(True)
        x = torch.randn(1,4,4,4096,device=0)
        plan.begin(x)
        original = x.clone()
        weights[1][0].resize_(24,16383)
        with self.assertRaisesRegex(RuntimeError,'changed fixed binding'):
            plan.middle(x, torch.zeros(1,4,4096,device=0))
        self.assertTrue(torch.equal(x,original))

    def test_changed_exposed_scratch_rejects_before_mutation(self):
        plan = self.plan(self.weights(0))
        plan.set_graphs(True)
        x = torch.randn(1,4,4,4096,device=0)
        output = plan.begin(x)
        original = x.clone()
        output.resize_(1,4,4097)
        with self.assertRaisesRegex(RuntimeError,'changed fixed binding'):
            plan.middle(x, torch.zeros(1,4,4096,device=0))
        self.assertTrue(torch.equal(x,original))

    def test_unsupported_norm_rejected_at_construction(self):
        for device in range(torch.cuda.device_count()):
            for site in (0, 1):
                for invalid in ('dtype', 'shape', 'layout'):
                    weights = self.weights(device)
                    w = weights[site][3]
                    weights[site][3] = (w.float() if invalid == 'dtype' else
                                       w.view(1,4096) if invalid == 'shape' else
                                       torch.stack((w,w), dim=1)[:,0])
                    for factory in (self.plan, self.glm_plan):
                        with self.subTest(device=device, site=site, invalid=invalid, factory=factory.__name__):
                            with self.assertRaises(RuntimeError):
                                factory(weights)
            self.assertEqual(torch.ones(1,device=device).sum().item(), 1)

    def glm_plan(self, weights):
        return ext.BC_GlmBlock(weights[0]+weights[1], 1e-5, 1e-6, 20, 4, 1)

    def test_rebound_norm_rejects_before_residual_mutation(self):
        for device in range(torch.cuda.device_count()):
            for mode in ('eager', 'graphs', 'glm'):
                for invalid in ('dtype', 'shape', 'layout'):
                    with self.subTest(device=device, mode=mode, invalid=invalid):
                        weights = self.weights(device)
                        plan = self.glm_plan(weights) if mode == 'glm' else self.plan(weights)
                        if mode == 'graphs':
                            plan.set_graphs(True)
                        x = torch.randn(1,4,4,4096,device=device)
                        plan.begin(x)
                        original = x.clone()
                        w = weights[1][3]
                        w.data = (w.float() if invalid == 'dtype' else
                                  w.view(1,4096) if invalid == 'shape' else
                                  torch.stack((w,w), dim=1)[:,0])
                        for call in (lambda: plan.begin(x),
                                     lambda: plan.middle(x, torch.zeros(1,4,4096,device=device)),
                                     lambda: plan.finish(x, torch.zeros(1,4,4096,device=device))):
                            with self.assertRaises(RuntimeError):
                                call()
                            self.assertTrue(torch.equal(x, original))

    def test_supported_norm_dtypes_match_bespoke_and_graphs(self):
        for device in range(torch.cuda.device_count()):
            for dtype in (torch.float16, torch.bfloat16):
                weights = self.weights(device)
                for site in weights:
                    site[3] = site[3].to(dtype)
                plans = [self.glm_plan(weights), self.plan(weights), self.plan(weights)]
                plans[-1].set_graphs(True)
                x = torch.randn(1,4,4,4096,device=device)
                residuals = [x.clone() for _ in plans]
                outputs = [p.begin(r).clone() for p,r in zip(plans,residuals)]
                self.assertTrue(all(torch.equal(outputs[0],o) for o in outputs[1:]))
                y = torch.randn(1,4,4096,device=device)
                outputs = [p.middle(r,y).clone() for p,r in zip(plans,residuals)]
                self.assertTrue(all(torch.equal(outputs[0],o) for o in outputs[1:]))
                for p,r in zip(plans,residuals):
                    p.finish(r,y)
                self.assertTrue(all(torch.equal(residuals[0],r) for r in residuals[1:]))

    def test_destroy_uses_original_device_after_binding_replacement(self):
        weights = self.weights(0)
        plan = self.plan(weights)
        plan.set_graphs(True)
        x = torch.randn(1,4,4,4096,device=0)
        plan.begin(x)
        weights[0][0].data = weights[0][0].cpu()
        with self.assertRaises(RuntimeError):
            plan.begin(x)
        # Destruction must use the graph's captured device, not mutable weights.
        with torch.cuda.device(torch.cuda.device_count()-1):
            del plan
        self.assertEqual(torch.ones(1,device=0).sum().item(), 1)

    def test_current_stream_noncontiguous_boundary_and_weight_updates(self):
        weights = self.weights(0)
        weights[0][3] = weights[0][3].half()
        eager, graphed = self.plan(weights), self.plan(weights)
        graphed.set_graphs(True)
        stream = torch.cuda.Stream(device=0)
        stream.wait_stream(torch.cuda.current_stream(0))
        with torch.cuda.stream(stream):
            for _ in range(2):
                weights[0][1].add_(0.1)
                x = torch.randn(1,4,4,4096,device=0)
                reference, actual = x.clone(), x.clone()
                y = torch.randn(1,4,4096,2,device=0)[...,0]
                self.assertFalse(y.is_contiguous())
                self.assertTrue(torch.equal(eager.begin(reference), graphed.begin(actual)))
                self.assertTrue(torch.equal(eager.middle(reference,y), graphed.middle(actual,y)))
                self.assertTrue(torch.equal(reference,actual))
        stream.synchronize()
        self.assertEqual(graphed.graph_captures,2)
        self.assertEqual(graphed.graph_replays,4)


if __name__ == '__main__':
    unittest.main()
