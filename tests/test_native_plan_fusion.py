"""Run the full HC graph safety contract with upstream folds selected."""
import unittest
import torch
import test_native_plan_graphs as baseline


@unittest.skipUnless(hasattr(getattr(baseline.ext, 'BC_BlockPlan', object), 'set_hc_fusion'),
                     'requires HC fusion experiment')
class TestFusedPlanGraphs(baseline.TestPlanGraphs):
    def plan(self, weights, b=1, q=4):
        plan = super().plan(weights, b, q)
        plan.set_hc_fusion(True)
        return plan

    def weights(self, device):
        weights = super().weights(device)
        for site in weights:
            site[3] = site[3].to(getattr(self, 'norm_dtype', torch.bfloat16))
        return weights

    def test_replay_matches_eager_at_every_boundary(self):
        for dtype in (torch.float16, torch.bfloat16):
            self.norm_dtype = dtype
            with self.subTest(norm_dtype=dtype):
                super().test_replay_matches_eager_at_every_boundary()

    def test_switch_keeps_both_captures_and_respects_native_disable(self):
        for device in range(torch.cuda.device_count()):
            weights = self.weights(device)
            reference = super().plan(weights)
            plan = self.plan(weights)
            plan.set_graphs(True)
            retained = []
            for fused in (False, True, False, True):
                plan.set_hc_fusion(fused)
                for dtype in (torch.float16, torch.float32):
                    x = torch.randn(1,4,4,4096,device=device)
                    a,b = x.clone(),x.clone()
                    y = torch.randn(1,4,4096,device=device,dtype=dtype)
                    retained.extend((a,b,y))
                    self.assertTrue(torch.equal(reference.begin(a),plan.begin(b)))
                    self.assertTrue(torch.equal(reference.middle(a,y),plan.middle(b,y)))
                    reference.finish(a,y)
                    plan.finish(b,y)
                    self.assertTrue(torch.equal(a,b))
            self.assertEqual(plan.graph_captures,6)
            self.assertEqual(plan.fused_graph_captures,3)
            self.assertEqual(plan.graph_replays,16)
            self.assertEqual(plan.fused_graph_replays,8)
            plan.set_graphs(False)
            self.assertTrue(torch.equal(reference.begin(a),plan.begin(b)))
            self.assertEqual(plan.graph_replays,16)


if __name__ == '__main__':
    unittest.main()
