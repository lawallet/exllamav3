"""1.5.4 GR operand semantics: native IR must match linked stock sites exactly."""
import unittest
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.hyperconnections import GatedResidual
from exllamav3.modules.native_plan import GR_MIX, APPLY


@unittest.skipUnless(getattr(ext, 'native_qwen_unfolded_gr', 0) == 1 and torch.cuda.is_available(),
                     'requires rebuilt 1.5.4 native extension and CUDA')
class TestUnfoldedGR(unittest.TestCase):
    @torch.inference_mode()
    def test_linked_stock_and_native_stages(self):
        torch.manual_seed(154)
        device = torch.device('cuda:0')
        hidden, rank = 2560, 320
        sites = []
        for i in range(2):
            site = GatedResidual(None, f'site{i}', 4, hidden, 1e-6, use_combine=True)
            site.device = device
            site.norm_w_raw = torch.randn(4*hidden, device=device)*.1
            site._prepare(torch.randn(rank, 4*hidden, device=device)*.01,
                          torch.randn(4*hidden, rank, device=device)*.01,
                          torch.randn(4, 4*hidden, device=device)*.01, False)
            sites.append(site)
        sites[0].next_site = sites[1]
        weights = [[s.proj_h[:s.proj_m], s.upx_h, s.w_h] for s in sites]
        for batch, query in ((1, 1), (2, 1), (3, 1), (4, 1), (1, 5)):
            with self.subTest(batch=batch, query=query):
                plan = ext.BC_BlockPlan(weights, [GR_MIX]*2, [1e-6]*2, [0.]*2, [0]*2,
                    [[(GR_MIX, 0)], [(APPLY, 0), (GR_MIX, 1)], [(APPLY, 1)]],
                    [], None, batch, query)
                x = torch.randn(batch, query, 4, hidden, device=device)
                reference = x.clone()
                params = {}
                post, comb, mixed = sites[0].mix(reference, params)
                self.assertTrue(torch.equal(plan.begin(x), mixed))
                y = torch.randn(batch, query, hidden, device=device, dtype=torch.half)
                sites[0].apply_(reference, y, post, comb, params)
                self.assertIn('gr_weighted', params)
                post, comb, mixed = sites[1].mix(reference, params)
                self.assertTrue(torch.equal(plan.middle(x, y), mixed))
                self.assertTrue(torch.equal(x, reference))
                sites[1].apply_(reference, y, post, comb, params)
                plan.finish(x, y)
                self.assertTrue(torch.equal(x, reference))


if __name__ == '__main__':
    unittest.main()
