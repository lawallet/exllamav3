"""Graph flags are process-cached: exercise real kernels in fresh interpreters."""
import itertools
import os
import subprocess
import sys
import textwrap
import unittest

import torch
from exllamav3.ext import exllamav3_ext as ext


@unittest.skipUnless(torch.cuda.is_available() and hasattr(getattr(ext, 'BC_BlockPlan', object), 'set_graphs'),
                     'requires the graph-capable extension and CUDA')
class TestGraphPolicyProcess(unittest.TestCase):
    def test_global_and_native_flags(self):
        code = textwrap.dedent('''
            import os
            from types import SimpleNamespace as NS
            import torch
            from exllamav3.ext import exllamav3_ext as ext
            from exllamav3.modules import native_tp
            from test_native_plan_graphs import TestPlanGraphs

            factory = TestPlanGraphs()
            global_on = (os.environ.get('EXL3_GRAPHS') == '1' if torch.version.hip
                         else os.environ.get('EXL3_GRAPHS', '1') != '0')
            native_on = os.environ.get('EXL3_NATIVE_TP_GRAPHS', '1') == '1'
            fusion_on = os.environ.get('EXL3_NATIVE_TP_HC_FUSION', '0') == '1'
            torch.manual_seed(819)
            for device in range(torch.cuda.device_count()):
                for kind in ('direct', 'glm', 'deepseek'):
                    weights = factory.weights(device)
                    sites = [(NS(fn_h=w[0],base=w[1],scale=w[2],rms_eps=1e-5,
                                 hc_eps=1e-6,sinkhorn_iters=20), NS(weight=w[3])) for w in weights]
                    eager = factory.plan(weights)
                    if kind == 'direct':
                        candidate = factory.plan(weights)
                        candidate.set_graphs(True)
                        if hasattr(candidate, 'set_hc_fusion'):
                            candidate.set_hc_fusion(fusion_on)
                    else:
                        candidate = native_tp._hc_plan(sites,1,4,kind=kind)
                    retained = []
                    for dtype in (torch.float16, torch.float32):
                        x = torch.randn(1,4,4,4096,device=device)
                        a,b = x.clone(),x.clone()
                        y = torch.randn(1,4,4096,device=device,dtype=dtype)
                        retained.extend((a,b,y))
                        assert torch.equal(eager.begin(a),candidate.begin(b))
                        assert torch.equal(eager.middle(a,y),candidate.middle(b,y))
                        eager.finish(a,y)
                        candidate.finish(b,y)
                        assert torch.equal(a,b)
                    expected = global_on and (kind == 'direct' or (kind == 'glm' and native_on))
                    assert candidate.graph_captures == (3 if expected else 0), (kind, candidate.graph_captures)
                    assert candidate.graph_replays == (4 if expected else 0), (kind, candidate.graph_replays)
                    if hasattr(candidate, 'fused_graph_replays'):
                        assert candidate.fused_graph_replays == (4 if expected and fusion_on else 0)
                        assert candidate.fused_graph_captures == (3 if expected and fusion_on else 0)
            print('graph policy passed')
        ''')
        # Use the same source tree and isolated extension selected by the parent.
        code = 'import sys\nsys.path[:0] = ' + repr(sys.path) + '\n' + code
        code += '\nassert ext.__file__ == ' + repr(ext.__file__) + '\n'
        for global_flag,native_flag,fusion_flag in itertools.product((None,'0','1'), repeat=3):
            with self.subTest(global_flag=global_flag,native_flag=native_flag,fusion_flag=fusion_flag):
                env = os.environ.copy()
                for key,value in (('EXL3_GRAPHS',global_flag),('EXL3_NATIVE_TP_GRAPHS',native_flag),
                                  ('EXL3_NATIVE_TP_HC_FUSION',fusion_flag)):
                    if value is None:
                        env.pop(key,None)
                    else:
                        env[key] = value
                result = subprocess.run([sys.executable,'-c',code], env=env,
                                        capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('graph policy passed',result.stdout)


if __name__ == '__main__':
    unittest.main()
