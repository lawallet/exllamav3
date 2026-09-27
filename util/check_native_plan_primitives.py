"""Exact GLM stage oracle across all production shapes, without loading a model."""
import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--extension-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(a.extension_dir.resolve()))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import native_plan

torch.manual_seed(1234)
native_plan.enabled = True
results = []
for b, q in [(b, 1) for b in range(1, 5)] + [(b, 4) for b in range(1, 4)]:
    weights = []
    for site in range(2):
        weights.extend([torch.randn(24, 16384, device='cuda', dtype=torch.float16) * .001,
                        torch.randn(24, device='cuda'), torch.randn(3, device='cuda'),
                        torch.randn(4096, device='cuda', dtype=torch.float16)])
    old = ext.BC_GlmBlock(weights, 1e-5, 1e-6, 20, q, b)
    plan = native_plan.glm(weights, b, q)
    x = torch.randn(b, q, 4, 4096, device='cuda')
    ref = x.clone()
    assert torch.equal(old.begin(ref), plan.begin(x))
    y = torch.randn(b, q, 4096, device='cuda', dtype=torch.float16)
    assert torch.equal(old.middle(ref, y), plan.middle(x, y))
    assert torch.equal(ref, x)
    old.finish(ref, y); plan.finish(x, y)
    assert torch.equal(ref, x)
    before = x.clone()
    if b != q:
        try:
            plan.begin(x.view(q, b, 4, 4096))
        except RuntimeError:
            pass
        else:
            raise AssertionError('B and Q were treated as interchangeable')
    try:
        plan.middle(x, y[:, :, :-1])
    except RuntimeError:
        pass
    else:
        raise AssertionError('invalid boundary accepted')
    assert torch.equal(before, x), 'invalid boundary mutated residual'
    results.append(dict(batch=b, query=q, exact=True, invalid_boundary_transactional=True, batch_query_distinct=True))
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(results, indent=2))
print(json.dumps(results))
