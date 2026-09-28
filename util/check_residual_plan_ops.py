"""Generic residual op parity and preflight safety, independent of architecture."""
import argparse
import json
from pathlib import Path
import sys
p = argparse.ArgumentParser()
p.add_argument('--extension-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(a.extension_dir.resolve())]
import torch
from exllamav3.ext import exllamav3_ext as ext
assert ext.native_block_plan_v2_residual_ops == 1
specs = [(0, 1), (1, 1), (2, 1), (-1, 0)]
code = [[(3, 0, 3, 0)], [(7, 1, 3, 0)], [(8, 2, 0, -1)]]
def make(w, program=code):
    return ext.BC_BlockPlanV2(specs, [w], [[1e-5, .25, .75]], program, [3, 3, -1], 2, 6, 256)
def reject(fn, x):
    before = x.clone()
    try: fn()
    except RuntimeError: pass
    else: raise AssertionError('Invalid contract accepted')
    assert torch.equal(x, before)
records = []
with torch.inference_mode():
    for dev in (0, 1):
        for dtype in (torch.half, torch.bfloat16):
            w = torch.randn(256, device=f'cuda:{dev}', dtype=dtype)
            x = torch.randn(2, 6, 256, device=w.device)
            y = torch.randn_like(x)
            reference = x.clone()
            output = torch.empty_like(x, dtype=torch.half)
            plan = make(w)
            ext.rms_norm_res_in(y.view(12,256), w, output.view(12,256), reference.view(12,256), 1e-5, .25, .75)
            assert torch.equal(plan.middle(x, y), output) and torch.equal(x, reference)
            reference += y
            plan.finish(x, y)
            assert torch.equal(x, reference)
            for bad in (y.half(), y.cpu(), y.to(f'cuda:{1-dev}'), y[:1], y.transpose(0,1), x):
                reject(lambda: plan.middle(x, bad), x)
                reject(lambda: plan.finish(x, bad), x)
            for bad in ([code[0], [(7,0,3,0)], code[2]],
                        [code[0], [(7,1,0,0)], code[2]],
                        [code[0], code[1], [(8,2,0,0)]],
                        [code[0], code[1], [(8,2,3,-1)]]):
                reject(lambda: make(w, bad), x)
            w.resize_(128)
            reject(lambda: plan.middle(x, y), x)
            reject(lambda: plan.finish(x, y), x)
            records.append(dict(device=dev, dtype=str(dtype), exact=True, rejection_safe=True))
a.output.parent.mkdir(parents=True, exist_ok=True)
assert not a.output.exists()
a.output.write_text(json.dumps(records, indent=2))
print('PASS generic residual operations', flush=True)
