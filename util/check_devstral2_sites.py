"""Exact actual-weight v2 stages and invalid-boundary mutation safety."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--extension-dir', type=Path, default=ROOT/'native-results/devstral2/extension')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--devices', type=int, nargs='+', default=[0, 1])
a = p.parse_args()
sys.path.insert(1, str(a.extension_dir.resolve()))
a.output.mkdir(parents=True, exist_ok=False)
import torch
from exllamav3 import Config, Model
from exllamav3.modules import TransformerBlock, native_plan
from exllamav3.ext import exllamav3_ext as ext


def reject(call, residual):
    before = residual.clone()
    try:
        call()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Invalid boundary accepted')
    assert torch.equal(before, residual), 'Rejected boundary mutated residual'


@torch.inference_mode()
def main():
    assert native_plan.v2_available
    cfg = Config.from_directory(r'D:\text-generation-webui\user_data\models\turboderp_Devstral-2-123B-Instruct-2512-exl3_4.00bpw')
    model = Model.from_config(cfg)
    torch.manual_seed(734)
    records = []
    for block in model.modules:
        if type(block) is not TransformerBlock:
            continue
        dev = torch.device('cuda', a.devices[block.layer_idx % len(a.devices)])
        norms = [getattr(block, n) for n in ('attn_norm', 'mlp_norm')]
        for n in norms:
            n.load(dev)
        for b, q in ((1, 1), (2, 1), (3, 1), (4, 1), (1, 5), (2, 6)):
            plan = native_plan.devstral(norms, b, q)
            x = torch.randn((b, q, 12288), device=dev)
            z = x.clone()
            expected = norms[0].forward(x, {}, out_dtype=torch.half)
            assert torch.equal(expected, plan.begin(z)), (block.key, b, q, 'begin')
            y = torch.randn_like(x)
            expected = norms[1].forward(y, {}, out_dtype=torch.half, residual_in=x)
            assert torch.equal(expected, plan.middle(z, y)), (block.key, b, q, 'middle')
            assert torch.equal(x, z)
            y = torch.randn_like(x)
            x += y
            plan.finish(z, y)
            assert torch.equal(x, z), (block.key, b, q, 'finish')
            for bad in (y[..., :32], y.half(), y.cpu(), y.transpose(1, 2)):
                reject(lambda: plan.finish(z, bad), z)
            if len(a.devices) > 1 and block.layer_idx == 0:
                bad = y.to(torch.device('cuda', a.devices[1]))
                reject(lambda: plan.finish(z, bad), z)
            reject(lambda: plan.middle(z, y.half()), z)
            reject(lambda: plan.finish(z, z), z)
            reject(lambda: plan.begin(z.half()), z)
            scratch = plan.begin(z)
            reject(lambda: plan.middle(z, scratch), z)
            if b != q:
                reject(lambda: plan.begin(z.view(q, b, 12288)), z)
            records.append(dict(layer=block.layer_idx, attention=type(block.attn).__name__,
                                batch=b, query=q, device=str(dev), exact=True, rejected_safely=True))
        for n in norms:
            n.unload()
    memory = [dict(device=i, peak=torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]
    assert all(r['peak'] == 0 for r in memory if r['device'] not in a.devices)
    (a.output/'sites.json').write_text(json.dumps(dict(records=records, memory=memory), indent=2))
    print(f'PASS {len(records)} exact Devstral sites and transactional rejections', flush=True)

if __name__ == '__main__': main()
