"""Exact actual-weight v2 stages and invalid-boundary mutation safety."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--extension-dir', type=Path, default=ROOT/'native-results/block-ir-v2/extension-r2')
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
    cfg = Config.from_directory(r'D:\text-generation-webui\user_data\models\turboderp_gemma-4-31b-it-exl3_6.00bpw')
    model = Model.from_config(cfg)
    torch.manual_seed(734)
    records = []
    for block in model.modules:
        if type(block) is not TransformerBlock:
            continue
        dev = torch.device('cuda', a.devices[block.layer_idx % len(a.devices)])
        norms = [getattr(block, n) for n in ('attn_norm', 'attn_post_norm', 'mlp_norm', 'mlp_post_norm')]
        for n in norms:
            n.load(dev)
        for b, q in ((1, 1), (2, 1), (3, 1), (4, 1), (1, 5), (2, 6)):
            plan = native_plan.gemma(norms, b, q)
            x = torch.randn((b, q, 5376), device=dev)
            z = x.clone()
            expected = norms[0].forward(x, {}, out_dtype=torch.half)
            assert torch.equal(expected, plan.begin(z)), (block.key, b, q, 'begin')
            y = torch.randn_like(x, dtype=torch.half)
            norms[1].forward(y, {}, residual=x)
            expected = norms[2].forward(x, {}, out_dtype=torch.half)
            assert torch.equal(expected, plan.middle(z, y)), (block.key, b, q, 'middle')
            assert torch.equal(x, z)
            y = torch.randn_like(x)
            norms[3].forward(y, {}, residual=x)
            plan.finish(z, y)
            assert torch.equal(x, z), (block.key, b, q, 'finish')
            for bad in (y[..., :32], y.half(), y.cpu(), y.transpose(1, 2)):
                reject(lambda: plan.finish(z, bad), z)
            if len(a.devices) > 1 and block.layer_idx == 0:
                bad = y.to(torch.device('cuda', a.devices[1]))
                reject(lambda: plan.finish(z, bad), z)
            reject(lambda: plan.middle(z, y), z)  # expected FP16 at attention boundary
            reject(lambda: plan.finish(z, z), z)
            reject(lambda: plan.begin(z.half()), z)
            scratch = plan.begin(z)
            reject(lambda: plan.middle(z, scratch), z)
            if b != q:
                reject(lambda: plan.begin(z.view(q, b, 5376)), z)
            records.append(dict(layer=block.layer_idx, attention=type(block.attn).__name__,
                                batch=b, query=q, device=str(dev), exact=True, rejected_safely=True))
        for n in norms:
            n.unload()
    # Exercise generic contracts independently of the Gemma emitter.
    device = torch.device('cuda', a.devices[0])
    x = torch.randn(1, 1, 256, device=device)
    specs = [(0, 1), (1, 0), (2, 1), (-1, 1)]
    code = [[(3, 0, 3, 0)], [(6, 1, 0, 0), (3, 1, 3, 0)], [(6, 2, 0, 0)]]
    generic = []
    for dtype in (torch.half, torch.bfloat16):
        w = torch.randn(256, dtype=dtype, device=device)
        plan = ext.BC_BlockPlanV2(specs, [w], [[1e-5, 0.25, 0.75]], code, [3, 3, -1], 1, 1, 256)
        out = torch.empty_like(x)
        ext.rms_norm(x.view(1, 256), w, out.view(1, 256), 1e-5, 0.25, 0.75, False, False, 1)
        assert torch.equal(out, plan.begin(x))
        y = torch.randn_like(x, dtype=torch.half)
        z = x.clone()
        ext.rms_norm(y.view(1, 256), w, z.view(1, 256), 1e-5, 0.25, 0.75, False, True, 1)
        ext.rms_norm(y.view(1, 256), w, out.view(1, 256), 1e-5, 0.25, 0.75, False, False, 1)
        assert torch.equal(out, plan.middle(x, y))
        assert torch.equal(x, z)
        generic.append(dict(weight_dtype=str(dtype), fp32_output=True, bias_scale=True))
    for weight in (w.float(), w[:128], w.cpu(), w.repeat(2)[::2]):
        reject(lambda: ext.BC_BlockPlanV2(specs, [weight], [[1e-5, 0, 1]], code,
                                        [3, 3, -1], 1, 1, 256), x)
    for config in ([0, 0, 1], [float('nan'), 0, 1], [float('inf'), 0, 1],
                   [1e-5, float('inf'), 1], [1e-5, 0, float('nan')]):
        reject(lambda: ext.BC_BlockPlanV2(specs, [w], [config], code,
                                        [3, 3, -1], 1, 1, 256), x)
    for invalid in ([[(3, 3, 3, 0)], code[1], code[2]],
                    [[(3, 1, 3, 0)], code[1], code[2]],
                    [code[0], [(6, 1, 0, 0), (3, 2, 3, 0)], code[2]],
                    [[(3, 0, 0, 0)], code[1], code[2]]):
        reject(lambda: ext.BC_BlockPlanV2(specs, [w], [[1e-5, 0, 1]], invalid,
                                        [3, 3, -1], 1, 1, 256), x)
    # A late invalid binding must be checked before an earlier instruction adds.
    second = w.clone()
    late = [code[0], [(6, 1, 0, 0), (3, 1, 3, 1)], code[2]]
    plan = ext.BC_BlockPlanV2(specs, [w, second], [[1e-5, 0, 1]]*2, late,
                            [3, 3, -1], 1, 1, 256)
    second.resize_(128)
    reject(lambda: plan.middle(x, torch.randn_like(x, dtype=torch.half)), x)
    memory = [dict(device=i, peak=torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]
    assert all(r['peak'] == 0 for r in memory if r['device'] not in a.devices)
    (a.output/'sites.json').write_text(json.dumps(dict(records=records, generic=generic, memory=memory), indent=2))
    print(f'PASS {len(records)} exact Gemma stages, generic norms and transactional rejections', flush=True)


if __name__ == '__main__':
    main()
