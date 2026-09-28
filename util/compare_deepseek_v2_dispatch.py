"""Alternate the exact pre-v2/current shared Python dispatch on one loaded model.

V1 native C++ and Qwen/GLM dispatch are unchanged. DeepSeek alone shares the
Python branch extended for single-stream residual/export/scaling behavior.
"""
import argparse
import inspect
import json
from pathlib import Path
from statistics import median
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--extension-dir', type=Path, default=ROOT/'native-results/block-ir-v2/extension-r2')
p.add_argument('--steps', type=int, default=128)
p.add_argument('--runs', type=int, default=5)
a = p.parse_args()
sys.path.insert(1, str(a.extension_dir.resolve()))
a.output.mkdir(parents=True, exist_ok=False)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules import transformer, TransformerBlock
from util.validate_deepseek_plan import direct, direct_benchmark
direct = torch.inference_mode()(direct)
direct_benchmark = torch.inference_mode()(direct_benchmark)

source = textwrap.dedent(inspect.getsource(TransformerBlock.forward))
changes = [
    ("            y = native.middle(x, y)\n            if x.ndim == 3:\n                params['residual'] = x\n            y = self.mlp.forward(y, params)",
     "            y = self.mlp.forward(native.middle(x, y), params)"),
    ("                exported = x.mean(dim=2) if x.ndim == 4 else x\n                s.append(exported.half().clamp_(-65504.0, 65504.0))\n            if x.ndim == 3 and self.layer_scalar_f is not None:\n                x *= self.layer_scalar_f",
     "                s.append(x.mean(dim=2).half().clamp_(-65504.0, 65504.0))"),
]
for new, old in changes:
    assert source.count(new) == 1, 'Dispatch changed: review pre-v2 reconstruction'
    source = source.replace(new, old)
(a.output/'pre_v2_forward.py').write_text(source)
scope = dict(vars(transformer))
exec(compile(source, str(a.output/'pre_v2_forward.py'), 'exec'), scope)
old_forward = scope['forward']

torch.set_grad_enabled(False)
cfg = Config.from_directory(r'D:\tabbyAPI\models\turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw')
model, tokenizer = Model.from_config(cfg), Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=8192, max_batch_size=4, max_history=5)
model.load(use_per_device=[85, 40], max_chunk_size=256, max_batch_size=4, progressbar=True)
blocks = [m for m in model.modules if type(m) is TransformerBlock]
original = [m.forward for m in blocks]
records = []
expected = None
try:
    cold = direct(model, cache, tokenizer, 1, 1, 2, 300, False)
    (a.output/'cold-stock.json').write_text(json.dumps(cold, indent=2))
    parity = []
    for enabled in (False, True):
        r = direct(model, cache, tokenizer, 1, 1, 4, 300, enabled)
        (a.output/f'final-parity-{int(enabled)}.json').write_text(json.dumps(r, indent=2))
        parity.append(r)
    assert parity[0]['records'] == parity[1]['records']
    assert parity[0]['boundaries'] == parity[1]['boundaries']
    for repeat in range(a.runs):
        for current in ((False, True) if repeat % 2 == 0 else (True, False)):
            for block, method in zip(blocks, original):
                block.forward = method if current else old_forward.__get__(block, type(block))
            r = direct_benchmark(model, cache, tokenizer, True, 1, a.steps, 300)
            exact = {k: r[k] for k in ('logits', 'tokens', 'state')}
            if expected is None: expected = exact
            assert exact == expected
            r.update(repeat=repeat, current=current)
            records.append(r)
            (a.output/'samples.json').write_text(json.dumps(records, indent=2))
        print('PASS dispatch pair', repeat, flush=True)
    rates = {str(v): median(r['tokens_per_second'] for r in records if r['current']==v) for v in (False, True)}
    changes = [next(r['tokens_per_second'] for r in records if r['repeat']==i and r['current']) /
               next(r['tokens_per_second'] for r in records if r['repeat']==i and not r['current'])-1 for i in range(a.runs)]
    (a.output/'performance.json').write_text(json.dumps(dict(median_tps=rates,
        paired_changes=changes, median_change=median(changes)), indent=2))
    assert median(changes) >= -0.03, 'Shared dispatch regression requires investigation'
finally:
    for block, method in zip(blocks, original): block.forward = method
    model.unload()
