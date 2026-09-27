"""Qwen long-QSA three-way gate using established generator/state validators."""
import argparse
from contextlib import ExitStack
import json
import shutil
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--extension-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--suite', choices=['ordinary', 'mtp'], default='ordinary')
p.add_argument('--prompt-tokens', type=int, choices=[32768, 65536], default=65536)
p.add_argument('--reference-dir', type=Path, help='Directory containing the protected prompt-token-ids.json and metadata.json')
p.add_argument('--tokens', type=int, default=64)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(a.extension_dir.resolve()))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules import native_plan
from util import check_qwen_concurrency as h
from util.check_qwen_native import digest
from util.native_plan_validation import backend
from exllamav3.modules.attention_fn.bc_attn import BCAttn
from exllamav3.util.tensor import g_tensor_cache

cfg = Config.from_directory(r'D:\tabbyAPI\models\Qwen3.8-Flash-Next-Uncensored-exl3-4bpw')
model, draft = Model.from_config(cfg), Model.from_config(cfg, component='mtp')
tokenizer = Tokenizer.from_config(cfg)
reference = a.reference_dir or ROOT/f'native-results/long-qsa-39b6a5e-{a.prompt_tokens//1024}k'
metadata = json.loads((reference/'metadata.json').read_text())
(a.output/'input-reference').mkdir()
for name in ('metadata.json', 'prompt-token-ids.json'):
    shutil.copy2(reference/name, a.output/'input-reference'/name)
ids = torch.tensor(json.loads((reference/'prompt-token-ids.json').read_text()), dtype=torch.long)
assert ids.shape == (1, a.prompt_tokens)
capacity = metadata['cache_tokens']
assert a.prompt_tokens+a.tokens+5 <= capacity
cache = Cache(model, capacity, max_batch_size=1, max_history=4)
dc = Cache(draft, capacity, max_batch_size=1)
draft.load(device=torch.device('cuda:0'), max_chunk_size=2048)
model.load(device=torch.device('cuda:0'), max_chunk_size=2048)
draft.attach_to(model)
original_encode = tokenizer.encode
def encode(text, *args, **kwargs):
    return ids.clone() if text == h.PROMPTS[0] else original_encode(text, *args, **kwargs)
def kv(c, job):
    sequence = job.sequences[0]
    valid = sequence.kv_position
    if valid < ids.shape[1]-1:
        return {'stock_prefill_not_compared': True}
    pages = sequence.block_index_tensor[0, :len(sequence.allocated_pages)].long()
    result = {}
    for key, layer in c.layers.items():
        planes = {}
        for name in ('k', 'v', 'raw_k', 'pooled'):
            tensor = getattr(layer, name)
            length = valid//layer.compress_ratio if name == 'pooled' else valid
            planes[name] = digest(tensor.index_select(0, pages.to(tensor.device)).flatten(0, 1)[:length])
        result[str(key)] = dict(valid=valid, planes=planes)
    return result

def recurrent(state):
    if state is None: return None
    history = state.last_history
    layers = {}
    for key, layer in state.cache.get_all_recurrent_layers().items():
        conv, rec = layer.get_state_tensors()
        layers[str(key)] = [digest(conv[state.slot, :, :layer.module.conv_kernel_size+history]),
                            digest(rec[state.slot, :history+1])]
    return dict(position=state.position, last_history=history, layers=layers)

def run(name):
    sparse = []
    active = [False]
    roles = {id(m): role for role, mod in [('target', model), ('draft', draft)] for m in mod}
    def record_indices(indices, module, role, lengths, query):
        rows = []
        for i, row in enumerate(indices.cpu()):
            selected = row[row >= 0]
            visible = int(lengths[i//query]) + i%query + 1
            assert len(selected) == min(visible//module.compress_ratio, module.block_topk)*module.compress_ratio + visible%module.compress_ratio
            assert selected.numel() and int(selected.max()) < visible
            rows.append(dict(visible=visible, count=len(selected), sha256=digest(selected)))
        sparse.append(dict(role=role, key=module.key, rows=rows))
    with backend([model, draft], name == 'plan'), ExitStack() as scope:
        original_step = BCAttn.step
        def step(attn, x, *args, **kwargs):
            y = original_step(attn, x, *args, **kwargs)
            lengths = kwargs.get('host_seqlens')
            if y is not None and attn.qsa and not active[0] and int(lengths.max())+x.shape[1] > attn.qsa_threshold:
                rows = x.shape[0]*x.shape[1]
                width = attn._qsa_sparse_geometry(rows)[0]
                indices = g_tensor_cache.get_bucketed(attn.device, rows*width, torch.int32, 'bca_qsa_indices').view(rows, width)
                record_indices(indices, attn.qsa_idx, roles[id(attn.module)], lengths, x.shape[1])
            return y
        scope.enter_context(patch.object(BCAttn, 'step', step))
        scope.enter_context(patch.object(tokenizer, 'encode', encode))
        scope.enter_context(patch.object(h, 'kv_snapshot', kv))
        scope.enter_context(patch.object(h, 'recurrent_snapshot', recurrent))
        scope.enter_context(patch.object(h, 'digest', lambda t: 'stock-prefill-not-compared' if active[0] else digest(t)))
        for role, mod in [('target', model), ('draft', draft)]:
            for method in ('forward', 'prefill'):
                original = getattr(mod, method)
                def prefill(input_ids, params=None, _f=original, _r=role):
                    past = int(params['cache_seqlens'][0])
                    long_prefill = past < a.prompt_tokens-1
                    before = active[0]; active[0] = long_prefill
                    try: return _f(input_ids, params)
                    finally:
                        active[0] = before
                        if _r == 'target' and long_prefill:
                            print(name, 'prefill', past, flush=True)
                scope.enter_context(patch.object(mod, method, prefill))
            for module in mod:
                if not hasattr(module, 'select_indices_paged'): continue
                original_select = module.select_indices_paged
                def select(layer, q_idx, block_table, lengths, _f=original_select, _m=module, _r=role):
                    indices = _f(layer, q_idx, block_table, lengths)
                    if not active[0]:
                        record_indices(indices, _m, _r, lengths, q_idx.shape[1])
                    return indices
                scope.enter_context(patch.object(module, 'select_indices_paged', select))
        events, result = h.run_jobs(model, draft, cache, dc, tokenizer, native=name != 'stock',
            mtp=a.suite=='mtp', count=1, lengths=[a.tokens], correctness=True)
    for suffix, obj in [('events', events), ('sparse', sparse), ('summary', result)]:
        (a.output/f'{name}-{suffix}.json').write_text(json.dumps(obj))
    assert sparse and all(r['visible'] >= a.prompt_tokens for e in sparse for r in e['rows'])
    return events, sparse, result

with torch.inference_mode():
    stock, bespoke, plan = (run(name) for name in ('stock', 'bespoke', 'plan'))
    assert stock[:2] == bespoke[:2] == plan[:2], 'Long QSA event/state/selection mismatch'
    assert plan[2]['calls']['native'] > 0
    (a.output/'parity.json').write_text(json.dumps(dict(exact=True, three_way=True,
        prompt_tokens=a.prompt_tokens, output_tokens=a.tokens, sparse_events=len(plan[1]), calls=plan[2]['calls']), indent=2))
    print('PASS', a.output, flush=True)
