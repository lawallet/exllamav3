"""Exact Laguna v2 outputs, live SWA/paged state, scheduling and timed pairs."""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def snapshot(cache, states, tables):
    result = {}
    for i, state in enumerate(states):
        p = state.position
        result[f'{i}:position'] = (p, state.window_beg)
        for key, layer in cache.get_all_recurrent_layers().items():
            valid = p-state.window_beg
            assert 0 <= valid <= layer.module.kv_state_size
            for name in ('k_state', 'v_state'):
                result[f'{i}:{key}:{name}'] = digest(getattr(layer, name)[state.slot, :valid])
        for key, layer in cache.layers.items():
            pages = tables[i].to(layer.device)
            for name in ('k', 'v'):
                result[f'{i}:{key}:{name}'] = digest(getattr(layer, name)[pages].flatten(0, 1)[:p])
    return result


@contextmanager
def observe(models, enabled):
    from unittest.mock import patch
    from exllamav3.modules import native_plan, TransformerBlock
    counts, events = Counter(), []
    class Stages:
        def __init__(self, native): self.native = native
        def begin(self, x):
            counts['begin'] += 1
            counts['target:begin'] += 1
            return self.native.begin(x)
        def middle(self, x, y):
            counts['middle'] += 1
            return self.native.middle(x, y)
        def finish(self, x, y):
            counts['finish'] += 1
            self.native.finish(x, y)
    emit = native_plan.devstral
    def construct(*args):
        result = emit(*args)
        assert type(result).__name__ == 'BC_BlockPlanV2'
        counts['constructed'] += 1
        return result
    previous = native_plan.enabled
    native_plan.enabled = enabled
    try:
        with ExitStack() as scope:
            scope.enter_context(patch.object(native_plan, 'devstral', construct))
            for model in models:
                for m in model.modules:
                    move = m.prepare_for_device
                    def transfer(x, params, move=move, m=m):
                        src = str(x.device)
                        y = move(x, params)
                        if x.ndim == 3 and src != str(y.device):
                            events.append(('transfer', m.key, src, str(y.device), digest(y)))
                        return y
                    scope.enter_context(patch.object(m, 'prepare_for_device', transfer))
                    if type(m) is not TransformerBlock: continue
                    for child_name in ('attn', 'mlp'):
                        child = getattr(m, child_name)
                        child_forward = child.forward
                        def sublayer(x, params, child_forward=child_forward, child=child, **kw):
                            incoming = digest(x)
                            y = child_forward(x, params, **kw)
                            events.append(('sublayer', child.key, incoming, digest(y), str(y.dtype)))
                            return y
                        scope.enter_context(patch.object(child, 'forward', sublayer))
                    if hasattr(m.mlp, 'routing_fn'):
                        route, mlp = m.mlp.routing_fn, m.mlp
                        def routing(rows, cfg, y, params, route=route, mlp=mlp):
                            ids, weights = route(rows, cfg, y, params)
                            counts['python_route'] += 1
                            events.append(('route', mlp.key, digest(y), digest(ids), digest(weights)))
                            return ids, weights
                        scope.enter_context(patch.object(mlp, 'routing_fn', routing))
                        class Experts:
                            def __init__(self, bc, mlp): self.bc, self.mlp = bc, mlp
                            def __getattr__(self, name): return getattr(self.bc, name)
                            def run_bszN(self, y, ids, weights):
                                counts['python_moe_native_calls'] += 1
                                self.bc.run_bszN(y, ids, weights)
                                events.append(('experts', self.mlp.key,
                                    digest(self.mlp.experts_cfg.out_bszn[:y.shape[0]])))
                        if mlp.bc is not None:
                            scope.enter_context(patch.object(mlp, 'bc', Experts(mlp.bc, mlp)))
                    m._native_laguna = {}
                    get, forward = m._native_plan_get, m.forward
                    def select(block, x, params, get=get):
                        n = get(block, x, params)
                        counts['eligible' if n is not None else 'fallback'] += 1
                        if n is None:
                            counts[f'fallback:{tuple(x.shape[:2])}:prefill={bool(params.get("prefill"))}'] += 1
                            if x.shape[1] == 1 and not params.get('prefill'):
                                counts['target:decode_fallback'] += 1
                        return Stages(n) if n is not None else None
                    def fwd(x, params, forward=forward, m=m, **kw):
                        y = forward(x, params, **kw)
                        events.append((m.key, list(x.shape[:2]), digest(y)))
                        return y
                    scope.enter_context(patch.object(m, '_native_plan_get', select))
                    scope.enter_context(patch.object(m, 'forward', fwd))
            yield counts, events
    finally:
        native_plan.enabled = previous


def direct(model, cache, tokenizer, batch, query, steps, context, enabled, measured=False,
           expected_native_blocks=48, decode_norm_override=None):
    import torch
    from exllamav3.modules import native_plan
    prompt = tokenizer.encode(model.default_chat_prompt('Explain how computers execute programs.'), encode_special_tokens=True)
    ids = prompt.repeat(batch, (context+prompt.shape[1]-1)//prompt.shape[1])[:, :context]
    capacity = cache.max_num_tokens//batch//256*256
    states = [cache.get_new_state() for _ in range(batch)]
    records, tables = [], None
    def forward(tokens, exports=False):
        nonlocal tables
        params = dict(attn_mode='flash_attn', cache=cache, recurrent_states=states,
                      batch_shape=(batch, capacity), past_len=states[0].position)
        if exports: params['export_state_layers'] = {0, 5, 47}
        logits = model.forward(tokens, params)
        tables = params['block_table']
        return logits, params
    previous = native_plan.enabled
    old_config = None
    try:
        native_plan.enabled = False
        for start in range(0, context, 256):
            forward(ids[:, start:start+256])
        if decode_norm_override:
            obj, name, value = decode_norm_override
            old_config = getattr(obj, name)
            setattr(obj, name, value)
        current = ids[:, -1:].repeat(1, query)
        if measured:
            native_plan.enabled = enabled
            for _ in range(8): current = forward(current)[0][:, -1:].argmax(-1).cpu()
            for dev in model.active_devices: torch.cuda.synchronize(dev)
            start = time.perf_counter()
            for _ in range(steps): current = forward(current)[0][:, -1:].argmax(-1).cpu()
            for dev in model.active_devices: torch.cuda.synchronize(dev)
            elapsed = time.perf_counter()-start
            return dict(seconds=elapsed, tps=batch*steps/elapsed, tokens=current.tolist())
        with observe([model], enabled) as (counts, events):
            for step in range(steps):
                logits, params = forward(current, exports=True)
                records.append(dict(logits=digest(logits),
                    exports=[digest(t) for t in params.get('export_states', [])],
                    state=snapshot(cache, states, tables) if step in (0, steps-1) else None))
                current = logits[:, -1:].argmax(-1).cpu().repeat(1, query)
            if enabled:
                expected = expected_native_blocks if query <= 6 else 0
                assert counts['begin'] == counts['middle'] == counts['finish'] == expected*steps, counts
                assert counts['fallback'] == (48-expected)*steps and counts['constructed'] == expected, counts
            return dict(records=records, boundaries=events, counts=dict(counts))
    finally:
        native_plan.enabled = previous
        if decode_norm_override and old_config is not None:
            setattr(decode_norm_override[0], decode_norm_override[1], old_config)
        for state in states: state.free()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extension-dir', type=Path, default=ROOT/'native-results/devstral2/extension')
    p.add_argument('--phase', choices=['short', 'generator', 'long', 'benchmark'], default='short')
    p.add_argument('--context', type=int, default=300)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--budgets', type=float, nargs='+', default=[55, 22])
    p.add_argument('--reload', action='store_true')
    p.add_argument('--reference-dir', type=Path)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--case', type=int, action='append')
    p.add_argument('--cache-tokens', type=int)
    a = p.parse_args()
    sys.path.insert(1, str(a.extension_dir.resolve()))
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, obj): (a.output/name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.modules import native_plan
    from exllamav3.ext import exllamav3_ext as ext
    assert native_plan.v2_available
    torch.set_grad_enabled(False)
    cfg = Config.from_directory(r'D:\text-generation-webui\user_data\models\turboderp_Laguna-S-2.1-exl3_4.00bpw')
    model, tokenizer = Model.from_config(cfg), Tokenizer.from_config(cfg)
    capacity = max(8192, ((a.context+a.steps*6+512+255)//256)*256*4)
    if a.cache_tokens: capacity = a.cache_tokens
    cache = Cache(model, max_num_tokens=capacity, max_batch_size=4, max_history=5)
    model.load(use_per_device=a.budgets, max_chunk_size=256, max_batch_size=4, progressbar=True)
    save('environment.json', dict(python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        extension=ext.__file__, budgets=a.budgets, cache_tokens=capacity))
    save('placement.json', [dict(key=m.key, device=str(m.device)) for m in model.modules])
    try:
        # Warm immutable projection/attention companions identically before A/B.
        save('warm-stock.json', direct(model, cache, tokenizer, 1, 1, 2, a.context, False))
        if a.phase in ('short', 'long'):
            shapes = [(1, 1), (2, 1), (3, 1), (4, 1), (1, 5), (2, 6), (1, 7)] if a.phase=='short' else [(1, 1), (1, 5)]
            for b, q in shapes:
                results = []
                for enabled in (False, True):
                    r = direct(model, cache, tokenizer, b, q, a.steps, a.context, enabled)
                    save(f'b{b}-q{q}-{int(enabled)}.json', r)
                    results.append(r)
                for key in ('records', 'boundaries'):
                    assert results[0][key] == results[1][key], f'{key} mismatch B{b}/Q{q}'
                print(f'PASS exact B{b}/Q{q}', flush=True)
            from exllamav3.modules.native_laguna import get_native_block
            block = model.modules[model.first_block_idx]
            x = torch.zeros((1, 1, cfg.hidden_size), device=block.device)
            native_plan.enabled = True
            for params in ({'capture': None}, {'loras': []}, {'prefill': True}, {'backend': None}):
                assert get_native_block(block, x, params) is None
                assert torch.count_nonzero(x) == 0
            assert get_native_block(block, x.repeat(1, 7, 1), {}) is None
            norm = block.attn_norm
            old = norm.groups
            norm.groups = 2
            assert get_native_block(block, x, {}) is None
            norm.groups = old
            plan = get_native_block(block, x, {})
            weight = norm.weight
            norm.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
            assert get_native_block(block, x, {}) is not plan
            norm.weight = weight
            save('fallback.json', dict(modes=True, shape=True, config=True, identity_rebuild=True))
            # span_heads is a valid stock Q1 configuration, outside the v2
            # binding contract. Prove real stock dispatch, not just guard None.
            override = (norm, 'span_heads', True)
            stock = direct(model, cache, tokenizer, 1, 1, 2, a.context, False,
                           decode_norm_override=override)
            fallback = direct(model, cache, tokenizer, 1, 1, 2, a.context, True,
                              expected_native_blocks=47, decode_norm_override=override)
            save('config-stock.json', stock)
            save('config-fallback.json', fallback)
            assert stock['records'] == fallback['records']
            assert stock['boundaries'] == fallback['boundaries']
            if a.reload:
                before = direct(model, cache, tokenizer, 1, 1, 2, a.context, True)
                save('reload-before.json', before)
                save('reload-placement-before.json', [dict(key=m.key, device=str(m.device)) for m in model.modules])
                # The identity-rebinding probe must not retain old norm arenas
                # or plan scratch across unload: autosplit budgets are additive
                # to the calling process's existing reservation.
                del plan, weight, x
                devices = list(model.active_devices)
                model.unload()
                assert all(not getattr(m, '_native_laguna', {}) for m in model.modules)
                # Stock's process-wide decode workspace cache is intentionally
                # not model-owned. Reset it in this isolated reload experiment
                # so the additive loader budget measures the same workload.
                from exllamav3.util.tensor import g_tensor_cache
                from exllamav3.util.memory import free_mem
                free_mem()
                memory_before = {str(d): torch.cuda.memory_reserved(d) for d in devices}
                g_tensor_cache.drop_all()
                free_mem()
                save('reload-memory.json', dict(before_workspace_clear=memory_before,
                     after_workspace_clear={str(d): torch.cuda.memory_reserved(d) for d in devices}))
                model.load(use_per_device=a.budgets, max_chunk_size=256, max_batch_size=4)
                after = direct(model, cache, tokenizer, 1, 1, 2, a.context, True)
                save('reload-after.json', after)
                save('reload-placement-after.json', [dict(key=m.key, device=str(m.device)) for m in model.modules])
                assert json.loads((a.output/'reload-placement-before.json').read_text()) == json.loads((a.output/'reload-placement-after.json').read_text()), 'Reload placement changed'
                assert before == after
                save('reload.json', dict(exact=True))
        elif a.phase=='generator':
            # Reuse the established scheduler/job oracle, replacing only its
            # architecture-specific stage observer and valid-state serializer.
            from util import validate_deepseek_plan as jobs
            jobs.observe, jobs.snapshot = observe, snapshot
            cases = [dict(count=b) for b in (1, 2, 3, 4)] + [dict(count=3, stagger=True),
                dict(count=3, stagger=True, cancel=True), dict(count=2, stochastic=True)]
            for i, case in enumerate(cases):
                if a.case and i not in a.case: continue
                results = []
                for enabled in (False, True):
                    r = jobs.jobs_run(model, cache, tokenizer, None, None, enabled,
                                      steps=a.steps, context=a.context, **case)
                    save(f'case{i}-{int(enabled)}.json', r)
                    results.append(r)
                for key in ('jobs', 'states', 'logits', 'boundaries', 'reuse', 'cancelled'):
                    assert results[0][key] == results[1][key], f'case {i}: {key} mismatch'
                if a.cache_tokens and case.get('stagger'):
                    assert results[1]['reuse'].get('pages', 0) > 0, 'Page reuse not exercised'
                    assert results[1]['reuse'].get('slots', 0) > 0, 'Slot reuse not exercised'
                if case.get('cancel'):
                    assert results[1]['cancelled'], 'No live cancellation observed'
                print(f'PASS generator case {i}', flush=True)
        else:
            from statistics import median
            records, summary = [], []
            for b in (1, 4):
                for repeat in range(a.runs):
                    pair = []
                    for enabled in ((False, True) if repeat%2==0 else (True, False)):
                        r = direct(model, cache, tokenizer, b, 1, a.steps, a.context, enabled, measured=True)
                        r.update(batch=b, repeat=repeat, plan=enabled)
                        records.append(r); pair.append(r)
                        save('samples.json', records)
                    assert pair[0]['tokens'] == pair[1]['tokens']
                rates = {str(v): median(r['tps'] for r in records if r['batch']==b and r['plan']==v) for v in (False, True)}
                pairs = [next(r['tps'] for r in records if r['batch']==b and r['repeat']==i and r['plan']) /
                         next(r['tps'] for r in records if r['batch']==b and r['repeat']==i and not r['plan'])-1 for i in range(a.runs)]
                summary.append(dict(batch=b, median_tps=rates, paired_changes=pairs, median_change=median(pairs)))
                save('performance.json', summary)
                assert median(pairs)>=-0.03, 'Repeatable regression requires investigation'
        from util.native_plan_validation import memory_report
        memory_report(model, a.output)
        if a.reference_dir:
            for path in a.output.glob('b*-q*-*.json'):
                assert json.loads(path.read_text()) == json.loads((a.reference_dir/path.name).read_text()), path.name
            assert json.loads((a.output/'placement.json').read_text()) == json.loads((a.reference_dir/'placement.json').read_text())
            save('topology.json', dict(exact=True))
    finally:
        model.unload()


if __name__ == '__main__': main()
