"""Isolated real-generation IR investigation. Never installs or changes packages.

Timing and instrumentation are separate. Each sample uses a fresh generator,
65536-token target/draft caches, greedy sampling and a fixed prompt. Use the
existing validate_native_tp.py for exact logits/state and local-stage parity.
"""
import argparse
from collections import Counter
import cProfile
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_mode(context, enabled, graphs=False):
    from exllamav3.modules import native_plan, native_tp
    native_plan.enabled = enabled
    native_tp.hc_graphs_enabled = graphs
    context['final_ir_graphs'] = graphs
    for block in context['modules']:
        for signature, plan in getattr(block, '_native_tp_plans', {}).values():
            if hasattr(plan, 'set_graphs'):
                plan.set_graphs(graphs and getattr(block, '_native_tp_kind', None) == 'glm')


def rank_report(context):
    import torch
    from exllamav3.modules.native_tp import diagnostics
    result = diagnostics(context)
    plans = [plan for block in context['modules']
             for signature,plan in getattr(block, '_native_tp_plans', {}).values()]
    result['graph_captures'] = sum(getattr(p, 'graph_captures', 0) for p in plans)
    result['graph_replays'] = sum(getattr(p, 'graph_replays', 0) for p in plans)
    device = context['device']
    result['memory'] = dict(allocated=torch.cuda.memory_allocated(device),
                           reserved=torch.cuda.memory_reserved(device),
                           free=torch.cuda.mem_get_info(device)[0])
    return result


def rank_profile(context, path):
    # The output-device pseudo-worker executes on the parent thread and is
    # already covered by the parent profile. cProfile instances cannot nest.
    if context['device'] == context['output_device']:
        return
    if path:
        context['final_ir_profile'] = cProfile.Profile()
        context['final_ir_profile_path'] = path
        context['final_ir_profile'].enable()
    else:
        profiler = context.pop('final_ir_profile')
        profiler.disable()
        profiler.dump_stats(context.pop('final_ir_profile_path') + f'-rank{context["rank"]}.prof')


def rank_trace(context, path):
    if context['device'] == context['output_device']:
        return
    import torch
    if path:
        trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA])
        context['final_ir_trace'] = trace
        context['final_ir_trace_path'] = path
        trace.__enter__()
    else:
        trace = context.pop('final_ir_trace')
        trace.__exit__(None, None, None)
        trace.export_chrome_trace(context.pop('final_ir_trace_path') + f'-rank{context["rank"]}.json')


class TimedPlan:
    def __init__(self, plan, measure, key):
        self.plan, self.measure, self.key = plan, measure, key

    def begin(self, x):
        return self.measure(self.key + ':begin', self.plan.begin, x)

    def middle(self, x, y):
        return self.measure(self.key + ':middle', self.plan.middle, x, y)

    def finish(self, x, y):
        return self.measure(self.key + ':finish', self.plan.finish, x, y)


def rank_timers(context, start):
    """Inclusive host timings, not synchronized GPU execution times."""
    if not start:
        saved, totals = context.pop('final_ir_timers')
        for obj, name, original in reversed(saved):
            setattr(obj, name, original)
        return dict(rank=context['rank'], device=context['device'], totals=totals)
    from exllamav3.modules import TransformerBlock
    totals, saved, plans = {}, [], {}

    def measure(key, fn, *args, **kw):
        t = time.perf_counter_ns()
        try:
            return fn(*args, **kw)
        finally:
            entry = totals.setdefault(key, [0, 0])
            entry[0] += 1
            entry[1] += time.perf_counter_ns()-t

    def wrap(obj, name, key):
        fn = getattr(obj, name)
        saved.append((obj, name, fn))
        def call(*args, **kw):
            return measure(key, fn, *args, **kw)
        setattr(obj, name, call)

    for block in context['modules']:
        if not isinstance(block, TransformerBlock):
            continue
        key = block.key
        for name in ('attn', 'mlp'):
            child = getattr(block, name)
            if child is not None:
                wrap(child, 'forward', key + ':' + name)
        get = block._native_plan_get
        if get is not None:
            saved.append((block, '_native_plan_get', get))
            def get_plan(b, x, params, get=get, key=key):
                plan = measure(key + ':select', get, b, x, params)
                if plan is None:
                    return None
                if id(plan) not in plans:
                    plans[id(plan)] = TimedPlan(plan, measure, key)
                return plans[id(plan)]
            block._native_plan_get = get_plan
    for name in ('all_reduce', 'broadcast', 'gather', 'gather_small'):
        wrap(context['backend'], name, 'collective_launch:' + name)
    context['final_ir_timers'] = (saved, totals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 1024, 2048, 4096, 8192])
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--modes', choices=['stock', 'native', 'graphs'], nargs='+', default=['stock', 'native'])
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--single', action='store_true')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--profile-tokens', type=int, default=0)
    p.add_argument('--host-timers', action='store_true', help='use explicit per-layer host timers instead of cProfile')
    p.add_argument('--trace-tokens', type=int, default=0)
    p.add_argument('--profile-prefix', type=int, default=128,
                   help='generate this many tokens before enabling observers (excludes prefill)')
    p.add_argument('--warm-longest', action='store_true', help='warm the entire longest request before timing')
    p.add_argument('--affinity', type=int, nargs='+', help='optional CPU affinity for this test process and its workers')
    p.add_argument('--window', type=int, default=512)
    a = p.parse_args()
    if a.single and a.modes != ['native']:
        p.error('Single-device baseline requires --modes native; disabling v2 can select the older native executor, not stock')
    if a.runs < 0 or a.window < 1 or a.batch < 1 or a.profile_prefix < 0:
        p.error('invalid run/window/batch count')
    longest = max(*a.lengths, 128, a.profile_prefix+max(a.profile_tokens, a.trace_tokens))
    if min(a.lengths) < 1 or longest + 1024 > 65536 // a.batch:
        p.error('invalid lengths for shared 65536-token cache')
    sys.path.insert(1, str(a.extension_dir.resolve()))
    os.environ.update(EXL3_NATIVE_PLAN='1', EXL3_NATIVE_QWEN_BLOCK='1', EXL3_NATIVE_GLM_BLOCK='1')
    import torch
    import exllamav3
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules import native_plan
    torch.set_grad_enabled(False)
    assert Path(exllamav3.__file__).parent == ROOT / 'exllamav3'
    assert Path(ext.__file__).parent.resolve() == a.extension_dir.resolve()
    if 'graphs' in a.modes:
        assert not a.single and hasattr(ext.BC_BlockPlan, 'set_graphs'), 'Graphs require the experimental TP extension'
    assert torch.cuda.device_count() == 1 if a.single else torch.cuda.device_count() >= 2
    a.output.mkdir(parents=True, exist_ok=False)

    single_counts = dict(executions=0, builds=0)
    if a.single:
        # The TP diagnostics do not cover single-device registrations. Count
        # completed native stages so a fallback-only run cannot masquerade as IR.
        class CountStages:
            def __init__(self, plan):
                self.plan = plan
            def __getattr__(self, name):
                return getattr(self.plan, name)
            def finish(self, x, y):
                self.plan.finish(x, y)
                single_counts['executions'] += 1
        for name in ('devstral', 'gemma', 'deepseek', 'qwen', 'glm'):
            original_emit = getattr(native_plan, name)
            def emit(*args, original=original_emit, **kw):
                plan = original(*args, **kw)
                single_counts['builds'] += 1
                return CountStages(plan)
            setattr(native_plan, name, emit)

    def save(name, value):
        (a.output / name).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

    def hashes():
        return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (ROOT / 'exllamav3').rglob('*.py')}

    initial = hashes()
    save('source-hashes.json', initial)
    (a.output / 'driver.py').write_bytes(Path(__file__).read_bytes())
    cfg = Config.from_directory(a.model)
    if 'graphs' in a.modes and not cfg.arch_string.startswith('Glm5'):
        p.error('Stage graph comparisons are enabled only for the validated GLM TP path')
    model, tokenizer = Model.from_config(cfg), Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=65536, max_batch_size=a.batch, max_history=5)
    draft = draft_cache = None
    save('inputs.json', dict(args=vars(a), revision=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), python=sys.version,
        torch=torch.__version__, cuda=torch.version.cuda,
        triton=importlib.metadata.version('triton-windows'), package=exllamav3.__file__,
        extension=ext.__file__, extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
        visible=os.environ.get('CUDA_VISIBLE_DEVICES'), cache_tokens=65536,
        architecture=cfg.arch_string, model_config=json.loads((Path(a.model)/'config.json').read_text()),
        weights=[dict(name=f.name, bytes=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
                 for f in sorted(Path(a.model).glob('*.safetensors'))],
        devices=[dict(logical=i, name=torch.cuda.get_device_name(i),
                      uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]))

    def configure(mode):
        enabled = mode != 'stock'
        native_plan.enabled = enabled
        if not a.single:
            model.tp_dispatch_all(set_mode, (enabled, mode == 'graphs'))

    def reports():
        if a.single:
            torch.cuda.synchronize()
            return []
        return model.tp_worker_dispatch_wait_multi(model.active_devices, rank_report, ())

    text = model.default_chat_prompt('Explain how computers execute programs. Give concrete examples and describe each step in detail.')
    input_ids = tokenizer.encode(text, encode_special_tokens=True)
    save('prompt.json', dict(text=text, ids=input_ids.tolist()))

    def sample(mode, length, instrument=False, cuda_trace=False):
        configure(mode)
        setup_start = time.perf_counter()
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                        draft_model=draft, draft_cache=draft_cache, num_draft_tokens=0,
                        dynamic_draft_tokens=False, max_batch_size=a.batch, max_chunk_size=256,
                        ngram_match_min=0, record_draft_stats=True)
        generator_seconds = time.perf_counter()-setup_start
        before = reports()
        shapes, draft_shapes = Counter(), Counter()
        original = model.forward
        original_draft = draft.forward if draft else None
        profiler = None
        trace = None
        observed_start = None
        rank_host, parent_host = [], {}
        timers_active = False

        def start_observers():
            nonlocal profiler, trace, timers_active
            def timed_forward(key, fn, input_ids, params, *args, **kw):
                if not a.host_timers or cuda_trace:
                    return fn(input_ids, params, *args, **kw)
                t = time.perf_counter_ns()
                try:
                    return fn(input_ids, params, *args, **kw)
                finally:
                    entry = parent_host.setdefault(key, [0, 0])
                    entry[0] += 1
                    entry[1] += time.perf_counter_ns()-t
            def observe(input_ids, params=None, *args, **kw):
                shapes[(tuple(input_ids.shape), bool((params or {}).get('prefill')))] += 1
                if cuda_trace:
                    with torch.profiler.record_function(f'final_ir.target.{tuple(input_ids.shape)}'):
                        return original(input_ids, params, *args, **kw)
                return timed_forward('target', original, input_ids, params, *args, **kw)
            model.forward = observe
            if draft:
                def observe_draft(input_ids, params=None, *args, **kw):
                    draft_shapes[(tuple(input_ids.shape), bool((params or {}).get('prefill')))] += 1
                    if cuda_trace:
                        with torch.profiler.record_function(f'final_ir.draft.{tuple(input_ids.shape)}'):
                            return original_draft(input_ids, params, *args, **kw)
                    return timed_forward('draft', original_draft, input_ids, params, *args, **kw)
                draft.forward = observe_draft
            if cuda_trace:
                if not a.single:
                    model.tp_dispatch_all(rank_trace, (str((a.output / f'trace-{mode}').resolve()),))
                trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
                trace.__enter__()
            elif a.host_timers:
                if not a.single:
                    model.tp_dispatch_all(rank_timers, (True,))
                timers_active = True
            else:
                if not a.single:
                    model.tp_dispatch_all(rank_profile, (str((a.output / f'profile-{mode}').resolve()),))
                profiler = cProfile.Profile()
                profiler.enable()
        if instrument and a.profile_prefix == 0:
            start_observers()
            observed_start = 0
        completions = [''] * a.batch
        single_before = single_counts.copy()
        jobs = [Job(input_ids=input_ids.clone(), max_new_tokens=length, min_new_tokens=length,
                    sampler=GreedySampler(), seed=1234+i, stop_conditions=[]) for i in range(a.batch)]
        windows, last = [], {}
        serials = {}
        started_unix = time.time()
        start = time.perf_counter()
        for i, job in enumerate(jobs):
            serials[gen.enqueue(job)] = i
        previous = (0, start, 0, 0)
        next_window = a.window * a.batch
        try:
            while gen.num_remaining_jobs():
                for event in gen.iterate():
                    if event['stage'] == 'error':
                        raise event['error']
                    if event['stage'] == 'streaming':
                        completions[serials[event['serial']]] += event.get('text', '')
                    if event['eos']:
                        last[event['serial']] = event
                count = sum(j.new_tokens for j in jobs)
                if instrument and observed_start is None and count >= a.profile_prefix * a.batch:
                    start_observers()
                    observed_start = count
                if count >= next_window or not gen.num_remaining_jobs():
                    now = time.perf_counter()
                    accepted = sum(j.accepted_draft_tokens for j in jobs)
                    rejected = sum(j.rejected_draft_tokens for j in jobs)
                    windows.append(dict(tokens=count, wall_seconds=now-start,
                        cumulative_wall_tps=count/(now-start), window_tokens=count-previous[0],
                        window_tps=(count-previous[0])/(now-previous[1]),
                        accepted=accepted, rejected=rejected,
                        window_accepted=accepted-previous[2], window_rejected=rejected-previous[3]))
                    previous = (count, now, accepted, rejected)
                    next_window += a.window * a.batch
            elapsed = time.perf_counter()-start
        finally:
            if timers_active and not a.single:
                rank_host = model.tp_worker_dispatch_wait_multi(model.active_devices, rank_timers, (False,))
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(a.output / f'profile-{mode}-parent.prof'))
                if not a.single:
                    model.tp_dispatch_all(rank_profile, (None,))
            if trace:
                trace.__exit__(None, None, None)
                trace.export_chrome_trace(str(a.output / f'trace-{mode}-parent.json'))
                if not a.single:
                    model.tp_dispatch_all(rank_trace, (None,))
            model.forward = original
            if draft:
                draft.forward = original_draft
            gen.clear_queue()
            gen.filter_pool.shutdown(wait=True)
        after = reports()
        single_deltas = {k:single_counts[k]-single_before[k] for k in single_counts}
        if a.single:
            assert single_deltas['executions'] > 0, 'Single-device sample only fell back'
        assert len(last) == a.batch
        stats = [{k:e.get(k) for k in ('new_tokens', 'prompt_tokens', 'cached_tokens',
                 'time_prefill', 'time_generate', 'accepted_draft_tokens', 'rejected_draft_tokens')}
                 for e in last.values()]
        assert all(s['new_tokens'] == length for s in stats)
        deltas = [dict(rank=r['rank'], executions=sum(b['executions']-old['executions']
                        for b,old in zip(r['blocks'], prior['blocks'])),
                       builds=sum(b['builds']-old['builds'] for b,old in zip(r['blocks'], prior['blocks'])),
                       graph_captures=r['graph_captures']-prior['graph_captures'],
                       graph_replays=r['graph_replays']-prior['graph_replays'],
                       fallbacks=dict(sum((Counter(b['fallbacks'])-Counter(old['fallbacks'])
                           for b,old in zip(r['blocks'], prior['blocks'])), Counter())))
                  for r,prior in zip(after,before)]
        if not a.single:
            assert deltas and all(r['blocks'] for r in after), 'No native TP registrations'
            if mode == 'stock':
                assert all(r['executions'] == 0 for r in deltas), 'Stock sample executed IR'
            elif a.batch == 1:
                assert all(r['executions'] > 0 for r in deltas), 'Native sample only fell back'
            if mode == 'graphs':
                assert all(r['graph_replays'] > 0 for r in deltas), 'Graph sample did not replay any IR graph'
        return dict(mode=mode, length=length, instrumented=instrument, seconds=elapsed,
                    started_unix=started_unix, observed_start_tokens=observed_start,
                    rank_host_timings=rank_host, parent_host_timings=parent_host,
                    generator_seconds=generator_seconds,
                    before_counts=[dict(rank=r['rank'], executions=sum(b['executions'] for b in r['blocks']),
                                        builds=sum(b['builds'] for b in r['blocks'])) for r in before],
                    wall_tps=length*a.batch/elapsed,
                    decode_tps=length*a.batch/max(s['time_generate'] for s in stats),
                    completion_hashes=[hashlib.sha256(c.encode()).hexdigest() for c in completions],
                    stats=stats, windows=windows, draft_tokens=gen.num_draft_tokens,
                    target_shapes={str(k):v for k,v in shapes.items()},
                    draft_shapes={str(k):v for k,v in draft_shapes.items()}, deltas=deltas,
                    single_deltas=single_deltas if a.single else None,
                    rank_memory=[dict(rank=r['rank'], before=p.get('memory'), after=r.get('memory'))
                                 for r,p in zip(after,before)])

    results, references = [], {}
    try:
        if a.mtp:
            draft = Model.from_config(cfg, component='mtp')
            draft_cache = Cache(draft, max_num_tokens=65536, max_batch_size=a.batch)
            draft.load(device='cuda:0', max_chunk_size=256, max_batch_size=a.batch)
        if a.single:
            model.load(device='cuda:0', max_chunk_size=256, max_batch_size=a.batch, progressbar=True)
        else:
            model.load(tensor_p=True, tp_backend='native', reserve_per_device=2.,
                       max_chunk_size=256, max_batch_size=a.batch, progressbar=True)
            save('allocation.json', model.plan)
        if draft:
            draft.attach_to(model)
        if a.affinity:
            import psutil
            parent = psutil.Process()
            records = []
            for process in [parent, *parent.children(recursive=True)]:
                previous = process.cpu_affinity()
                process.cpu_affinity(a.affinity)
                records.append(dict(pid=process.pid, previous=previous, current=process.cpu_affinity()))
            save('affinity.json', records)
        for mode in a.modes:
            print('WARM', mode, flush=True)
            save(f'warm-{mode}.json', sample(mode, 128))
        if a.warm_longest:
            for mode in a.modes:
                print('WARM LONGEST', mode, max(a.lengths), flush=True)
                save(f'warm-longest-{mode}.json', sample(mode, max(a.lengths)))
        for mode in a.modes if a.profile_tokens else []:
            print('PROFILE', mode, flush=True)
            save(f'profile-{mode}.json', sample(mode, a.profile_prefix+a.profile_tokens, instrument=True))
        for mode in a.modes if a.trace_tokens else []:
            print('TRACE', mode, flush=True)
            save(f'trace-{mode}.json', sample(mode, a.profile_prefix+a.trace_tokens, instrument=True, cuda_trace=True))
        if a.profile_tokens or a.trace_tokens:
            for mode in a.modes:
                save(f'post-profile-warm-{mode}.json', sample(mode, 128))
        for length in a.lengths:
            for pair in range(a.runs):
                for mode in (a.modes if pair % 2 == 0 else a.modes[::-1]):
                    print('START', length, pair, mode, flush=True)
                    row = dict(pair=pair, **sample(mode, length))
                    results.append(row)
                    save('results.json', results)
                    if torch.cuda.device_count() <= 2 and a.batch == 1:
                        parity = (row['completion_hashes'], [(s['new_tokens'], s['accepted_draft_tokens'],
                                  s['rejected_draft_tokens']) for s in row['stats']])
                        previous = references.setdefault(length, parity)
                        assert parity == previous, 'Two-rank/single output or acceptance mismatch; timings retained'
                    print('RESULT', length, pair, mode, round(row['decode_tps'], 3), flush=True)
        save('diagnostics.json', reports())
        assert initial == hashes(), 'Product code changed during measurements'
        save('completed.json', dict(samples=len(results), source_unchanged=True))
    finally:
        model.unload()
        if draft:
            draft.unload()


if __name__ == '__main__':
    main()
