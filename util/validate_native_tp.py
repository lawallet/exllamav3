"""Paired real-model TP correctness and timing, without installing a package.

Use CUDA_VISIBLE_DEVICES to select at least two GPUs, or --single for TP-off.
All runs allocate a 65536-token cache. Worker traces are outside timed regions.
"""
import argparse
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


class ShadowPlan:
    """Compare local IR stages exactly using the SAME upstream reduced outputs.

    Three-rank CPU reductions round in arrival order. This oracle isolates IR
    arithmetic without replaying attention, changing the backend or duplicating
    any collective/cache update. It is disabled in timing regions.
    """
    def __init__(self, plan, block, params, context):
        self.plan, self.block, self.context = plan, block, context
        self.params = params.copy()
        self.params.pop('gr_weighted', None)

    def check(self, a, b, stage):
        import torch
        assert torch.equal(a, b), (self.block.key, stage, float((a-b).abs().max()))
        self.context['shadow_checks'] += 1

    def norm(self, site):
        import torch
        h, n = getattr(self.block, site+'_hc'), getattr(self.block, site+'_norm')
        if h is not None:
            self.post, self.comb, y = h.mix(self.x, self.params)
            y = y.half()
        else:
            y = self.x
        return n.forward(y, self.params, out_dtype=torch.half) if n else y.half()

    def apply(self, site, y):
        h, n = getattr(self.block, site+'_hc'), getattr(self.block, site+'_post_norm')
        if h is not None:
            self.x = h.apply_(self.x, y, self.post, self.comb, self.params)
        elif n is not None:
            n.forward(y, self.params, residual=self.x)
        else:
            self.x += y

    def begin(self, x):
        self.x = x.clone()
        expected = self.norm('attn')
        actual = self.plan.begin(x)
        self.check(expected, actual, 'begin')
        return actual

    def middle(self, x, y):
        import torch
        b = self.block
        if (b.attn_hc is None and b.attn_post_norm is None and
                b.mlp_norm.can_fuse_residual(self.x, y)):
            expected = b.mlp_norm.forward(y, self.params, out_dtype=torch.half, residual_in=self.x)
        else:
            self.apply('attn', y)
            expected = self.norm('mlp')
        actual = self.plan.middle(x, y)
        self.check(self.x, x, 'attention residual')
        self.check(expected, actual, 'middle')
        return actual

    def finish(self, x, y):
        self.apply('mlp', y)
        self.plan.finish(x, y)
        self.check(self.x, x, 'finish')


def mode(context, enabled, trace=False, reset=True):
    from exllamav3.modules import native_plan, native_tp, TransformerBlock
    native_plan.enabled = enabled
    context['ir_trace'] = []
    context['ir_trace_enabled'] = trace
    context['shadow_checks'] = 0
    for m in context['modules']:
        if not isinstance(m, TransformerBlock):
            continue
        if reset and m._native_tp_kind is not None:
            native_tp.clear(m)
        if not hasattr(m, '_validation_forward'):
            if m._native_tp_kind is not None:
                original_get = m._native_plan_get
                def get(block, x, params, original=original_get):
                    plan = original(block, x, params)
                    if plan is not None and context['world_size'] > 2 and context['ir_trace_enabled']:
                        return ShadowPlan(plan, block, params, context)
                    return plan
                m._native_plan_get = get
            m._validation_forward = m.forward
            def forward(x, params, m=m, **kw):
                result = m._validation_forward(x, params, **kw)
                if context['ir_trace_enabled']:
                    context['ir_trace'].append((m.key, list(result.shape), digest(result)))
                return result
            m.forward = forward


def report(context):
    from exllamav3.modules.native_tp import diagnostics
    result = diagnostics(context)
    result['trace'] = context['ir_trace']
    result['shadow_checks'] = context.get('shadow_checks', 0)
    return result


def profile(context, path):
    import cProfile
    if path is not None:
        context['ir_profile'] = cProfile.Profile()
        context['ir_profile'].enable()
        context['ir_profile_path'] = path
    else:
        profiler = context.pop('ir_profile')
        profiler.disable()
        profiler.dump_stats(context.pop('ir_profile_path') + f'-rank{context["rank"]}.prof')


def snapshot(context, cache_id, batch, position, slots, step_ends):
    """Hash initialized rank-local KV rows and live recurrent state, outside timing."""
    tensors = {}
    pages_per_sequence = 65536 // batch // 256
    for m in context['kv_modules']:
        layer = m.tp_cache_lookup.get(cache_id)
        if layer is None:
            continue
        for i, tensor in enumerate(layer.get_tensors()):
            if tensor is None:
                continue
            # QSA/MLA can mix full-rate and compressed planes in one layer.
            epp = tensor.shape[1]
            assert 256 % epp == 0
            entries = position // (256 // epp)
            for b in range(batch):
                start = b * pages_per_sequence
                initialized = tensor[start:start+(entries+epp-1)//epp].flatten(0,1)[:entries]
                tensors[f'{m.key}:kv:{i}:{b}'] = digest(initialized)
    for m in context['recurrent_modules']:
        layer = m.tp_recurrent_lookup.get(cache_id)
        if layer is None:
            continue
        if hasattr(layer, 'ring'):
            # DSV4 TP does not clear unused capacity on every slot acquisition.
            # Match the existing DeepSeek oracle's logical live-state slices.
            for b, (slot, window_beg) in enumerate(slots):
                tensors[f'{m.key}:ring:{b}'] = digest(layer.ring[slot, :position-window_beg])
                for name in ('comp_buf_kv','comp_buf_gate','idx_buf_kv','idx_buf_gate'):
                    t = getattr(layer, name, None)
                    if t is not None:
                        tensors[f'{m.key}:{name}:{b}'] = digest(t[slot, :min(position,layer.buf_rows)])
                for name in ('comp_ovl','idx_ovl'):
                    t = getattr(layer, name, None)
                    if t is not None:
                        # dsv4_compress saves only the LAST completed window in
                        # each call, not every prefill window (see its w == 0
                        # snapshot store). Other ring entries are stale capacity.
                        written, previous = set(), 0
                        for end in step_ends:
                            if end//m.compress_rate > previous//m.compress_rate:
                                written.add((end//m.compress_rate-1) % layer.ovl_depth)
                            previous = end
                        tensors[f'{m.key}:{name}:{b}'] = digest(t[slot, sorted(written)])
            continue
        values = layer.get_state_tensors() if hasattr(layer, 'get_state_tensors') else layer._tensors()
        for i, tensor in enumerate(values):
            for b, (slot, window_beg) in enumerate(slots):
                live = tensor[slot]
                if hasattr(layer, 'k_state'):
                    live = live[:position-window_beg]
                tensors[f'{m.key}:recurrent:{i}:{b}'] = digest(live)
    return tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--single', action='store_true')
    p.add_argument('--allow-host-ngram-baseline', action='store_true',
                   help='measure Qwen single-GPU using its unchanged, existing host ngram table')
    p.add_argument('--context', type=int, default=256)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--bench-steps', type=int, default=128)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 2])
    p.add_argument('--queries', type=int, nargs='+', default=[1])
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--reload', action='store_true')
    p.add_argument('--profile', action='store_true', help='profile timed regions; reported timings are instrumented')
    p.add_argument('--generator', action='store_true', help='also test concurrent generation and prompt-cache reuse')
    p.add_argument('--mtp', action='store_true', help='attach the existing single-device MTP draft for generator tests')
    p.add_argument('--generator-bench', action='store_true', help='benchmark actual MTP generation without traces/hashes')
    p.add_argument('--generator-bench-runs', type=int, default=3)
    a = p.parse_args()
    if a.generator_bench and not a.mtp:
        p.error('--generator-bench requires --mtp')
    sys.path.insert(1, str(a.extension_dir.resolve()))
    # Spawned workers inherit this path and these architecture switches.
    os.environ['EXL3_NATIVE_PLAN'] = '1'
    os.environ['EXL3_NATIVE_QWEN_BLOCK'] = '1'
    os.environ['EXL3_NATIVE_GLM_BLOCK'] = '1'
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules import native_plan
    torch.set_grad_enabled(False)
    if not a.single and torch.cuda.device_count() < 2:
        p.error('TP validation requires at least two visible GPUs; one-rank TP is prohibited')
    if a.single and torch.cuda.device_count() != 1:
        p.error('--single requires one visible GPU with TP disabled')
    if a.context + max(a.steps * max(a.queries), a.bench_steps) + 16 > 65536 // max(a.batches):
        p.error('workload exceeds the 65536-token cache capacity')
    assert native_plan.available and native_plan.v2_residual_available, ext.__file__
    assert Path(ext.__file__).parent.resolve() == a.extension_dir.resolve(), ext.__file__
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'validator.py').write_bytes(Path(__file__).read_bytes())
    def save(name, value):
        (a.output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    def source_hashes():
        return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'exllamav3').rglob('*.py')}
    initial_sources = source_hashes()
    save('source-hashes.json', initial_sources)
    lifecycle = []
    def memory_event(event):
        entry = dict(event=event, parent_allocated=[torch.cuda.memory_allocated(i)
                     for i in range(torch.cuda.device_count())])
        if 'unload' in event:
            # Validation wrappers contain cycles (bound methods/context). Record
            # both immediate and collected memory; do not hide it with empty_cache.
            import gc
            gc.collect()
            entry['after_gc_allocated'] = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
        lifecycle.append(entry)
        save('lifecycle-memory.json', lifecycle)
    memory_event('before model construction')
    single_counts = dict(executions=0, builds=0)
    if a.single:
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
    cfg = Config.from_directory(a.model)
    if a.single and not a.allow_host_ngram_baseline and (Path(a.model) / 'ngram_embedding.safetensors').exists():
        reason = ('This checkpoint uses an existing host-resident ngram table and its complete weights '
                  'do not fit entirely on the Pro 6000. Skipping the strict VRAM-resident single-GPU '
                  'baseline; no CPU offload is enabled to force it.')
        save('skipped.json', dict(reason=reason, model=a.model, max_context=65536))
        print('SKIP', reason, flush=True)
        return
    model, tokenizer = Model.from_config(cfg), Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=65536, max_batch_size=max(a.batches), max_history=5)
    draft = draft_cache = None
    import importlib.metadata
    with open(ext.__file__, 'rb') as binary:
        extension_hash = hashlib.file_digest(binary, 'sha256').hexdigest()
    save('inputs.json', dict(model=a.model, architecture=cfg.arch_string, max_context=65536,
                            cache_tokens=cache.max_num_tokens, args=vars(a) | {'output':str(a.output), 'extension_dir':str(a.extension_dir)},
                            visible=os.environ.get('CUDA_VISIBLE_DEVICES'), extension=ext.__file__,
                            extension_sha256=extension_hash,
                            config_sha256=hashlib.sha256((Path(a.model)/'config.json').read_bytes()).hexdigest(),
                            weight_files=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                          for p in sorted(Path(a.model).glob('*.safetensors'))],
                            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                            triton=importlib.metadata.version('triton-windows'),
                            devices=[dict(logical=i, name=torch.cuda.get_device_name(i),
                                          uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]))
    def load():
        nonlocal draft, draft_cache
        if a.mtp and draft is None:
            draft = Model.from_config(cfg, component='mtp')
            draft_cache = Cache(draft, max_num_tokens=65536, max_batch_size=max(a.batches))
            draft.load(device='cuda:0', max_chunk_size=256, max_batch_size=max(a.batches))
        if a.single:
            model.load(device='cuda:0', max_chunk_size=256, max_batch_size=max(a.batches), progressbar=True)
        else:
            model.load(tensor_p=True, tp_backend='native', reserve_per_device=2.0,
                       max_chunk_size=256, max_batch_size=max(a.batches), progressbar=True)
            assert model.loaded_tp and len(model.active_devices) >= 2
            save('allocation.json', model.plan)
            distributed = []
            for key in model.plan[model.active_devices[0]]:
                shards = [model.plan[d].get(key) for d in model.active_devices]
                shards = [tuple(s) for s in shards if s is not None and s[1] > s[0]
                          and s[2] in ('heads','K-heads','channels','slices')]
                if len(set(shards)) > 1:
                    distributed.append(key)
            assert distributed, 'No distributed tensor shards: refusing a layer-split substitute'
            save('execution-layout.json', dict(active_devices=model.active_devices,
                                               distributed_tensor_modules=distributed))
        if draft is not None:
            draft.attach_to(model)
        print('WARMUP', cfg.arch_string, flush=True)
        # A cold rank can spend minutes compiling while the other finishes.
        # Extend only the harness's host RPC wait during upstream null-backend
        # warmup. Real inference retains the upstream dispatch/collective limits.
        import exllamav3.model.model_tp as tp_runtime
        timeout = tp_runtime.DISPATCH_TIMEOUT
        real_forward_tp = model.forward_tp
        def compile_forward(x, params, last_kv_module_idx, modules):
            # Null-backend warmup has no inter-rank dependencies. Compile ranks
            # serially to avoid Windows Triton cache-file replacement races
            # when two identical GPUs compile the same previously unseen key.
            assert params.get('tp_warmup')
            x, reserve = model.prepare_inputs_for_tp(x, params)
            out = None
            for device in model.active_devices:
                model.tp_worker_dispatch(device, tp_runtime.mp_model_forward,
                                         (x, params, last_kv_module_idx, False, None))
                value = model.tp_worker_result(device)
                if device == model.tp_output_device:
                    out = value
            model.restore_tp_params(params, reserve)
            return out
        try:
            tp_runtime.DISPATCH_TIMEOUT = 600
            if model.loaded_tp:
                model.forward_tp = compile_forward
            failures = model.warmup(cache=cache, max_chunk_size=256, max_batch_size=max(a.batches),
                                    max_q_len=6, long_context=True)
            if model.loaded_tp:
                model.tp_drain_acks()
        finally:
            tp_runtime.DISPATCH_TIMEOUT = timeout
            model.forward_tp = real_forward_tp
        save('warmup.json', dict(failures=failures))
        assert not failures, 'Upstream warmup failed; see warmup.json and log'
        memory_event('loaded and warmed')
    def set_mode(enabled, trace=False, reset=True):
        if a.single:
            native_plan.enabled = enabled
            if reset:
                single_counts.update(executions=0, builds=0)
        else:
            model.tp_dispatch_all(mode, (enabled, trace, reset))
    def reports():
        if a.single:
            torch.cuda.synchronize()
            return []
        return model.tp_worker_dispatch_wait_multi(model.active_devices, report, ())
    prompt = tokenizer.encode(model.default_chat_prompt('Explain how computers execute programs.'), encode_special_tokens=True)
    def generator_run(enabled, count):
        from exllamav3 import Generator, GreedySampler
        set_mode(enabled, trace=not a.single and torch.cuda.device_count() > 2)
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                        draft_model=draft, draft_cache=draft_cache,
                        num_draft_tokens=0, dynamic_draft_tokens=False,
                        max_batch_size=max(a.batches), max_chunk_size=256,
                        ngram_match_min=0, record_draft_stats=True)
        logits, completions, summaries = [], [], []
        original = model.forward
        def capture(*args, **kw):
            result = original(*args, **kw)
            logits.append(digest(result))
            return result
        model.forward = capture
        texts = [model.default_chat_prompt(f'Explain how computers execute programs. Example {i}.') for i in range(count)]
        try:
            # Same generator: the second request exercises reused pages/slots.
            for repeat in range(2):
                completion, last = gen.generate(texts, max_new_tokens=a.steps, min_new_tokens=a.steps,
                    sampler=GreedySampler(), seed=1234, stop_conditions=[],
                    encode_special_tokens=True, completion_only=True, return_last_results=True)
                completions.append(completion)
                summaries.append([{k:r.get(k) for k in ('new_tokens', 'accepted_draft_tokens', 'rejected_draft_tokens', 'cached_tokens')}
                                  for r in last])
            evidence = reports()
            fallback_only = False
            if enabled and not a.single:
                assert evidence and all(r['blocks'] for r in evidence), 'No native TP registrations'
                active = all(b['executions'] > 0 for r in evidence for b in r['blocks'])
                if not active:
                    allowed = {'shape outside existing small-row IR domain', 'shape outside existing Qwen IR domain',
                               'prefill or execution override'}
                    fallback_only = (a.mtp and count > 1 and
                        all(b['executions'] == 0 and b['fallbacks'] and set(b['fallbacks']) <= allowed
                            for r in evidence for b in r['blocks']))
                    assert fallback_only, 'Generator neither executed native TP nor reported the expected MTP shape fallback'
            return dict(completions=completions, logits=logits, summaries=summaries, evidence=evidence,
                        fallback_only=fallback_only)
        finally:
            model.forward = original
            gen.clear_queue()
            gen.filter_pool.shutdown(wait=True)
    def run(enabled, batch, query, timed=False, forced=None):
        steps = a.bench_steps if timed else a.steps
        states = [cache.get_new_state() for _ in range(batch)] if model.caps.get('recurrent_states') else None
        past = 0
        step_ends = []
        def forward(ids):
            nonlocal past
            params = dict(attn_mode='flash_attn', cache=cache, batch_shape=(batch, 65536//batch//256*256),
                          past_len=past, last_tokens_only=1)
            if states is not None:
                params['recurrent_states'] = states
            out = model.forward(ids, params)
            past += ids.shape[1]
            step_ends.append(past)
            return out
        try:
            set_mode(enabled)
            ids = prompt.repeat(batch, (a.context+prompt.shape[1]-1)//prompt.shape[1])[:, :a.context]
            prefill_start = time.perf_counter()
            for start in range(0, a.context, 256):
                prefill_logits = forward(ids[:, start:start+256])
            prefill_logits = prefill_logits.cpu().clone()
            prefill_seconds = time.perf_counter() - prefill_start
            current = ids[:, -1:].repeat(1, query)
            if timed:
                for _ in range(4):
                    warm_logits = forward(current)
                    if a.single or torch.cuda.device_count() == 2:
                        current = warm_logits[:, -1:].argmax(-1).cpu().repeat(1, query)
            set_mode(enabled, trace=not timed, reset=False)
            if timed and a.profile and not a.single:
                model.tp_dispatch_all(profile, (str((a.output / f'profile-{int(enabled)}').resolve()),))
            outputs = []
            generated = []
            start = time.perf_counter()
            for step in range(steps):
                logits = forward(current)
                current = logits[:, -1:].argmax(-1).cpu().repeat(1, query)
                generated.append(current.tolist())
                if forced is not None:
                    current = torch.tensor(forced[step], dtype=current.dtype)
                if not timed:
                    outputs.append(logits.cpu().clone())
            elapsed = time.perf_counter() - start
            if timed and a.profile and not a.single:
                model.tp_dispatch_all(profile, (None,))
            evidence = reports()
            state = None
            if not timed and not a.single:
                slots = [(s.slot, getattr(s, 'window_beg', 0)) for s in states] if states else []
                state = model.tp_worker_dispatch_wait_multi(model.active_devices, snapshot, (id(cache), batch, past, slots, step_ends))
            if enabled and not a.single:
                assert len(evidence) == torch.cuda.device_count(), evidence
                for rank in evidence:
                    assert rank['blocks'], rank
                    assert all(b['executions'] >= steps for b in rank['blocks']), rank
            if enabled and a.single:
                assert single_counts['executions'] >= steps, 'Single-GPU IR did not execute'
            return dict(prefill_seconds=prefill_seconds, seconds=elapsed,
                        tokens_per_second=batch*steps*query/elapsed, steps=steps, evidence=evidence,
                        state=state,
                        generated=generated, last_logits=digest(logits),
                        peak_allocated=torch.cuda.max_memory_allocated() if a.single else None,
                        single_counts=single_counts.copy() if a.single else None), outputs, prefill_logits
        finally:
            if states:
                for state in states:
                    state.free()

    def generator_benchmark(enabled, warm=False):
        from exllamav3 import Generator, GreedySampler
        # Keep warmed native bindings across alternating samples. Each request
        # gets a fresh generator/page table, so prefix-cache reuse is controlled.
        set_mode(enabled, trace=False, reset=warm)
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                        draft_model=draft, draft_cache=draft_cache,
                        num_draft_tokens=0, dynamic_draft_tokens=False,
                        max_batch_size=max(a.batches), max_chunk_size=256,
                        ngram_match_min=0, record_draft_stats=True)
        text = model.default_chat_prompt('Explain how computers execute programs. Give concrete examples and describe each step in detail.')
        before = reports()
        before_single = single_counts['executions']
        try:
            start = time.perf_counter()
            completion, last = gen.generate(text, max_new_tokens=a.bench_steps, min_new_tokens=a.bench_steps,
                sampler=GreedySampler(), seed=1234, stop_conditions=[],
                encode_special_tokens=True, completion_only=True, return_last_results=True)
            elapsed = time.perf_counter()-start
            evidence = reports()
            deltas = [[b['executions']-old['executions'] for b,old in zip(r['blocks'],p['blocks'])]
                      for r,p in zip(evidence,before)]
            assert all(r['shadow_checks']==0 and not r['trace'] for r in evidence), 'Instrumented MTP timing'
            assert all(all(n>0 if enabled else n==0 for n in rank) for rank in deltas), 'MTP execution path mismatch'
            if a.single:
                assert enabled and single_counts['executions'] > before_single, 'Single-GPU MTP IR did not execute'
            assert last['new_tokens'] == a.bench_steps and last['time_generate'] > 0
            assert gen.num_draft_tokens > 0 and last['accepted_draft_tokens']+last['rejected_draft_tokens'] > 0
            return dict(native=enabled, seconds=elapsed, completion=completion,
                        tokens_per_second=last['new_tokens']/last['time_generate'],
                        wall_tokens_per_second=last['new_tokens']/elapsed,
                        draft_tokens=gen.num_draft_tokens,
                        stats={k:last.get(k) for k in ('new_tokens','prompt_tokens','cached_tokens','time_prefill',
                               'time_generate','accepted_draft_tokens','rejected_draft_tokens')},
                        single_executions_delta=single_counts['executions']-before_single if a.single else None,
                        executions_delta=deltas, evidence=evidence)
        finally:
            gen.clear_queue()
            gen.filter_pool.shutdown(wait=True)
    results = []
    try:
        load()
        for cycle in range(2 if a.reload else 1):
            if cycle:
                model.unload()
                memory_event('unloaded before reload')
                load()
            for batch in a.batches:
                for query in a.queries:
                    if query > 1 and batch > 1:
                        continue  # Multi-row verification is separately exercised by the generator.
                    if a.single:
                        ref, ref_logits, ref_pre = run(True, batch, query)
                        torch.save(dict(logits=ref_logits, prefill=ref_pre), a.output/f'logits-{cycle}-{batch}-{query}.pt')
                        result = dict(cycle=cycle, batch=batch, query=query, single=ref)
                    else:
                        ref, ref_logits, ref_pre = run(False, batch, query)
                        three = torch.cuda.device_count() > 2
                        new, new_logits, new_pre = run(True, batch, query, forced=ref['generated'] if three else None)
                        torch.save(dict(logits=new_logits, prefill=new_pre), a.output/f'logits-{cycle}-{batch}-{query}.pt')
                        errors = [float((x.float()-y.float()).abs().max()) for x,y in zip(ref_logits,new_logits)]
                        result = dict(cycle=cycle, batch=batch, query=query, reference=ref, native=new,
                                      max_abs=errors, exact=all(torch.equal(x,y) for x,y in zip(ref_logits,new_logits)),
                                      prefill_exact=torch.equal(ref_pre,new_pre),
                                      trace_exact=[x['trace']==y['trace'] for x,y in zip(ref['evidence'],new['evidence'])])
                        result['ranks_identical'] = all(r['trace'] == new['evidence'][0]['trace'] for r in new['evidence'])
                        result['state_exact'] = ref['state'] == new['state']
                        if three:
                            repeat, repeat_logits, repeat_pre = run(False, batch, query, forced=ref['generated'])
                            def errors_against(values, pre):
                                delta = torch.cat([(x.float()-y.float()).flatten() for x,y in
                                                   zip([ref_pre]+ref_logits, [pre]+values)])
                                return dict(max_abs=float(delta.abs().max()), rmse=float(delta.square().mean().sqrt()))
                            noise = errors_against(repeat_logits, repeat_pre)
                            error = errors_against(new_logits, new_pre)
                            result['stock_repeat_noise'] = noise
                            result['native_error'] = error
                            result['local_stages_exact'] = all(r['shadow_checks'] >= 4*a.steps*len(r['blocks']) for r in new['evidence'])
                            # Arrival-order BF16/FP16 rounding is measured on the
                            # same forced tokens. Exact local stage checks above
                            # are the stronger IR-specific correctness gate.
                            result['noise_envelope'] = (error['rmse'] <= 2*noise['rmse']+0.01 and
                                                        error['max_abs'] <= 2*noise['max_abs']+0.1)
                        save(f'parity-{cycle}-{batch}-{query}.json', result)
                        valid = (result['local_stages_exact'] and result['noise_envelope']) if three else (
                            result['exact'] and result['prefill_exact'] and all(result['trace_exact']) and result['state_exact'])
                        assert valid and result['ranks_identical'], 'See saved parity report'
                    results.append(result)
                    save('correctness.json', results)
                    print('PARITY', cycle, batch, query, 'PASS', flush=True)
            if cycle == 0:
                timings = []
                for pair in range(a.runs):
                    for enabled in ([True] if a.single else ([False,True] if pair%2==0 else [True,False])):
                        measured, _, _ = run(enabled, 1, 1, timed=True,
                                             forced=timings[0]['generated'] if timings and not a.single and torch.cuda.device_count()>2 else None)
                        timings.append(dict(pair=pair, native=enabled, **measured))
                        save('timings.json', timings)
                        if a.single or torch.cuda.device_count() == 2:
                            assert all(t['generated'] == timings[0]['generated'] and t['last_logits'] == timings[0]['last_logits'] for t in timings), 'Timed trajectory mismatch'
                        print('TIMING', pair, enabled, measured['tokens_per_second'], flush=True)
            if a.generator or a.mtp:
                for count in a.batches:
                    reference = generator_run(True if a.single else False, count)
                    candidate = generator_run(True, count)
                    exact = all(reference[k] == candidate[k] for k in ('completions', 'logits', 'summaries'))
                    save(f'generator-{cycle}-{count}.json', dict(reference=reference, native=candidate, exact=exact,
                                                                reference_native=a.single))
                    if not a.single and torch.cuda.device_count() > 2:
                        assert candidate['fallback_only'] or all(r['shadow_checks'] > 0 for r in candidate['evidence']), 'No exact local generator checks'
                    else:
                        assert exact, 'Generator mismatch: see report'
                    print('GENERATOR', cycle, count, 'PASS', 'STOCK FALLBACK ONLY' if candidate['fallback_only'] else 'NATIVE ACTIVE', flush=True)
            if a.generator_bench and cycle == 0:
                warmups = [generator_benchmark(True, warm=True)] if a.single else [
                    generator_benchmark(False, warm=True), generator_benchmark(True, warm=True)]
                save('generator-benchmark-warmup.json', warmups)
                measured = []
                for pair in range(a.generator_bench_runs):
                    for enabled in ([True] if a.single else ([False,True] if pair%2==0 else [True,False])):
                        row = dict(pair=pair, **generator_benchmark(enabled))
                        measured.append(row)
                        save('generator-timings.json', measured)
                        if torch.cuda.device_count() == 2:
                            assert all(r['completion']==measured[0]['completion'] and
                                       r['stats']['accepted_draft_tokens']==measured[0]['stats']['accepted_draft_tokens'] and
                                       r['stats']['rejected_draft_tokens']==measured[0]['stats']['rejected_draft_tokens']
                                       for r in measured), 'MTP timed trajectory mismatch'
                        print('MTP TIMING', pair, enabled, row['tokens_per_second'], flush=True)
        assert source_hashes() == initial_sources, 'Product sources changed during validation'
        save('passed.json', dict(passed=True, max_context=65536))
    finally:
        if model.loaded_tp or model.output_device is not None:
            model.unload()
        if draft is not None:
            draft.unload()
        memory_event('final unload')


if __name__ == '__main__':
    import torch
    with torch.inference_mode():
        main()
