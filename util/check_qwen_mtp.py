"""Exact stock/native MTP comparison through the real Generator, including rejection.

All tensor reads synchronize and belong ONLY to correctness tests, not benchmarks.
Records use logical page order, not allocator-dependent physical page numbers.
"""
from collections import Counter
from contextlib import ExitStack
import hashlib
from unittest.mock import patch
import torch
from util.check_qwen_native import digest


class NativeCounts:
    def __init__(self, model, draft, generator=None):
        self.models = [('target', model), ('draft', draft)]
        self.counts = Counter()
        self.scope = ExitStack()
        self.generator = generator

    def __enter__(self):
        from exllamav3.modules import TransformerBlock, native_qwen
        roles = {}
        for role, model in self.models:
            for block in model.modules:
                if not isinstance(block, TransformerBlock):
                    continue
                roles[id(block)] = role
                original = block.forward
                def forward(x, params, *a, _f=original, _r=role, **kw):
                    self.counts[f'{_r}/q{x.shape[1]}/all'] += 1
                    return _f(x, params, *a, **kw)
                self.scope.enter_context(patch.object(block, 'forward', forward))
        original = native_qwen.get_native_block
        def choose(block, x, params):
            result = original(block, x, params)
            if result is not None:
                self.counts[f'{roles[id(block)]}/q{x.shape[1]}/native'] += 1
            return result
        self.scope.enter_context(patch.object(native_qwen, 'get_native_block', choose))
        if self.generator is not None:
            propose = self.generator.iterate_draftmodel_mtp_gen
            def draft_tokens(results):
                ids = propose(results)
                if ids is not None:
                    self.counts['draft_tokens_proposed'] += ids.numel()
                return ids
            self.scope.enter_context(patch.object(self.generator, 'iterate_draftmodel_mtp_gen', draft_tokens))
        return self

    def __exit__(self, *exc):
        self.scope.__exit__(*exc)

    def report(self):
        result = dict(sorted(self.counts.items()))
        result['total'] = sum(v for k, v in self.counts.items() if k.endswith('/all'))
        result['native'] = sum(v for k, v in self.counts.items() if k.endswith('/native'))
        result['fallback'] = result['total'] - result['native']
        return result


def clear_kv(cache):
    for layer in cache.layers.values():
        for tensor in layer.get_tensors():
            tensor.zero_()


def recurrent_snapshot(state):
    if state is None:
        return None
    return dict(position=state.position, last_history=state.last_history, layers={
        str(key): [digest(t[state.slot]) for t in layer.get_state_tensors()]
        for key, layer in state.cache.get_all_recurrent_layers().items()})


def kv_snapshot(cache, job):
    # All allocated pages, including speculative suffixes and QSA side planes.
    # Both runs explicitly zero backing storage first, making unwritten tails defined.
    pages = job.sequences[0].block_index_tensor[0, :len(job.sequences[0].allocated_pages)].long()
    return {str(key): [digest(t.index_select(0, pages.to(t.device))) for t in layer.get_tensors()]
            for key, layer in cache.layers.items()}


@torch.inference_mode()
def check_mtp(model, draft, cache, draft_cache, tokenizer, prompt, make_generator,
              output, tokens=128, force_rejections=False):
    from exllamav3 import Job, GreedySampler
    from exllamav3.modules import TransformerBlock, transformer
    from exllamav3.modules.gated_delta_net import GDNState
    original_flag = transformer._native_qwen_enabled

    def run(enabled):
        transformer._native_qwen_enabled = enabled
        clear_kv(cache)
        clear_kv(draft_cache)
        gen = make_generator()
        job = Job(tokenizer.encode(prompt, encode_special_tokens=True), max_new_tokens=tokens,
                  min_new_tokens=tokens, sampler=GreedySampler(), seed=1234, stop_conditions=[])
        records = []
        phase = ''
        round_idx = 0
        rewinds = Counter()
        forced = 0

        def record(kind, **data):
            records.append(dict(kind=kind, round=round_idx, **data))

        def snapshot(kind):
            seq = job.sequences[0]
            record(kind, recurrent=recurrent_snapshot(job.recurrent_state),
                   target_kv=kv_snapshot(cache, job), draft_kv=kv_snapshot(draft_cache, job),
                   kv_position=seq.kv_position,
                   page_positions=[p.kv_position for p in seq.allocated_pages],
                   accepted=job.accepted_draft_tokens, rejected=job.rejected_draft_tokens,
                   token_ids=seq.sequence_ids.torch().tolist(),
                   carry=digest(job.mtp_last_hidden) if job.mtp_last_hidden is not None else None)

        with NativeCounts(model, draft, gen) as counts, ExitStack() as scope:
            for role, mod in [('target', model), ('draft', draft)]:
                original = mod.forward
                def forward(input_ids, params=None, _f=original, _r=role):
                    y = _f(input_ids, params)
                    record('forward', role=_r, ids=input_ids.tolist(), result=digest(y),
                           history=bool(params.get('recurrent_history')),
                           exports=[digest(t) for t in params.get('export_states', [])])
                    return y
                scope.enter_context(patch.object(mod, 'forward', forward))
                for block in mod.modules:
                    if not isinstance(block, TransformerBlock):
                        continue
                    original = block.forward
                    def block_forward(x, params, *a, _f=original, _key=block.key, **kw):
                        y = _f(x, params, *a, **kw)
                        record('block', key=_key, shape=list(x.shape), residual=digest(y))
                        return y
                    scope.enter_context(patch.object(block, 'forward', block_forward))

            original = draft.sample_from_state
            def sample(state, params):
                nonlocal phase
                phase = 'draft_sample'
                try:
                    ids = original(state, params)
                    record('draft_ids', ids=ids.cpu().tolist())
                    return ids
                finally:
                    phase = ''
            scope.enter_context(patch.object(draft, 'sample_from_state', sample))
            lm = model.modules[model.logit_layer_idx]
            original_lm = lm.forward
            def head(*a, **kw):
                y = original_lm(*a, **kw)
                if phase == 'draft_sample':
                    record('draft_logits', logits=digest(y))
                return y
            scope.enter_context(patch.object(lm, 'forward', head))

            original_draft = gen.iterate_draftmodel_mtp_gen
            def propose(results):
                nonlocal forced
                ids = original_draft(results)
                if ids is not None:
                    record('proposals', ids=ids.tolist())
                    if force_rejections:
                        # Deliberate test fault only: overwrite one proposal, retaining
                        # the genuine draft-logit/proposal record for comparison.
                        col = round_idx % ids.shape[1]
                        ids[0, col] = (ids[0, col] + 1) % model.config.vocab_size
                        forced += 1
                        record('injected_proposals', column=col, ids=ids.tolist())
                return ids
            scope.enter_context(patch.object(gen, 'iterate_draftmodel_mtp_gen', propose))

            original_rewind = GDNState.rewind
            def rewind(state, n):
                if state is job.recurrent_state:
                    snapshot('before_rewind')
                    rewinds[n] += 1
                    original_rewind(state, n)
                    snapshot('after_rewind')
                else:
                    original_rewind(state, n)
            scope.enter_context(patch.object(GDNState, 'rewind', rewind))
            original_deallocate = job.deallocate_pages
            def release():
                if job.recurrent_state is not None:
                    snapshot('final_before_release')
                return original_deallocate()
            scope.enter_context(patch.object(job, 'deallocate_pages', release))
            original_verify = gen.iterate_gen
            def verify(results, draft_tokens=None):
                nonlocal round_idx
                original_verify(results, draft_tokens)
                if job.recurrent_state is not None:
                    snapshot('committed')
                    if job.recurrent_state.position != job.sequences[0].kv_position:
                        raise AssertionError('Committed recurrent/KV positions differ')
                round_idx += 1
                if round_idx % 16 == 0:
                    print(f'MTP parity native={enabled} forced={force_rejections}: round {round_idx}', flush=True)
            scope.enter_context(patch.object(gen, 'iterate_gen', verify))
            try:
                gen.enqueue(job)
                while gen.num_remaining_jobs():
                    for result in gen.iterate():
                        if result.get('stage') == 'error':
                            raise result['error']
            finally:
                gen.clear_queue()
                gen.filter_pool.shutdown(wait=True)
        if not any(n > 0 and count for n, count in rewinds.items()):
            raise AssertionError('Parity run did not exercise a real speculative rollback')
        if job.accepted_draft_tokens == 0:
            raise AssertionError('Parity run did not exercise accepted draft tokens')
        if counts.counts['draft_tokens_proposed'] != job.accepted_draft_tokens + job.rejected_draft_tokens:
            raise AssertionError('Draft proposal accounting mismatch')
        return records, dict(rounds=round_idx, accepted=job.accepted_draft_tokens,
                             rejected=job.rejected_draft_tokens, rewind_counts=dict(rewinds),
                             forced_windows=forced, calls=counts.report())

    try:
        stock, stock_summary = run(False)
        native, native_summary = run(True)
        import json
        output.mkdir(parents=True, exist_ok=True)
        for name, records in [('stock', stock), ('native', native)]:
            (output / f'{name}.json').write_text(json.dumps(records), encoding='utf-8')
        if len(stock) != len(native):
            raise AssertionError(f'MTP trace length mismatch: {len(stock)} vs {len(native)}')
        for i, (a, b) in enumerate(zip(stock, native)):
            if a != b:
                fields = [key for key in a if a[key] != b.get(key)]
                raise AssertionError(f'MTP exact parity failed at event {i}, {a["kind"]}, round {a["round"]}: {fields}')
        nc = native_summary['calls']
        if not nc.get('target/q5/native') or not nc.get('draft/q1/native'):
            raise AssertionError(f'MTP fast path was not exercised: {nc}')
        result = dict(exact=True, tokens=tokens, force_rejections=force_rejections,
                      stock=stock_summary, native=native_summary, compared_events=len(stock))
        (output / 'parity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        transformer._native_qwen_enabled = original_flag
