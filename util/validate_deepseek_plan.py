"""Two-way real-checkpoint gates in the pinned Tabby runtime; no installation."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE_SAMPLES = []
sys.path.insert(0, str(ROOT))


def digest(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def observe(models, enabled):
    """Count real stages and compare boundary outputs outside timed runs."""
    from exllamav3.modules import native_plan, TransformerBlock
    counts, events, saved = Counter(), [], []
    def patch(obj, name, value):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)
    class Stages:
        def __init__(self, native, role): self.native, self.role = native, role
        def begin(self, x):
            counts['begin'] += 1
            counts[f'{self.role}:begin'] += 1
            return self.native.begin(x)
        def middle(self, x, y):
            counts['middle'] += 1
            return self.native.middle(x, y)
        def finish(self, x, y):
            counts['finish'] += 1
            return self.native.finish(x, y)
    emit = native_plan.deepseek
    def construct(*args):
        counts['constructed'] += 1
        result = emit(*args)
        assert type(result).__name__ == 'BC_BlockPlan'
        return result
    patch(native_plan, 'deepseek', construct)
    for model_idx, model in enumerate(models):
        role = 'target' if model_idx == 0 else 'draft'
        for module in model.modules:
            prepare=module.prepare_for_device
            def move(x,params,prepare=prepare,module=module):
                src=str(x.device)
                y=prepare(x,params)
                if src!=str(y.device) and x.ndim>=3 and x.shape[1]<=6:
                    events.append(('transfer',module.key,src,str(y.device),digest(y)))
                return y
            patch(module,'prepare_for_device',move)
        for block in model.modules:
            if not isinstance(block, TransformerBlock): continue
            block._native_deepseek = {}
            get, forward = block._native_plan_get, block.forward
            def select(b, x, params, get=get, role=role):
                n = get(b, x, params)
                counts['eligible' if n is not None else 'fallback'] += 1
                counts[f'{role}:{"eligible" if n is not None else "fallback"}:{tuple(x.shape[:2])}:prefill={bool(params.get("prefill"))}'] += 1
                if n is None:
                    counts[f'fallback:{tuple(x.shape[:2])}:prefill={bool(params.get("prefill"))}'] += 1
                    if x.shape[1] == 1 and not params.get('prefill'):
                        counts[f'{role}:decode_fallback'] += 1
                return Stages(n, role) if n is not None else None
            def fwd(x, params, forward=forward, block=block, **kwargs):
                y = forward(x, params, **kwargs)
                if not params.get('prefill') and x.shape[1] <= 6:
                    events.append((block.key, list(x.shape[:2]), digest(y)))
                return y
            patch(block, '_native_plan_get', select)
            patch(block, 'forward', fwd)
    previous = native_plan.enabled
    native_plan.enabled = enabled
    try:
        yield counts, events
    finally:
        native_plan.enabled = previous
        for obj, name, owned, value in reversed(saved):
            if owned: setattr(obj, name, value)
            else: delattr(obj, name)


def snapshot(cache, states, tables=None):
    """Only initialized logical state, excluding stale pool capacity."""
    result = {}
    for i, s in enumerate(states):
        p = s.position
        result[f'{i}:position'] = (p, s.window_beg)
        for key, layer in cache.get_all_recurrent_layers().items():
            result[f'{i}:{key}:ring'] = digest(layer.ring[s.slot, :p-s.window_beg])
            for name in ('comp_buf_kv','comp_buf_gate','idx_buf_kv','idx_buf_gate'):
                t = getattr(layer, name, None)
                if t is not None:
                    result[f'{i}:{key}:{name}'] = digest(t[s.slot, :min(p, layer.buf_rows)])
            for name in ('comp_ovl','idx_ovl'):
                t = getattr(layer, name, None)
                if t is not None:
                    result[f'{i}:{key}:{name}'] = digest(t[s.slot, :min(p//layer.module.compress_rate, layer.ovl_depth)])
        for key, layer in cache.layers.items():
            pages = layer.slot_bt(cache.num_slots)[s.slot] if tables is None else tables[i].to(layer.device)
            n = p // layer.compress_rate
            assert n <= len(pages) * layer.epp
            for name in ('pool_c','pool_r','pool_idx'):
                t = getattr(layer, name, None)
                if t is not None:
                    result[f'{i}:{key}:{name}'] = digest(t[pages].flatten(0,1)[:n])
    return result


def jobs_run(model, cache, tokenizer, draft, dc, enabled, count, steps, context,
             stagger=False, cancel=False, stochastic=False, force=False, measured=False):
    import torch
    from contextlib import ExitStack
    from unittest.mock import patch
    from exllamav3 import Generator, Job, GreedySampler, DefaultSampler
    from exllamav3.modules import native_plan
    from exllamav3.cache.dsa import DSV4State
    native_plan.enabled = enabled
    gen = Generator(model,cache,tokenizer,max_batch_size=count,draft_model=draft,draft_cache=dc,
                    num_draft_tokens=5 if draft else 0,dynamic_draft_tokens=False,
                    ngram_match_min=0,record_draft_stats=True)
    jobs=[]
    for i in range(count + int(stagger)):
        ids=tokenizer.encode(model.default_chat_prompt(f'Explain computer science topic {i} clearly with examples.'),encode_special_tokens=True)
        ids=ids.repeat(1,(context+ids.shape[1]-1)//ids.shape[1])[:,:context]
        n=steps if i else max(16,steps//2) if stagger else steps
        jobs.append(Job(ids,max_new_tokens=n,min_new_tokens=n,stop_conditions=[],
                        sampler=DefaultSampler() if stochastic else GreedySampler(),seed=1234+i,identifier=str(i)))
    state_events,rewinds,logits_events=[],[],[]
    page_owners,slot_owners={},{}
    reuse=Counter()
    cancelled=[]
    counts,events=Counter(),[]
    cycle=0
    joined=not stagger
    for d in model.active_devices: torch.cuda.synchronize(d)
    start=time.perf_counter()
    try:
        with ExitStack() as scope:
            if not measured:
                counts,events=scope.enter_context(observe([model]+([draft] if draft else []),enabled))
                original=model.forward
                def forward(input_ids,params,**kw):
                    y=original(input_ids,params,**kw)
                    logits_events.append(digest(y))
                    return y
                scope.enter_context(patch.object(model,'forward',forward))
                rewind=DSV4State.rewind
                def rw(s,n):
                    before=s.position
                    rewind(s,n)
                    assert s.position==before-n
                    rewinds.append(n)
                scope.enter_context(patch.object(DSV4State,'rewind',rw))
            if force and draft:
                sample=draft.sample_from_state
                def reject(state,params):
                    y=sample(state,params)
                    y[:,1:]=(y[:,1:]+17)%model.config.vocab_size
                    params['draft_confidence_len']=5
                    return y
                scope.enter_context(patch.object(draft,'sample_from_state',reject))
            gen.enqueue(jobs[:count])
            while gen.num_remaining_jobs():
                results=gen.iterate()
                for r in results:
                    if r.get('stage')=='error': raise r['error']
                cycle+=1
                if not measured:
                    for j in gen.active_jobs:
                        s=j.recurrent_state
                        if s is None: continue
                        seq=j.sequences[0]
                        if not j.is_prefill_done(): continue
                        assert s.position==seq.kv_position
                        tab=seq.block_index_tensor
                        record=dict(job=j.identifier,state=snapshot(cache,[s],tab),
                            ids=seq.sequence_ids.torch().tolist(),accepted=j.accepted_draft_tokens,rejected=j.rejected_draft_tokens)
                        if draft:
                            record['draft']={}
                            for key,layer in dc.layers.items():
                                for name,t in vars(layer).items():
                                    if torch.is_tensor(t) and t.ndim==3 and t.shape[0]==dc.max_num_tokens//256:
                                        record['draft'][f'{key}:{name}']=digest(t[tab[0].to(t.device)].flatten(0,1)[:seq.kv_position])
                        state_events.append(record)
                        for page in seq.allocated_pages:
                            idx=page.page_index
                            old=page_owners.get(idx)
                            if old is not None and old!=j.identifier: reuse['pages']+=1
                            page_owners[idx]=j.identifier
                        old=slot_owners.get(s.slot)
                        if old is not None and old!=j.identifier: reuse['slots']+=1
                        slot_owners[s.slot]=j.identifier
                # Cancel a live decoding job. A fast MTP job may already finish
                # before an arbitrary iteration count, which would test nothing.
                if cancel and not cancelled and jobs[0] in gen.active_jobs and jobs[0].new_tokens >= 3:
                    gen.cancel(jobs[0]); cancelled.append('0')
                if not joined and (jobs[0].is_finished or cancelled):
                    gen.enqueue(jobs[-1]);joined=True
            for d in model.active_devices: torch.cuda.synchronize(d)
            elapsed=time.perf_counter()-start
            result=dict(jobs=[dict(ids=j.sequences[0].sequence_ids.torch().tolist(),tokens=j.new_tokens,
                accepted=j.accepted_draft_tokens,rejected=j.rejected_draft_tokens) for j in jobs],
                states=state_events,logits=logits_events,boundaries=events,rewinds=rewinds,
                counts=dict(counts),reuse=dict(reuse),cancelled=cancelled,seconds=elapsed,
                tokens_per_second=sum(j.new_tokens for j in jobs)/elapsed)
            if enabled and not measured:
                assert counts['begin']>0 and counts['begin']==counts['middle']==counts['finish']
                assert counts['eligible'] == counts['begin']
                assert counts['target:begin'] > 0
                if draft and count <= 2:
                    assert counts['draft:begin'] > 0, 'Draft silently fell back'
                if not draft:
                    # Layer-split prefill marks only its final block as prefill;
                    # earlier blocks reject its long query by shape instead.
                    assert counts['target:decode_fallback'] == 0, counts
            return result
    finally:
        gen.clear_queue()
        gen.filter_pool.shutdown(wait=True)


def direct(model, cache, tokenizer, batch, query, steps, context, enabled):
    import torch
    prompt = tokenizer.encode(model.default_chat_prompt('Explain how computers execute programs.'), encode_special_tokens=True)
    ids = prompt.repeat(batch, (context + prompt.shape[1] - 1)//prompt.shape[1])[:, :context]
    states = [cache.get_new_state() for _ in range(batch)]
    records = []
    def forward(tokens, exports=False):
        params = dict(attn_mode='flash_attn', cache=cache, recurrent_states=states, input_ids=tokens)
        if exports: params['export_state_layers'] = {40,41,42}
        x = tokens
        for m in model.modules:
            x = m.forward(m.prepare_for_device(x, params), params)
        for s in states:
            s.position += tokens.shape[1]
            s.post_advance()
        return x, params
    try:
        with observe([model], enabled) as (counts, events):
            # Explicit stock prefill avoids counting short trailing chunks as decode.
            from exllamav3.modules import native_plan
            native_plan.enabled = False
            for start in range(0,context,256):
                forward(ids[:, start:start+256])
                if context >= 8192 and (start+256) % 8192 == 0:
                    print(f'Prefill {min(start+256,context)}/{context} for B{batch}/Q{query} plan={enabled}',flush=True)
            events.clear()
            native_plan.enabled = enabled
            current = ids[:, -1:].repeat(1,query)
            for step in range(steps):
                logits, params = forward(current, exports=query > 1)
                records.append(dict(logits=digest(logits), exports=[digest(t) for t in params.get('export_states',[])],
                    state=snapshot(cache,states) if step in (0,steps-1) else None))
                if step==0 and batch==1 and query==1:
                    layer=cache.get_all_recurrent_layers()[(36,0)]
                    STATE_SAMPLES.append(layer.idx_buf_kv[states[0].slot].cpu().clone())
                current = logits[:,-1:].argmax(-1).cpu().repeat(1,query)
            if enabled:
                expected = 43 * steps
                assert counts['begin'] == counts['middle'] == counts['finish'] == expected, counts
                assert counts['fallback'] == 0 and counts['constructed'] == 43, counts
            return dict(records=records, boundaries=events, counts=dict(counts))
    finally:
        for s in states: s.free()


def direct_benchmark(model, cache, tokenizer, enabled, batch, steps, context):
    """Warm direct decode; prefill, state hashing and counters are outside timing."""
    import torch
    from exllamav3.modules import native_plan
    ids = tokenizer.encode(model.default_chat_prompt('Explain how computers execute programs.'), encode_special_tokens=True)
    ids = ids.repeat(batch, (context + ids.shape[1] - 1)//ids.shape[1])[:, :context]
    states = [cache.get_new_state() for _ in range(batch)]
    def forward(tokens):
        params = dict(attn_mode='flash_attn', cache=cache, recurrent_states=states, input_ids=tokens)
        x = tokens
        for m in model.modules:
            x = m.forward(m.prepare_for_device(x, params), params)
        for s in states:
            s.position += tokens.shape[1]
            s.post_advance()
        return x
    try:
        native_plan.enabled = False
        for start in range(0, context, 256): forward(ids[:, start:start+256])
        native_plan.enabled = enabled
        current = ids[:, -1:]
        for _ in range(8): current = forward(current)[:, -1:].argmax(-1).cpu()
        for d in model.active_devices: torch.cuda.synchronize(d)
        start = time.perf_counter()
        for _ in range(steps):
            logits = forward(current)
            current = logits[:, -1:].argmax(-1).cpu()
        for d in model.active_devices: torch.cuda.synchronize(d)
        elapsed = time.perf_counter() - start
        return dict(seconds=elapsed, tokens_per_second=batch*steps/elapsed,
                    logits=digest(logits), tokens=current.tolist(), state=snapshot(cache, states))
    finally:
        for s in states: s.free()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=r'D:\tabbyAPI\models\turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase', choices=['short','long','generator','benchmark'], default='short')
    p.add_argument('--context', type=int, default=300)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--draft-device', default='cuda:0', help='Validation placement only; keep the draft on a participating device')
    p.add_argument('--reload', action='store_true')
    p.add_argument('--case', type=int, action='append')
    p.add_argument('--cache-tokens', type=int)
    p.add_argument('--max-batch', type=int, default=4, choices=(1,2,3,4))
    p.add_argument('--budgets', type=float, nargs='+', default=[85,40], help='Validation-only per-device loading budgets in GiB')
    p.add_argument('--bench-batch', type=int, action='append', choices=(1,2,3,4))
    p.add_argument('--diagnose', action='store_true')
    p.add_argument('--extension-dir',type=Path,default=ROOT/'native-results/block-ir/extension')
    p.add_argument('--reference-dir',type=Path,help='Require exact parity with a retained equivalent topology run')
    args = p.parse_args()
    sys.path.insert(1,str(args.extension_dir.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, data): (args.output/name).write_text(json.dumps(data,indent=2), encoding='utf-8')
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.modules import native_plan
    import exllamav3_ext
    torch.set_grad_enabled(False)
    save('environment.json', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), extension=exllamav3_ext.__file__,
        environment={k:v for k,v in os.environ.items() if k.startswith(('EXL3_','CUDA_'))}))
    cfg = Config.from_directory(args.model)
    model, tokenizer = Model.from_config(cfg), Tokenizer.from_config(cfg)
    capacity = max(8192, ((args.context + args.steps*6 + 255)//256)*256*args.max_batch)
    if args.cache_tokens: capacity=args.cache_tokens
    cache = Cache(model, max_num_tokens=capacity, max_batch_size=args.max_batch, max_history=5)
    model.load(use_per_device=args.budgets, max_chunk_size=256, max_batch_size=args.max_batch, progressbar=True)
    save('placement.json', [dict(key=m.key, device=str(m.device)) for m in model.modules])
    if args.phase in ('short','long'):
        # Preserve a cold stock sample and warm the existing attention companions
        # before comparing backends. No plan or inference implementation changes.
        save('cold-stock.json',direct(model,cache,tokenizer,1,1,2,args.context,False))
        shapes = [(1,1)] if args.diagnose else [(1,1),(2,1),(3,1),(4,1),(1,6),(2,6)] if args.phase == 'short' else [(1,1),(1,6)]
        for b,q in shapes:
            if b > args.max_batch: continue
            runs = []
            for enabled in (False,True):
                result = direct(model,cache,tokenizer,b,q,args.steps,args.context,enabled)
                save(f'b{b}-q{q}-{int(enabled)}.json', result)
                runs.append(result)
            if args.diagnose:
                torch.save(STATE_SAMPLES,args.output/'state-samples.pt')
                diffs=[]
                for a,z in zip(STATE_SAMPLES,STATE_SAMPLES[1:]):
                    diff=(a.float()-z.float()).abs()
                    diffs.append(dict(equal=torch.equal(a,z), max=diff.max().item(),
                        differing=(a!=z).nonzero().tolist()[:20], n=int((a!=z).sum()),
                        nonfinite=int((~torch.isfinite(a)).sum())))
                save('state-differences.json',diffs)
            assert runs[0]['records'] == runs[1]['records'], f'output/state mismatch B{b}/Q{q}'
            assert runs[0]['boundaries'] == runs[1]['boundaries'], f'boundary mismatch B{b}/Q{q}'
            print(f'PASS exact B{b}/Q{q}', flush=True)
        from exllamav3.modules.native_deepseek import get_native_block
        block=model.modules[model.first_block_idx]
        x=torch.zeros((1,1,4,4096),device=block.device)
        native_plan.enabled=True
        for params in ({'capture':None},{'quant_preserve':False},{'loras':[]},{'prefill':True},{'backend':None}):
            assert get_native_block(block,x,params) is None
            assert torch.count_nonzero(x)==0
        assert get_native_block(block,x.repeat(1,7,1,1),{}) is None
        norm=block.attn_norm
        old=norm.constant_scale
        norm.constant_scale=2
        assert get_native_block(block,x,{}) is None
        norm.constant_scale=old
        assert get_native_block(block,x,{}) is not None
        save('fallback.json',dict(modes=True,shape=True,configuration=True,unchanged_residual=True))
        if args.reload:
            before=direct(model,cache,tokenizer,1,1,2,args.context,True)
            save('reload-before.json', before)
            before_placement = [(m.key, str(m.device)) for m in model.modules]
            model.unload()
            assert all(not getattr(m,'_native_deepseek',{}) for m in model.modules)
            model.load(use_per_device=args.budgets,max_chunk_size=256,max_batch_size=args.max_batch)
            save('reload-placement.json', [dict(key=m.key, device=str(m.device)) for m in model.modules])
            after_stock=direct(model,cache,tokenizer,1,1,2,args.context,False)
            save('reload-stock.json', after_stock)
            after=direct(model,cache,tokenizer,1,1,2,args.context,True)
            save('reload-after.json', after)
            for key in ('records', 'boundaries'):
                assert after_stock[key] == after[key], f'Reloaded stock/plan {key} mismatch'
            same_placement = before_placement == [(m.key, str(m.device)) for m in model.modules]
            # Autosplit measures transient allocations anew. Moving a block
            # between GPU architectures changes its arithmetic; compare each
            # loaded placement against stock, and retain cross-reload evidence.
            if same_placement:
                assert before == after, 'Same-placement reload mismatch'
            save('reload.json',dict(exact=True, placement_unchanged=same_placement,
                                   cross_reload_exact=before == after))
    else:
        draft=dc=None
        if args.mtp:
            draft=Model.from_config(cfg,component='mtp')
            dc=Cache(draft,max_num_tokens=capacity,max_batch_size=args.max_batch)
            draft.load(device=torch.device(args.draft_device),max_chunk_size=256,max_batch_size=args.max_batch)
            draft.attach_to(model)
            save('draft-placement.json', [dict(key=m.key, device=str(m.device)) for m in draft.modules])
        cases=[dict(count=b) for b in (1,2,3,4)]
        cases += [dict(count=3,stagger=True),dict(count=3,stagger=True,cancel=True),
                  dict(count=2,stochastic=True)]
        if draft: cases += [dict(count=2,force=True),dict(count=1,force=True)]
        for i,case in enumerate(cases):
            if args.case and i not in args.case: continue
            if case['count'] > args.max_batch: continue
            results=[]
            for enabled in (False,True):
                r=jobs_run(model,cache,tokenizer,draft,dc,enabled,steps=args.steps,context=args.context,**case)
                save(f'case{i}-{int(enabled)}.json',r)
                results.append(r)
            for key in ('jobs','states','logits','boundaries','rewinds','reuse','cancelled'):
                assert results[0][key]==results[1][key],f'case{i} {key} mismatch'
            print(f'PASS generator {case}',flush=True)
        if args.phase=='benchmark':
            samples=[]
            for batch in args.bench_batch or [1]:
                for repeat in range(3):
                    for enabled in ((False,True) if repeat%2==0 else (True,False)):
                        if draft:
                            jobs_run(model,cache,tokenizer,draft,dc,enabled,batch,32,args.context,measured=True)
                            r=jobs_run(model,cache,tokenizer,draft,dc,enabled,batch,max(256,args.steps),args.context,measured=True)
                        else:
                            r=direct_benchmark(model,cache,tokenizer,enabled,batch,max(256,args.steps),args.context)
                        samples.append(dict(batch=batch,repeat=repeat,plan=enabled,**r))
                        save('benchmark.json',samples)
                    pair=samples[-2:]
                    for key in (('jobs',) if draft else ('logits','tokens','state')):
                        assert pair[0][key] == pair[1][key], f'Benchmark {key} mismatch'
                    print(f'PASS benchmark B{batch} pair {repeat}',flush=True)
        if draft: draft.unload()
    participating = {torch.device(m.device).index for m in model.modules if m.device is not None and torch.device(m.device).type == 'cuda'}
    save('memory.json', {str(i):dict(participating=i in participating, peak=torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())})
    if args.reference_dir:
        compared=[]
        for path in sorted(args.output.glob('*.json')):
            if not (path.name.startswith('b') and '-q' in path.name or path.name.startswith('case')): continue
            a=json.loads(path.read_text())
            b=json.loads((args.reference_dir/path.name).read_text())
            for key in (('records','boundaries') if 'records' in a else ('jobs','states','logits','boundaries','rewinds','reuse','cancelled')):
                assert a[key]==b[key],f'Topology mismatch {path.name}: {key}'
            for key in ('constructed','eligible','fallback','begin','middle','finish'):
                assert a['counts'].get(key,0)==b['counts'].get(key,0),f'Topology count mismatch: {key}'
            compared.append(path.name)
        assert compared
        assert json.loads((args.output/'placement.json').read_text())==json.loads((args.reference_dir/'placement.json').read_text())
        for i in range(torch.cuda.device_count()):
            if i not in participating: assert torch.cuda.max_memory_allocated(i)==0
        save('topology-parity.json',dict(reference=str(args.reference_dir),files=compared,exact=True))
    save('passed.json', dict(phase=args.phase, context=args.context, steps=args.steps))
    model.unload()


if __name__ == '__main__':
    import torch
    with torch.inference_mode(): main()
