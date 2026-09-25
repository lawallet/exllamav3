"""Real dynamic-generator parity and latency measurement, with per-job state isolation."""
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from util.check_qwen_native import digest
from util.check_qwen_mtp import clear_kv, recurrent_snapshot, kv_snapshot

PROMPTS = [
    'Write a Python function that computes prime numbers. Explain the sieve algorithm step by step.',
    'Describe how to bake a chocolate cake, including ingredients and detailed preparation instructions.',
    'Explain how ocean tides arise from gravity. Describe the roles of the Moon and Sun.',
    'Write a fantasy story about a traveling musician who discovers a hidden city in the mountains.',
    'Explain how a relational database index speeds up queries, with a concrete SQL example.',
]


@torch.inference_mode()
def run_jobs(model, draft, cache, dc, tokenizer, *, native, mtp, count, lengths,
             stagger=False, correctness=False, force=False, stochastic=False):
    from exllamav3 import Generator, Job, GreedySampler, DefaultSampler
    from exllamav3.modules import TransformerBlock, transformer, native_qwen
    from exllamav3.modules.gated_delta_net import GDNState
    if correctness:
        print(f'START parity native={native} mtp={mtp} jobs={count} stagger={stagger}',flush=True)
    transformer._native_qwen_enabled = native
    if correctness:
        clear_kv(cache)
        if mtp:
            clear_kv(dc)
    gen = Generator(model, cache, tokenizer, max_batch_size=count,
                    draft_model=draft if mtp else None, draft_cache=dc if mtp else None,
                    num_draft_tokens=4 if mtp else 0, dynamic_draft_tokens=False,
                    ngram_match_min=0, record_draft_stats=True)
    jobs = [Job(tokenizer.encode(PROMPTS[i]), max_new_tokens=n, min_new_tokens=n,
                stop_conditions=[], sampler=DefaultSampler() if stochastic else GreedySampler(),
                seed=1234+i, identifier=str(i)) for i,n in enumerate(lengths)]
    records, counts, shapes = [], Counter(), Counter()
    rewinds = {j.identifier: [] for j in jobs}
    proposals = Counter()
    samples = {j.identifier: [] for j in jobs}
    endings, enqueued = {}, {}
    cycle, phase = 0, ''
    prev_active = None

    def record(kind, **kw):
        records.append(dict(kind=kind, cycle=cycle, **kw))

    def active():
        return [j for j in gen.active_jobs if j.is_prefill_done()]

    def state(j, full=True):
        s = j.sequences[0]
        value = dict(ids=s.sequence_ids.torch().tolist(), position=s.kv_position,
                     page_positions=[p.kv_position for p in s.allocated_pages],
                     accepted=j.accepted_draft_tokens, rejected=j.rejected_draft_tokens,
                     rng=hashlib.sha256(repr(j.rng.getstate()).encode()).hexdigest(),
                     carry=digest(j.mtp_last_hidden) if j.mtp_last_hidden is not None else None)
        if full:
            value.update(recurrent=recurrent_snapshot(j.recurrent_state), kv=kv_snapshot(cache,j))
            if mtp:
                value['draft_kv'] = kv_snapshot(dc,j)
        return value

    def snapshot(kind, full=True):
        values={j.identifier:state(j,full) for j in gen.active_jobs if j.recurrent_state is not None}
        record(kind, jobs=values)
        return values

    def ownership():
        live = [j for j in gen.active_jobs if j.recurrent_state is not None]
        slots = [j.recurrent_state.slot for j in live]
        assert len(slots) == len(set(slots)), 'Two live jobs share a recurrent slot'
        seen = set()
        for j in live:
            s = j.sequences[0]
            pages = [p.page_index for p in s.allocated_pages]
            assert not seen.intersection(pages), 'Independent prompts share mutable pages'
            seen.update(pages)
            assert s.block_index_tensor[0,:len(pages)].tolist() == pages
            assert j.recurrent_state.cache is cache
        # Allocator slot/page numbers may differ between runs; compare logical
        # ownership after asserting physical disjointness and table correctness.
        record('ownership', jobs=[j.identifier for j in live], unique_slots=len(set(slots)),
               page_counts=[len(j.sequences[0].allocated_pages) for j in live])

    torch.cuda.synchronize()
    start_epoch, start = time.time(), time.perf_counter()
    with ExitStack() as scope:
        roles = {}
        for role, mod in [('target',model)] + ([('draft',draft)] if mtp else []):
            for b in mod.modules:
                if not isinstance(b,TransformerBlock):
                    continue
                roles[id(b)] = role
                f = b.forward
                def block(x,params,*a,_f=f,_r=role,_k=b.key,**kw):
                    key = f'{_r}/b{x.shape[0]}/q{x.shape[1]}'
                    counts[key+'/all'] += 1
                    y = _f(x,params,*a,**kw)
                    if correctness:
                        record('block',role=_r,key=_k,shape=list(x.shape),rows=[digest(t) for t in y])
                    return y
                scope.enter_context(patch.object(b,'forward',block))
            if correctness:
                f = mod.forward
                def forward(input_ids,params=None,_f=f,_r=role):
                    ids=input_ids
                    # A prefill/repair call for one job must not mutate the other jobs.
                    table=params.get('block_table'); seqlens=params.get('cache_seqlens')
                    touched=set()
                    if table is not None and seqlens is not None:
                        from exllamav3.constants import PAGE_SIZE
                        for row in range(table.shape[0]):
                            pages=(int(seqlens[row])+ids.shape[1]+PAGE_SIZE-1)//PAGE_SIZE
                            touched.update(table[row,:pages].tolist())
                    others = [j for j in gen.active_jobs if j.recurrent_state is not None and
                              all(p.page_index not in touched for p in j.sequences[0].allocated_pages)]
                    before = {j.identifier:state(j) for j in others}
                    y = _f(ids,params)
                    assert before == {j.identifier:state(j) for j in others}, 'Forward changed another job state'
                    record('forward',role=_r,ids=ids.tolist(),rows=[digest(t) for t in y],
                           exports=[[digest(t) for t in e] for e in params.get('export_states',[])])
                    return y
                scope.enter_context(patch.object(mod,'forward',forward))
        choose = native_qwen.get_native_block
        def native_block(b,x,params):
            result=choose(b,x,params)
            if result is not None:
                counts[f'{roles[id(b)]}/b{x.shape[0]}/q{x.shape[1]}/native'] += 1
            return result
        scope.enter_context(patch.object(native_qwen,'get_native_block',native_block))

        if mtp:
            propose = gen.iterate_draftmodel_mtp_gen
            def propose_tokens(results):
                ids=propose(results)
                if ids is not None:
                    live=active()
                    # Generator returns the full-capacity backing view; unused rows are stale.
                    for row,j in enumerate(live):
                        proposals[j.identifier] += ids.shape[1]
                        if correctness:
                            record('proposals',job=j.identifier,ids=ids[row].tolist())
                        if force and int(j.identifier)%2 == 0:
                            col=(cycle+int(j.identifier))%ids.shape[1]
                            ids[row,col]=(ids[row,col]+1)%model.config.vocab_size
                            if correctness:
                                record('injected',job=j.identifier,ids=ids[row].tolist())
                return ids
            scope.enter_context(patch.object(gen,'iterate_draftmodel_mtp_gen',propose_tokens))
            if correctness:
                sample = draft.sample_from_state
                def draft_sample(*a,**kw):
                    nonlocal phase
                    phase='draft'
                    try:
                        ids=sample(*a,**kw)
                        record('draft_sample',ids=ids.tolist())
                        return ids
                    finally:
                        phase=''
                scope.enter_context(patch.object(draft,'sample_from_state',draft_sample))
                head=model.modules[model.logit_layer_idx]
                f=head.forward
                def lm(*a,**kw):
                    y=f(*a,**kw)
                    if phase=='draft':record('draft_logits',rows=[digest(t) for t in y])
                    return y
                scope.enter_context(patch.object(head,'forward',lm))
                rewind=GDNState.rewind
                def rewind_state(rs,n):
                    owner=next((j for j in jobs if j.recurrent_state is rs),None)
                    if owner is None:return rewind(rs,n)
                    others=[j for j in gen.active_jobs if j is not owner and j.recurrent_state is not None]
                    before=snapshot('before_rewind')
                    rewind(rs,n)
                    rewinds[owner.identifier].append(n)
                    after=snapshot('after_rewind')
                    assert all(before[j.identifier]==after[j.identifier] for j in others), 'Rewind contaminated another job'
                scope.enter_context(patch.object(GDNState,'rewind',rewind_state))
        if correctness:
            for j in jobs:
                release=j.deallocate_pages
                def deallocate(_j=j,_f=release):
                    if _j.recurrent_state is not None:
                        record('final_state',job=_j.identifier,state=state(_j))
                    return _f()
                scope.enter_context(patch.object(j,'deallocate_pages',deallocate))
        verify=gen.iterate_gen
        def iteration(results,draft_tokens=None):
            nonlocal prev_active
            live=active()
            b,q=len(live),1 if draft_tokens is None else draft_tokens.shape[1]+1
            shapes[f'b{b}/q{q}'] += 1
            ids=[j.identifier for j in live]
            changed=ids!=prev_active
            if correctness:
                ownership()
            verify(results,draft_tokens)
            if correctness:
                full=mtp or changed or cycle in (0,1,7) or cycle%16==0
                snapshot('committed',full)
                for j in active():
                    assert j.recurrent_state.position==j.sequences[0].kv_position
            prev_active=ids
        scope.enter_context(patch.object(gen,'iterate_gen',iteration))
        try:
            gen.enqueue(jobs[:count])
            enqueued.update({j.identifier:time.perf_counter() for j in jobs[:count]})
            joined=not stagger
            join_after=None
            while gen.num_remaining_jobs():
                results=gen.iterate()
                now=time.perf_counter()
                for r in results:
                    if r.get('stage')=='error':raise r['error']
                    jid=r.get('identifier')
                    if jid is not None and r.get('token_ids') is not None:
                        samples[jid].append((now,r['token_ids'].numel()))
                    if jid is not None and r.get('eos'):
                        endings[jid]=dict(result=r,wall=now)
                cycle+=1
                if not joined and jobs[1].is_finished and join_after is None:
                    join_after=cycle+3  # Observe the smaller batch before growing it again.
                if not joined and join_after is not None and cycle>=join_after:
                    gen.enqueue(jobs[count]);enqueued[jobs[count].identifier]=time.perf_counter();joined=True
                if correctness and cycle%(8 if mtp else 16)==0:
                    print(f'parity native={native} mtp={mtp} jobs={count} stagger={stagger}: cycle {cycle}',flush=True)
        finally:
            gen.clear_queue()
            gen.filter_pool.shutdown(wait=True)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    total=sum(v for k,v in counts.items() if k.endswith('/all'))
    used=sum(v for k,v in counts.items() if k.endswith('/native'))
    report=dict(native=native,mtp=mtp,jobs=count,stagger=stagger,force=force,stochastic=stochastic,
                start_epoch=start_epoch,end_epoch=time.time(),seconds=elapsed,
                aggregate_tps=sum(j.new_tokens for j in jobs)/elapsed,shapes=dict(shapes),
                calls=dict(counts,total=total,native=used,fallback=total-used),per_job={})
    join_time=enqueued.get(str(count)) if stagger else None
    report['join_time_seconds']=join_time-start if join_time is not None else None
    for j in jobs:
        jid=j.identifier
        assert j.is_finished and j.new_tokens==lengths[int(jid)], 'Job did not complete requested output'
        events=samples[jid]
        done=endings[jid]
        duration=done['wall']-enqueued[jid]
        gaps=[b[0]-a[0] for a,b in zip(events,events[1:])]
        join_gaps=[b[0]-a[0] for a,b in zip(events,events[1:])
                   if join_time is not None and a[0]<=join_time<=b[0]]
        report['per_job'][jid]=dict(tokens=j.new_tokens,completion_seconds=duration,
            prompt_tokens=len(j.sequences[0].input_ids),
            first_token_seconds=events[0][0]-enqueued[jid] if events else None,
            effective_tps=j.new_tokens/duration,decode_tps=j.new_tokens/max(j.time_generate,1e-9),
            max_delivery_gap_seconds=max(gaps,default=0),proposed=proposals[jid],
            join_delivery_gap_seconds=max(join_gaps,default=0),
            accepted=j.accepted_draft_tokens,rejected=j.rejected_draft_tokens,
            acceptance=j.accepted_draft_tokens/max(1,proposals[jid]),rewinds=rewinds[jid],
            completion_sha256=hashlib.sha256(j.full_completion.encode()).hexdigest(),
            ids=j.sequences[0].sequence_ids.torch().tolist(),
            rng=hashlib.sha256(repr(j.rng.getstate()).encode()).hexdigest())
        if mtp:assert proposals[jid]==j.accepted_draft_tokens+j.rejected_draft_tokens
    return records,report


def check_jobs(model,draft,cache,dc,tokenizer,output,**kw):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    a,sa=run_jobs(model,draft,cache,dc,tokenizer,native=False,correctness=True,**kw)
    b,sb=run_jobs(model,draft,cache,dc,tokenizer,native=True,correctness=True,**kw)
    for name,data in [('stock',a),('native',b),('stock-summary',sa),('native-summary',sb)]:
        (output/(name+'.json')).write_text(json.dumps(data),encoding='utf-8')
    assert len(a)==len(b),(len(a),len(b))
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:raise AssertionError(f'Event {i}, {x["kind"]}, cycle {x["cycle"]}: {[k for k in x if x[k]!=y.get(k)]}')
    for jid in sa['per_job']:
        for k in ('ids','rng','completion_sha256','proposed','accepted','rejected','rewinds'):
            assert sa['per_job'][jid][k]==sb['per_job'][jid][k],(jid,k)
    if kw['mtp']:
        histories=[v['rewinds'] for v in sb['per_job'].values()]
        assert any(any(n>0 for n in h) for h in histories),'No rejection'
        assert len({tuple(h) for h in histories})>1,'Jobs had identical speculative histories'
    assert sb['calls']['native']>0,'No native execution'
    role='draft' if kw['mtp'] else 'target'
    assert sb['calls'].get(f'{role}/b{kw["count"]}/q1/native',0)>0,'Concurrent fast path not exercised'
    result=dict(exact=True,events=len(a),shapes=sb['shapes'],calls=sb['calls'],
                per_job={j:{k:v[k] for k in ('tokens','completion_sha256','proposed','accepted','rejected','rewinds')}
                         for j,v in sb['per_job'].items()})
    (output/'parity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS',output,flush=True)
    return result
