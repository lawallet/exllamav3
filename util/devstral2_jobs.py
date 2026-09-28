"""Paged-KV generator oracle, adapted from the established DeepSeek scheduler gate."""
from types import SimpleNamespace
from collections import Counter
import time
from util.validate_gemma4_v2 import digest

def jobs_run(model, cache, tokenizer, draft, dc, enabled, count, steps, context,
             stagger=False, cancel=False, stochastic=False, force=False, measured=False):
    import torch
    from contextlib import ExitStack
    from unittest.mock import patch
    from exllamav3 import Generator, Job, GreedySampler, DefaultSampler
    from exllamav3.modules import native_plan
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
                        if s is None: s = SimpleNamespace(position=j.sequences[0].kv_position)
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
                        old=slot_owners.get(getattr(s, 'slot', None))
                        if hasattr(s, 'slot') and old is not None and old!=j.identifier: reuse['slots']+=1
                        slot_owners[getattr(s, 'slot', None)]=j.identifier
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
