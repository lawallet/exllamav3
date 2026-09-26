"""Real dynamic-generator parity and latency measurement, with per-job state isolation."""
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from util.check_glm_native import digest
from util.check_qwen_mtp import clear_kv
from util.check_glm_mtp import recurrent as recurrent_snapshot, kv

def kv_snapshot(cache,job):
    return kv(cache,job,job.sequences[0].kv_position)

PROMPTS = [
    'Write a Python function that computes prime numbers. Explain the sieve algorithm step by step.',
    'Describe how to bake a chocolate cake, including ingredients and detailed preparation instructions.',
    'Explain how ocean tides arise from gravity. Describe the roles of the Moon and Sun.',
    'Write a fantasy story about a traveling musician who discovers a hidden city in the mountains.',
    'Explain how a relational database index speeds up queries, with a concrete SQL example.',
]


@torch.inference_mode()
def run_jobs(model, draft, cache, dc, tokenizer, *, native, mtp, count, lengths,
             stagger=False, correctness=False, force=False, stochastic=False, require_page_reuse=False, cancel=False, diagnostic=False, prefix_tokens=None):
    from exllamav3 import Generator, Job, GreedySampler, DefaultSampler
    from exllamav3.modules import TransformerBlock, transformer, native_glm
    from exllamav3.modules.gated_delta_net import GDNState
    if correctness:
        print(f'START parity native={native} mtp={mtp} jobs={count} stagger={stagger}',flush=True)
    transformer._native_glm_enabled = native
    from exllamav3.util.tensor import buffered_aranges
    buffered_aranges.clear()
    from exllamav3.cache.recurrent_util import _slot_tensors
    from exllamav3.modules.attn import _block_tables
    _slot_tensors.clear(); _block_tables.clear()
    instrument=correctness or diagnostic
    if correctness:
        clear_kv(cache)
        if mtp:
            clear_kv(dc)
    gen = Generator(model, cache, tokenizer, max_batch_size=count,
                    draft_model=draft if mtp else None, draft_cache=dc if mtp else None,
                    num_draft_tokens=3 if mtp else 0, dynamic_draft_tokens=False,
                    ngram_match_min=0, record_draft_stats=True)
    def prompt_ids(i):
        if count==1 and not stagger and not prefix_tokens:
            prompt=(Path(__file__).resolve().parents[1]/'doc/native_glm53_prompt.txt').read_text(encoding='utf-8')
            return tokenizer.encode(prompt,encode_special_tokens=True)
        text=PROMPTS[i]+(' Please give precise details.'*80 if prefix_tokens else '')
        ids=tokenizer.encode(model.default_chat_prompt(text),encode_special_tokens=True)
        if prefix_tokens:
            assert ids.shape[1]>prefix_tokens
            ids=torch.cat((ids[:,:prefix_tokens-2],ids[:,-2:]),dim=1)
        return ids
    jobs = [Job(prompt_ids(i), max_new_tokens=n, min_new_tokens=n,
                stop_conditions=[], sampler=DefaultSampler() if stochastic else GreedySampler(),
                seed=1234+i, identifier=str(i)) for i,n in enumerate(lengths)]
    records, counts, shapes = [], Counter(), Counter()
    host_seconds=Counter()
    draft_valid={}
    rewinds = {j.identifier: [] for j in jobs}
    proposals = Counter()
    samples = {j.identifier: [] for j in jobs}
    endings, enqueued = {}, {}
    cycle, phase = 0, ''
    prev_active = None
    page_owners={}
    page_reassignments=0
    slot_owners={}; slot_reassignments=0; cancelled=set()
    last_shape=None
    phases={}

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
                value['draft_kv'] = kv(dc,j,min(s.kv_position,draft_valid.get(j.identifier,0)))
        return value

    def snapshot(kind, full=True):
        values={j.identifier:state(j,full) for j in gen.active_jobs if j.recurrent_state is not None}
        record(kind, jobs=values)
        return values

    def ownership():
        nonlocal page_reassignments,slot_reassignments
        live = [j for j in gen.active_jobs if j.recurrent_state is not None]
        slots = [j.recurrent_state.slot for j in live]
        assert len(slots) == len(set(slots)), 'Two live jobs share a recurrent slot'
        seen = set()
        for j in live:
            slot=j.recurrent_state.slot
            if slot in slot_owners and slot_owners[slot]!=j.identifier: slot_reassignments+=1
            slot_owners[slot]=j.identifier
            s = j.sequences[0]
            pages = [p.page_index for p in s.allocated_pages]
            assert not seen.intersection(pages), 'Independent prompts share mutable pages'
            seen.update(pages)
            for page in pages:
                previous=page_owners.get(page)
                if previous is not None and previous!=j.identifier:
                    page_reassignments+=1
                page_owners[page]=j.identifier
            assert s.block_index_tensor[0,:len(pages)].tolist() == pages
            assert j.recurrent_state.cache is cache
        # Allocator slot/page numbers may differ between runs; compare logical
        # ownership after asserting physical disjointness and table correctness.
        record('ownership', jobs=[j.identifier for j in live], unique_slots=len(set(slots)),
               page_counts=[len(j.sequences[0].allocated_pages) for j in live])

    for d in model.active_devices: torch.cuda.synchronize(d)
    start_epoch, start = time.time(), time.perf_counter()
    with ExitStack() as scope:
        roles = {}
        for role, mod in ([('target',model)] + ([('draft',draft)] if mtp else [])) if instrument else []:
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
            if diagnostic:
                for method in ('forward','prefill'):
                    timed_fn=getattr(mod,method)
                    def timed(*a,_f=timed_fn,_key=role+'/'+method,**kw):
                        begin=time.perf_counter()
                        try: return _f(*a,**kw)
                        finally: host_seconds[_key]+=time.perf_counter()-begin
                    scope.enter_context(patch.object(mod,method,timed))
            if correctness:
                for b in mod.modules:
                    prepare=b.prepare_for_device
                    def prepare_device(x,params,_f=prepare,_k=b.key):
                        src=str(x.device); before=digest(x)
                        y=_f(x,params)
                        if src!=str(y.device): record('transfer',key=_k,source=src,dest=str(y.device),before=before,after=digest(y))
                        return y
                    scope.enter_context(patch.object(b,'prepare_for_device',prepare_device))
                    if isinstance(b,TransformerBlock) and hasattr(b.mlp,'routing_fn'):
                        rf=b.mlp.routing_fn
                        def route(*a,_f=rf,_k=b.key,**kw):
                            ids,w=_f(*a,**kw); record('routing',key=_k,ids=digest(ids),weights=digest(w)); return ids,w
                        scope.enter_context(patch.object(b.mlp,'routing_fn',route))
                for method in ('forward','prefill'):
                    f = getattr(mod,method)
                    def forward(input_ids,params=None,_f=f,_r=role,_method=method):
                        ids=input_ids
                        # Model.prefill does not call Model.forward. Wrap both so
                        # new arrivals and accepted draft-cache repair are covered.
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
                        bindings=[]
                        if table is not None and seqlens is not None:
                            for row in range(table.shape[0]):
                                owner=next(j for j in gen.active_jobs if j.sequences[0].allocated_pages and j.sequences[0].allocated_pages[0].page_index==int(table[row,0]))
                                expected_pages=[p.page_index for p in owner.sequences[0].allocated_pages]
                                assert table[row,:len(expected_pages)].tolist()==expected_pages
                                if params.get('recurrent_states'):
                                    assert params['recurrent_states'][row] is owner.recurrent_state
                                bindings.append((owner,int(seqlens[row])))
                        if _r=='target':
                            for j,n in bindings:
                                if n==0:
                                    rs=j.recurrent_state
                                    assert rs.position==0 and rs.last_history==0 and j.mtp_last_hidden is None
                                    for layer in cache.get_all_recurrent_layers().values():
                                        c,r=layer.get_state_tensors()
                                        assert not torch.count_nonzero(c[rs.slot,:,:layer.module.conv_kernel_size]).item()
                                        assert not torch.count_nonzero(r[rs.slot,0]).item()
                                    record('fresh_state_zero',job=j.identifier)
                        record('before_model',role=_r,method=_method,jobs={j.identifier:state(j,False) for j,_ in bindings},lengths=[n for _,n in bindings],history=bool(params.get('recurrent_history')))
                        y = _f(ids,params)
                        if _r=='draft':
                            for j,n in bindings: draft_valid[j.identifier]=n+ids.shape[1]
                        if mtp or _method=='prefill':
                            record('after_model',role=_r,method=_method,jobs={j.identifier:dict(state=state(j,False),written_kv=kv(dc if _r=='draft' else cache,j,n+ids.shape[1])) for j,n in bindings})
                        assert before == {j.identifier:state(j) for j in others}, 'Forward/prefill changed another job state'
                        record(_method,role=_r,ids=ids.tolist(),rows=[digest(t) for t in y] if y is not None else None,
                               exports=[[digest(t) for t in e] for e in params.get('export_states',[])])
                        return y
                    scope.enter_context(patch.object(mod,method,forward))
        choose = native_glm.get_native_block
        def native_block(b,x,params):
            result=choose(b,x,params)
            if result is not None:
                counts[f'{roles[id(b)]}/b{x.shape[0]}/q{x.shape[1]}/native'] += 1
            return result
        if instrument: scope.enter_context(patch.object(native_glm,'get_native_block',native_block))
        if instrument:
            import sys
            from exllamav3.util.device_copy import to_device
            def move(t,device,*a,**kw):
                if device is not None and t.device!=torch.device(device):
                    counts[f'transfer/{t.device}->{torch.device(device)}/{tuple(t.shape)}/{t.dtype}']+=1
                if not diagnostic: return to_device(t,device,*a,**kw)
                begin=time.perf_counter()
                result=to_device(t,device,*a,**kw)
                if device is not None and t.device!=torch.device(device):
                    host_seconds[f'transfer/{t.device}->{torch.device(device)}']+=time.perf_counter()-begin
                return result
            for name,mod in list(sys.modules.items()):
                if name.startswith('exllamav3.') and getattr(mod,'to_device',None) is to_device:
                    scope.enter_context(patch.object(mod,'to_device',move))

        if correctness:
            head=model.modules[model.logit_layer_idx]; hf=head.forward
            def head_forward(x,params,*a,**kw):
                y=hf(x,params,*a,**kw); record('head',phase=phase,hidden=digest(x),logits=digest(y)); return y
            scope.enter_context(patch.object(head,'forward',head_forward))
        if mtp:
            propose = gen.iterate_draftmodel_mtp_gen
            def propose_tokens(results):
                begin=time.perf_counter() if diagnostic else 0.0
                ids=propose(results)
                if diagnostic: host_seconds['generator/draft']+=time.perf_counter()-begin
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
                    expected={}
                    for k,l in cache.get_all_recurrent_layers().items():
                        c,r=l.get_state_tensors(); h=rs.last_history; end=c.shape[-1]-n
                        expected[str(k)]=(digest(r[rs.slot,h+1-n if n else 0]),digest(c[rs.slot,:,end-l.module.conv_kernel_size:end]))
                    position=rs.position
                    rewind(rs,n)
                    assert rs.position==position-n and rs.last_history==0
                    for k,l in cache.get_all_recurrent_layers().items():
                        c,r=l.get_state_tensors()
                        assert expected[str(k)]==(digest(r[rs.slot,0]),digest(c[rs.slot,:,:l.module.conv_kernel_size]))
                    rewinds[owner.identifier].append(n)
                    record('rewind',job=owner.identifier,depth=n)
                    after=snapshot('after_rewind')
                    assert all(before[j.identifier]==after[j.identifier] for j in others), 'Rewind contaminated another job'
                scope.enter_context(patch.object(GDNState,'rewind',rewind_state))
        if correctness:
            for j in jobs:
                release=j.deallocate_pages
                def deallocate(_j=j,_f=release):
                    if _j.recurrent_state is not None:
                        record('final_state',job=_j.identifier,state=state(_j))
                    others=[o for o in gen.active_jobs if o is not _j and o.recurrent_state is not None]
                    before={o.identifier:state(o) for o in others}
                    result=_f()
                    assert before=={o.identifier:state(o) for o in others}, 'Release contaminated another job'
                    return result
                scope.enter_context(patch.object(j,'deallocate_pages',deallocate))
        verify=gen.iterate_gen
        def iteration(results,draft_tokens=None):
            nonlocal prev_active,last_shape
            live=active()
            b,q=len(live),1 if draft_tokens is None else draft_tokens.shape[1]+1
            shapes[f'b{b}/q{q}'] += 1
            last_shape=f'b{b}/q{q}'
            ids=[j.identifier for j in live]
            changed=ids!=prev_active
            if correctness:
                ownership()
            begin=time.perf_counter() if diagnostic else 0.0
            verify(results,draft_tokens)
            if diagnostic: host_seconds['generator/verify']+=time.perf_counter()-begin
            if correctness:
                full=True
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
                tick=time.perf_counter()
                prior_tokens=sum(j.new_tokens for j in jobs)
                results=gen.iterate()
                now=time.perf_counter()
                if last_shape is not None:
                    phase_data=phases.setdefault(last_shape,dict(tokens=0,seconds=0,iterations=0))
                    phase_data['tokens']+=sum(j.new_tokens for j in jobs)-prior_tokens
                    phase_data['seconds']+=now-tick
                    phase_data['iterations']+=1
                for r in results:
                    if r.get('stage')=='error':raise r['error']
                    jid=r.get('identifier')
                    if jid is not None and r.get('token_ids') is not None:
                        samples[jid].append((now,r['token_ids'].numel()))
                    if jid is not None and r.get('eos'):
                        endings[jid]=dict(result=r,wall=now)
                cycle+=1
                # MTP acceptance can change completion order. Trigger on the
                # actual shrink, not the identity of an assumed second finisher.
                if cancel and not cancelled and cycle>=8:
                    victim=jobs[0]
                    gen.cancel(victim);cancelled.add(victim.identifier)
                    endings[victim.identifier]=dict(wall=now,result={})
                    if not joined:
                        gen.enqueue(jobs[count]);enqueued[jobs[count].identifier]=time.perf_counter();joined=True
                finished=sum(j.is_finished for j in jobs[:count])
                if not joined and finished>=max(1,count-2) and join_after is None:
                    join_after=cycle+3  # Observe the smaller batch before growing it again.
                if not joined and join_after is not None and cycle>=join_after:
                    gen.enqueue(jobs[count]);enqueued[jobs[count].identifier]=time.perf_counter();joined=True
                if correctness and cycle%(8 if mtp else 16)==0:
                    print(f'parity native={native} mtp={mtp} jobs={count} stagger={stagger}: cycle {cycle}',flush=True)
        finally:
            gen.clear_queue()
            gen.filter_pool.shutdown(wait=True)
    for d in model.active_devices: torch.cuda.synchronize(d)
    elapsed=time.perf_counter()-start
    total=sum(v for k,v in counts.items() if k.endswith('/all'))
    used=sum(v for k,v in counts.items() if k.endswith('/native'))
    report=dict(native=native,mtp=mtp,jobs=count,stagger=stagger,force=force,stochastic=stochastic,
                instrumented_state_checks=correctness,
                start_epoch=start_epoch,end_epoch=time.time(),seconds=elapsed,
                aggregate_tps=sum(j.new_tokens for j in jobs)/elapsed,shapes=dict(shapes),
                calls=dict(counts,total=total,native=used,fallback=total-used),per_job={},
                page_reassignments=page_reassignments,slot_reassignments=slot_reassignments,phases=phases,diagnostic=diagnostic,
                diagnostic_host_seconds=dict(host_seconds),timing_semantics='inclusive host wall time; transfer calls may include waits for prior GPU work; not isolated copy/kernel time')
    for v in phases.values():v['effective_tps']=v['tokens']/max(v['seconds'],1e-9)
    if require_page_reuse:assert page_reassignments>0,'Test never reused a page from a completed job'
    join_time=enqueued.get(str(count)) if stagger else None
    report['join_time_seconds']=join_time-start if join_time is not None else None
    for j in jobs:
        jid=j.identifier
        assert jid in cancelled or (j.is_finished and j.new_tokens==lengths[int(jid)]), 'Job did not complete requested output'
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
        if mtp and jid not in cancelled:assert proposals[jid]==j.accepted_draft_tokens+j.rejected_draft_tokens
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
    if kw.get('prefix_tokens'):
        assert any(e['kind']=='committed' and len(e['jobs'])>=2 and all(v['position']>=256 for v in e['jobs'].values()) for e in b), 'Jobs did not cross the page boundary concurrently'
    if kw.get('stagger'): assert sb['slot_reassignments']>0, 'No recurrent slot reuse'
    if kw['mtp']:
        histories=[v['rewinds'] for v in sb['per_job'].values()]
        assert any(any(n>0 for n in h) for h in histories),'No rejection'
        assert len({tuple(h) for h in histories})>1,'Jobs had identical speculative histories'
        by_cycle={}
        for event in b:
            if event['kind']=='rewind': by_cycle.setdefault(event['cycle'],[]).append(event['depth'])
        divergent=sum(len(set(depths))>1 for depths in by_cycle.values())
        assert divergent>0,'No simultaneous window had different per-job rollback depths'
    transfers={k:(sa['calls'].get(k,0),sb['calls'].get(k,0)) for k in sa['calls'].keys() | sb['calls'].keys() if k.startswith('transfer/')}
    assert all(a==b for a,b in transfers.values()), {k:v for k,v in transfers.items() if v[0]!=v[1]}
    if not kw['mtp']:
        assert sb['calls'].get(f'target/b{kw["count"]}/q1/native',0)>0,'Concurrent fast path not exercised'
    else:
        assert sb['calls'].get(f'target/b{kw["count"]}/q4/native',0)>0,'Concurrent MTP target fast path not exercised'
        assert not any(k.startswith('draft/') and k.endswith('/native') for k in sb['calls'])
        assert sum(v['accepted'] for v in sb['per_job'].values())>0
        if kw.get('force'):
            assert {1,2,3}.issubset({n for v in sb['per_job'].values() for n in v['rewinds']})
    result=dict(exact=True,events=len(a),divergent_rewind_cycles=divergent if kw['mtp'] else 0,shapes=sb['shapes'],calls=sb['calls'],
                page_reassignments=sb['page_reassignments'],slot_reassignments=sb['slot_reassignments'],transfers_exact=True,
                per_job={j:{k:v[k] for k in ('tokens','completion_sha256','proposed','accepted','rejected','rewinds')}
                         for j,v in sb['per_job'].items()})
    import exllamav3_ext
    with Path(exllamav3_ext.__file__).open('rb') as extension_file:
        result['extension_sha256']=hashlib.file_digest(extension_file,'sha256').hexdigest()
    (output/'parity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS',output,flush=True)
    return result
