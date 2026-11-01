"""Paired Nemotron validation and alternating decode benchmark for native MoE.

Uses the investigation's prompts and workload, without modifying its artifacts.
Choose --phase validate before --phase bench. Select the isolated extension via
PYTHONPATH and physical device with CUDA_VISIBLE_DEVICES=2.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from investigate_nemotron_shapes import prompt


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tune-from', type=Path, required=True)
    p.add_argument('--phase', choices=['validate', 'bench', 'trace', 'host'], required=True)
    p.add_argument('--contexts', default='0')
    p.add_argument('--batches', default='3,4')
    p.add_argument('--rounds', type=int, default=32)
    p.add_argument('--warmup', type=int, default=12)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--reload', action='store_true', help='Validate real unload/reload and arena release')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rt = a.output/'runtime'; rt.mkdir()
    shutil.copyfile(a.tune_from, rt/'tune.bin')
    shutil.copyfile(__file__, a.output/'source.py')
    os.environ['EXLLAMAV3_TUNE_CACHE'] = str((rt/'tune.bin').resolve())
    os.environ['TRITON_CACHE_DIR'] = str((rt/'triton').resolve())
    for flag in ('EXL3_NATIVE_PLAN', 'EXL3_NATIVE_QWEN_BLOCK', 'EXL3_NATIVE_GLM_BLOCK'):
        os.environ[flag] = '0'
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules import native_moe
    assert getattr(ext, 'native_moe_assignment_abi', None) == 1, ext.__file__
    torch.set_grad_enabled(False)
    contexts = [int(x) for x in a.contexts.split(',')]
    batches = [int(x) for x in a.batches.split(',')]
    cfg = Config.from_directory(a.model)
    model = Model.from_config(cfg); draft = Model.from_config(cfg, component='mtp')
    capacity = 4 * ((max(contexts)+4096+255)//256*256)
    cache = Cache(model, max_num_tokens=capacity, max_batch_size=4, max_history=3)
    dc = Cache(draft, max_num_tokens=capacity, max_batch_size=4)
    tok = Tokenizer.from_config(cfg)
    draft.load(device=torch.device('cuda:0'), progressbar=False)
    model.load(device=torch.device('cuda:0'), progressbar=False); draft.attach_to(model)
    moes = [m.mlp for m in model.modules if getattr(m, 'mlp', None) is not None]
    save(a.output/'environment.json', dict(
        branch=subprocess.check_output(['git','branch','--show-current'], text=True).strip(),
        head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        python=sys.version, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        extension=ext.__file__, extension_sha256=sha(ext.__file__), tune_input=sha(a.tune_from),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        cache_capacity=capacity, loaded_allocated=torch.cuda.memory_allocated(),
        moe=[dict(key=m.key, arch=m.config.arch_string, activation=m.activation_fn_idx,
                  K=[m.multi_up.K,m.multi_down.K], mul1=[m.multi_up.mul1,m.multi_down.mul1],
                  mcg=[m.multi_up.mcg,m.multi_down.mcg], I=m.intermediate_size_padded,
                  router=m.routing_fn.__name__, routing_first=m.routing_first,
                  routing_device=m.routing_device) for m in moes]))
    print('LOADED', flush=True)

    def ids_for(context, i):
        if not context: return tok.encode(prompt(i), encode_special_tokens=True)
        head = tok.encode('<|im_start|>user\nRepository context for agent '+str(i)+':\n```python\n', encode_special_tokens=True)
        body = tok.encode('\n'.join(f'def helper_{i}_{j}(items):\n    """Return stable unique values for task {i}."""\n    return list(dict.fromkeys(items))\n' for j in range(500)))
        tail = tok.encode('\n```\nUsing this context, '+prompt(i).split('user\n')[1], encode_special_tokens=True)
        n = context-head.shape[1]-tail.shape[1]
        return torch.cat((head, body.repeat(1,(n+body.shape[1]-1)//body.shape[1])[:,:n], tail), dim=1)

    if a.phase == 'validate':
        checks = []
        g = torch.Generator(device='cuda').manual_seed(919)
        for rows in (12,16,8,9,16,12):
            for m in moes:
                x = torch.randn((1,rows,m.hidden_size), dtype=torch.half, device='cuda', generator=g)*.05
                native_moe.enabled = False; stock = m.forward(x, {}).clone()
                native_moe.enabled = True; candidate = m.forward(x, {}).clone()
                assert torch.equal(stock, candidate), (m.key, rows, (stock-candidate).abs().max().item())
                if rows in (12,16): assert m.__dict__.get('_native_moe_stages'), (m.key, 'not dispatched')
                checks.append(dict(key=m.key, rows=rows, exact=True))
        save(a.output/'component-parity.json', checks)
        print('COMPONENT PARITY', len(checks), flush=True)

        # Compare the complete fixed-shape target forward and cache/state writes
        # from one restored prefix, including speculative settlement. Raw byte
        # equality also handles untouched uninitialized/NaN cache storage.
        state_handles = [cache.get_new_state() for _ in range(4)]
        layers = cache.get_all_recurrent_layers()
        bt = torch.arange(16,dtype=torch.int32).view(4,4)
        prefix = tok.encode(prompt(0),encode_special_tokens=True)
        def fwd(ids, pos, history=False, row=0):
            b = ids.shape[0]
            return model.forward(ids,dict(attn_mode='flash_attn',cache=cache,
                block_table=bt[row:row+b].contiguous(),cache_seqlens=torch.full((b,),pos,dtype=torch.int32),
                recurrent_states=state_handles[row:row+b],recurrent_history=history))
        def snapshot():
            tensors = [t for l in layers.values() for t in (l.conv_state,l.recurrent_state)]
            tensors += [t for l in cache.layers.values() for t in l.get_tensors()]
            return [(t,t.clone()) for t in tensors], [{k:getattr(s,k) for k in
                ('position','parity','last_history','spec_row','spec_shape')} for s in state_handles]
        def restore(snap):
            for t,c in snap[0]: t.copy_(c)
            for s,meta in zip(state_handles,snap[1]):
                for k,v in meta.items(): setattr(s,k,v)
        state_results=[]
        try:
            for i,s in enumerate(state_handles):
                for l in layers.values(): l.clear(s.slot)
                native_moe.enabled=False; fwd(prefix,0,row=i)
            base=snapshot();pos=prefix.shape[1]
            for b,q in ((3,4),(4,4),(2,4),(3,3),(4,4),(3,4)):
                outcomes=[]
                for ir in (False,True):
                    restore(base);native_moe.enabled=ir
                    output=fwd(prefix[:,-q:].repeat(b,1),pos,True).clone()
                    for i,s in enumerate(state_handles[:b]): s.rewind(q-(1+i%q))
                    outcomes.append((output,snapshot()))
                assert torch.equal(outcomes[0][0],outcomes[1][0])
                for (_,left),(_,right) in zip(outcomes[0][1][0],outcomes[1][1][0]):
                    assert torch.equal(left.view(torch.uint8),right.view(torch.uint8)), (b,q,'cache/state')
                assert outcomes[0][1][1]==outcomes[1][1][1]
                state_results.append(dict(batch=b,query=q,logits_exact=True,state_cache_exact=True,settlement_exact=True))
                del outcomes
            save(a.output/'state-cache-parity.json',state_results)
            del base
        finally:
            for s in state_handles:s.free()
        print('STATE/CACHE PARITY',len(state_results),flush=True)

    def cohort(context, batch, independent, ir, repeat, mode='mtp3', audit=False, trace=False):
        native_moe.enabled = ir
        name = f'c{context}-b{batch}-{"ind" if independent else "same"}-{mode}-{"ir" if ir else "stock"}-r{repeat}'
        gen = Generator(model=model, cache=cache, tokenizer=tok,
            draft_model=draft if mode!='plain' else None, draft_cache=dc if mode!='plain' else None,
            num_draft_tokens=3, dynamic_draft_tokens=False, draft_max_rows=8 if mode=='cap8' else 0,
            max_batch_size=4, max_chunk_size=2048, ngram_match_min=0, record_draft_stats=True)
        ids = [ids_for(context, i if independent else 0) for i in range(batch)]
        limit = a.tokens if audit else 2048
        jobs = [Job(input_ids=x, min_new_tokens=limit, max_new_tokens=limit, sampler=GreedySampler(),
                    seed=1234+i, stop_conditions=[]) for i,x in enumerate(ids)]
        counts, shapes, rounds, refreshes, states, logits = Counter(), Counter(), [], [], [], []
        host = a.phase=='host'
        host_spans=[]
        host_state=dict(in_moe=False,prep=None)
        saved = []
        def wrap(obj, name, fn): saved.append((obj,name,getattr(obj,name))); setattr(obj,name,fn)
        if audit or trace or host:
            original_run = native_moe.run
            def run(*args, **kw):
                if host and host_state['in_moe']: host_state['prep']=time.perf_counter_ns()
                with torch.profiler.record_function('IRMOE:stage') if trace else nullcontext():
                    out = original_run(*args, **kw)
                if out is not None: counts[str(args[1].shape[0])] += 1
                return out
            wrap(native_moe,'run',run)
            orig_forward = model.forward
            def forward(input_ids, params=None):
                shapes[str(tuple(input_ids.shape))] += 1
                t=time.perf_counter_ns() if host else None
                with torch.profiler.record_function('IRMOE:target') if trace else nullcontext():
                    out = orig_forward(input_ids, params)
                if host: host_spans.append(['target',t,time.perf_counter_ns()])
                if audit:
                    logits.append(hashlib.sha256(out.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
                return out
            wrap(model,'forward',forward)
            orig_refresh = draft.prefill
            def refresh(input_ids, params=None):
                tensors = [t for l in dc.layers.values() for t in l.get_tensors()] if audit and not refreshes else []
                before = [t.clone() for t in tensors]
                with torch.profiler.record_function('IRMOE:refresh') if trace else nullcontext():
                    out = orig_refresh(input_ids, params)
                if audit:
                    record=dict(ids=input_ids.tolist(), positions=params['cache_seqlens'].tolist())
                    if tensors:
                        first=[t.clone() for t in tensors]
                        mask=torch.zeros(tensors[0].shape[:2],dtype=torch.bool,device=tensors[0].device)
                        for row,pos in enumerate(record['positions']):
                            for j in range(input_ids.shape[1]):
                                at=pos+j;mask[int(params['block_table'][row,at//256]),at%256]=True
                        record['outside_refresh_window_exact']=all(torch.equal(x[~mask].view(torch.uint8),y[~mask].view(torch.uint8)) for x,y in zip(before,first))
                        orig_refresh(input_ids,params)
                        record['refresh_repeat_exact']=all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(first,tensors))
                        assert record['outside_refresh_window_exact'] and record['refresh_repeat_exact']
                    refreshes.append(record)
                return out
            wrap(draft,'prefill',refresh)
            if trace or host:
                orig_draft = draft.forward
                def draft_forward(*args,**kw):
                    t=time.perf_counter_ns() if host else None
                    with torch.profiler.record_function('IRMOE:draft') if trace else nullcontext(): out=orig_draft(*args,**kw)
                    if host: host_spans.append(['draft',t,time.perf_counter_ns()])
                    return out
                wrap(draft,'forward',draft_forward)
                def annotated(obj, method, label):
                    original = getattr(obj,method)
                    def call(*args,**kw):
                        t=time.perf_counter_ns() if host else None
                        if host and label=='moe':host_state.update(in_moe=True,prep=None)
                        if host and host_state['in_moe']:
                            if label=='sort':host_state['prep']=t
                            if label=='experts' and host_state['prep'] is not None:
                                host_spans.append(['preparation',host_state['prep'],t])
                        with torch.profiler.record_function('IRMOE:'+label) if trace else nullcontext():
                            out=original(*args,**kw)
                        if host:
                            host_spans.append([label,t,time.perf_counter_ns()])
                            if label=='moe':host_state['in_moe']=False
                        return out
                    wrap(obj,method,call)
                for module in moes:
                    annotated(module,'forward','moe')
                    annotated(module.shared_experts,'forward','shared')
                    annotated(module.latent_in,'forward','latent_in')
                    annotated(module.latent_out,'forward','latent_out')
                annotated(ext,'exl3_moe','experts')
                annotated(ext,'exl3_moe_gather','gather')
                annotated(ext,'native_moe_assign','assignment')
                annotated(torch.Tensor,'tolist','readback')
                if host:
                    annotated(torch.Tensor,'argsort','sort')
                    annotated(torch,'bincount','count')
        for j in jobs: gen.enqueue(j)
        def step():
            for event in gen.iterate():
                if event['stage']=='error': raise event['error']
            if audit:
                ss = []
                for j in gen.active_jobs:
                    if not j.is_prefill_done(): continue
                    s, seq = j.recurrent_state.target, j.sequences[0]
                    assert s.last_history == 0
                    assert s.position == seq.kv_position == len(seq.sequence_ids)-1
                    ss.append([s.position,seq.kv_position,len(seq.sequence_ids),s.last_history])
                states.append(ss)
        try:
            pre = time.perf_counter()
            while not all(j.is_prefill_done() for j in jobs): step()
            prefill = time.perf_counter()-pre
            for _ in range(0 if audit else a.warmup): step()
            torch.cuda.synchronize()
            begin = [j.new_tokens for j in jobs]
            ac = [j.accepted_draft_tokens for j in jobs]; re = [j.rejected_draft_tokens for j in jobs]
            allocations = torch.cuda.memory_allocated()
            host_spans.clear()
            torch.cuda.reset_peak_memory_stats()
            activities = [torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
            profile = torch.profiler.profile(activities=activities) if trace else nullcontext()
            with profile as prof:
                start = time.perf_counter()
                while gen.num_remaining_jobs() and (audit or len(rounds)<a.rounds):
                    before = [j.new_tokens for j in jobs]; t = time.perf_counter()
                    with torch.profiler.record_function('IRMOE:round') if trace else nullcontext(): step()
                    end = time.perf_counter()
                    rounds.append(dict(seconds=end-t, at=end-start, before=before,
                                       after=[j.new_tokens for j in jobs]))
                torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            if trace: prof.export_chrome_trace(str(a.output/(name+'.trace.json')))
            tokens = [j.sequences[0].sequence_ids.torch()[0,x.shape[1]:].tolist() for j,x in zip(jobs,ids)]
            record = dict(name=name, context=context, batch=batch, independent=independent, ir=ir, repeat=repeat,
                mode=mode, audit=audit, trace=trace, host=host, prompt_tokens=[x.shape[1] for x in ids],
                prefill_seconds=prefill, seconds=elapsed, begin=begin, end=[j.new_tokens for j in jobs],
                tokens=sum(j.new_tokens-n for j,n in zip(jobs,begin)), token_ids=tokens,
                completion_hashes=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens],
                accepted=[j.accepted_draft_tokens-n for j,n in zip(jobs,ac)],
                rejected=[j.rejected_draft_tokens-n for j,n in zip(jobs,re)],
                rounds=rounds, states=states, logits_hashes=logits, refreshes=refreshes,
                stage_calls=dict(counts), target_shapes=dict(shapes), tune_sha256=sha(rt/'tune.bin'),
                host_spans=host_spans,
                allocated=allocations, peak_allocated=torch.cuda.max_memory_allocated())
            save(a.output/(name+'.json'), record)
            print(name, record['tokens'], round(elapsed,3), round(record['tokens']/elapsed,2), 'calls',dict(counts), flush=True)
            return record
        finally:
            for obj,n,fn in reversed(saved): setattr(obj,n,fn)
            gen.close()

    results = []
    def pair(context, batch, independent, repeat, mode='mtp3', audit=False, trace=False):
        pair_results = [cohort(context,batch,independent,ir,repeat,mode,audit,trace)
                        for ir in ((False,True) if repeat%2==0 else (True,False))]
        left,right = pair_results
        for key in ('token_ids','begin','end','accepted','rejected'):
            assert left[key] == right[key], (left['name'],right['name'],key)
        if audit:
            for key in ('logits_hashes','states','refreshes','target_shapes'):
                assert left[key] == right[key], (left['name'],right['name'],key)
            cand = next(r for r in pair_results if r['ir'])
            if batch>1 and mode=='mtp3': assert cand['stage_calls'], cand['name']
            else: assert not cand['stage_calls'], cand['name']
        assert left['tune_sha256']==right['tune_sha256'], 'tuning changed within measured pair'
        results.extend(pair_results); save(a.output/'summary.json', results)
    if a.phase=='validate':
        for r in range(a.repeats):
            for independent in (False,True):
                for b in (1,3,4): pair(0,b,independent,r,audit=True)
            for mode in ('plain','cap8'): pair(0,4,True,r,mode,audit=True)
        if a.reload:
            import gc
            import weakref
            refs=[weakref.ref(w) for w in native_moe._arenas.values()]
            draft.unload();model.unload();gc.collect()
            assert all(ref() is None for ref in refs), 'arena retained across unload'
            draft.load(device=torch.device('cuda:0'),progressbar=False)
            model.load(device=torch.device('cuda:0'),progressbar=False);draft.attach_to(model)
            pair(0,3,True,99,audit=True)
            for current in results[-2:]:
                before=next(r for r in results if r['repeat']==0 and r['batch']==3 and r['independent'] and r['ir']==current['ir'])
                for key in ('token_ids','logits_hashes','accepted','rejected'):
                    assert before[key]==current[key], ('reload',key)
            save(a.output/'reload.json',dict(arenas_released=True,paired_generation_exact=True,reload_repeat_exact=True))
    else:
        for context in contexts:
            for b in batches:
                if a.phase=='bench':
                    pair(context,b,True,-2)  # unreported priming pair warms both implementations
                    for r in range(a.repeats): pair(context,b,True,r)
                else: pair(context,b,True,0,trace=a.phase=='trace')
    save(a.output/'workspace.json', [dict(key=str(key), bytes=sum(t.numel()*t.element_size() for t in
        [w.counts,w.tables,w.tokens,w.weights,w.inverse,w.output,w.scratch,*w.buffers]))
        for key,w in native_moe._arenas.items()])
    shutil.copyfile(rt/'tune.bin',a.output/'final-tune.bin')


if __name__=='__main__':
    import torch
    with torch.inference_mode(): main()
