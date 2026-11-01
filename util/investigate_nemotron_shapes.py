"""Reproducible Nemotron shape/state correctness diagnostics; no runtime optimizations.

Run with the existing TextGen Python and CUDA_VISIBLE_DEVICES=2. Evidence is
written under --output; source/model files are never rewritten by this tool.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = [
    'Write a Python implementation of a bounded LRU cache with get and put methods. Explain how eviction works and include unit tests for updating a key and exceeding capacity.',
    'Implement a Python topological sort for a directed graph. Detect cycles and include tests for disconnected vertices and a cycle.',
    'Write a Python function that merges overlapping half-open intervals. Explain the edge cases and provide unit tests.',
    'Implement a small Python retry decorator with exponential backoff, injectable sleep, and tests for success and exhausted retries.',
]


def prompt(i):
    return '<|im_start|>user\n' + PROMPTS[i] + '\n<|im_end|>\n<|im_start|>assistant\n'


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--phase', choices=['matrix', 'fixed', 'state'], default='matrix')
    p.add_argument('--audit-refresh', action='store_true')
    p.add_argument('--tune-from', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    runtime = a.output / 'runtime'
    runtime.mkdir(exist_ok=True)
    if a.tune_from:
        if (runtime/'tune.bin').exists():
            raise ValueError('Use a fresh output directory when copying a tuning cache')
        shutil.copyfile(a.tune_from,runtime/'tune.bin')
    os.environ['EXLLAMAV3_TUNE_CACHE'] = str((runtime / 'tune.bin').resolve())
    os.environ['TRITON_CACHE_DIR'] = str((runtime / 'triton').resolve())
    for name in ('EXL3_NATIVE_PLAN', 'EXL3_NATIVE_QWEN_BLOCK', 'EXL3_NATIVE_GLM_BLOCK'):
        os.environ[name] = '0'
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.ext import exllamav3_ext as ext
    torch.set_grad_enabled(False)
    cfg = Config.from_directory(a.model)
    model = Model.from_config(cfg)
    draft = Model.from_config(cfg, component='mtp')
    cache = Cache(model, max_num_tokens=4096, max_batch_size=4, max_history=3)
    dcache = Cache(draft, max_num_tokens=4096, max_batch_size=4)
    tok = Tokenizer.from_config(cfg)
    save(a.output / 'environment.json', dict(
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), extension=ext.__file__,
        extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
        model=a.model, pattern=cfg.hybrid_override_pattern,
        environment={k:v for k,v in os.environ.items() if k.startswith(('EXL3_', 'CUDA_', 'EXLLAMAV3_', 'TRITON_'))}))
    draft.load(device=torch.device('cuda:0'), progressbar=False)
    model.load(device=torch.device('cuda:0'), progressbar=False)
    draft.attach_to(model)
    print('LOADED', flush=True)
    if a.phase in ('fixed', 'state'):
        with torch.inference_mode():
            fixed(a, model, draft, cache, dcache, tok)
    else:
        matrix(a, model, draft, cache, dcache, tok)


def matrix(a, model, draft, cache, dcache, tok):
    import torch
    from exllamav3 import Generator, Job, GreedySampler
    records = []
    for repeat in range(a.repeats):
        for independent in (False, True):
            for batch in (1, 3, 4):
                for mode in ('plain', 'mtp3', 'cap8'):
                    name = f'r{repeat}-{"independent" if independent else "identical"}-b{batch}-{mode}'
                    gen = Generator(model=model, cache=cache, tokenizer=tok,
                        draft_model=draft if mode != 'plain' else None,
                        draft_cache=dcache if mode != 'plain' else None,
                        num_draft_tokens=3, dynamic_draft_tokens=False, draft_max_rows=8 if mode == 'cap8' else 0,
                        max_batch_size=4, ngram_match_min=0, record_draft_stats=True)
                    prompt_ids = [tok.encode(prompt(i if independent else 0), encode_special_tokens=True) for i in range(batch)]
                    jobs = [Job(input_ids=ids, max_new_tokens=a.tokens, min_new_tokens=a.tokens,
                                sampler=GreedySampler(), seed=1234+i, stop_conditions=[]) for i,ids in enumerate(prompt_ids)]
                    rounds, shapes, dshapes, refreshes = [], Counter(), Counter(), []
                    current = {}
                    original, doriginal, poriginal, soriginal = model.forward, draft.forward, draft.prefill, draft.sample_from_state
                    def forward(input_ids, params=None):
                        shapes[str(tuple(input_ids.shape))] += 1
                        logits = original(input_ids, params)
                        vals, picks = logits[..., :model.config.vocab_size].float().topk(5, dim=-1)
                        current.update(ids=input_ids.tolist(), top_ids=picks.cpu().tolist(), top_logits=vals.cpu().tolist())
                        return logits
                    def dforward(input_ids, params=None):
                        dshapes[str(tuple(input_ids.shape))] += 1
                        return doriginal(input_ids, params)
                    def dprefill(input_ids, params=None):
                        record=dict(shape=list(input_ids.shape), ids=input_ids.tolist(),
                                    positions=params['cache_seqlens'].tolist(), before=current.get('before'),
                                    target_ids=current.get('ids'))
                        refreshes.append(record)
                        tensors=[t for l in dcache.layers.values() for t in l.get_tensors()]
                        before=[t.clone() for t in tensors] if a.audit_refresh else None
                        out=poriginal(input_ids, params)
                        if a.audit_refresh:
                            first=[t.clone() for t in tensors]
                            mask=torch.zeros(tensors[0].shape[:2],dtype=torch.bool,device=tensors[0].device)
                            for row,pos in enumerate(record['positions']):
                                for j in range(input_ids.shape[1]):
                                    at=pos+j
                                    mask[int(params['block_table'][row,at//256]),at%256]=True
                            record['outside_write_window_exact']=all(torch.equal(x[~mask],y[~mask]) for x,y in zip(before,first))
                            poriginal(input_ids,params)
                            record['repeat_kv_exact']=all(torch.equal(x,y) for x,y in zip(first,tensors))
                            assert record['outside_write_window_exact'] and record['repeat_kv_exact']
                        return out
                    def sample(state, params):
                        ids = soriginal(state, params)
                        current.setdefault('draft_ids', []).append(ids.cpu().tolist())
                        return ids
                    model.forward, draft.forward, draft.prefill, draft.sample_from_state = forward, dforward, dprefill, sample
                    for j in jobs: gen.enqueue(j)
                    try:
                        while gen.num_remaining_jobs():
                            current = dict(before=[j.new_tokens for j in jobs])
                            for event in gen.iterate():
                                if event['stage'] == 'error': raise event['error']
                            current['after'] = [j.new_tokens for j in jobs]
                            current['accepted'] = [j.accepted_draft_tokens for j in jobs]
                            current['rejected'] = [j.rejected_draft_tokens for j in jobs]
                            current['states'] = []
                            for j in gen.active_jobs:
                                if not j.is_prefill_done(): continue
                                s = j.recurrent_state.target
                                seq = j.sequences[0]
                                current['states'].append(dict(position=s.position, kv=seq.kv_position,
                                    sequence=len(seq.sequence_ids), history=s.last_history, slot=s.slot, parity=s.parity))
                                assert s.last_history == 0
                                assert s.position == seq.kv_position == len(seq.sequence_ids)-1
                            if 'ids' in current: rounds.append(current)
                        tokens = [j.sequences[0].sequence_ids.torch()[0, ids.shape[1]:].tolist() for j,ids in zip(jobs,prompt_ids)]
                        record = dict(name=name, repeat=repeat, independent=independent, batch=batch, mode=mode,
                                      prompt_ids=[ids.tolist()[0] for ids in prompt_ids], tokens=tokens,
                                      texts=[tok.decode(torch.tensor([t])) for t in tokens], shapes=dict(shapes),
                                      draft_shapes=dict(dshapes), refreshes=refreshes, rounds=rounds)
                        save(a.output / (name+'.json'), record)
                        records.append(record)
                        plain = next(r for r in records if r['repeat']==repeat and r['independent']==independent and r['batch']==batch and r['mode']=='plain')
                        print(name, 'first_diff', [first_diff(x,y) for x,y in zip(plain['tokens'], tokens)], 'shapes',dict(shapes), flush=True)
                    finally:
                        model.forward, draft.forward, draft.prefill, draft.sample_from_state = original, doriginal, poriginal, soriginal
                        gen.close()
    summaries=[]
    for r in records:
        plain=next(x for x in records if x['repeat']==r['repeat'] and x['independent']==r['independent'] and x['batch']==r['batch'] and x['mode']=='plain')
        cold=next(x for x in records if x['repeat']==0 and x['independent']==r['independent'] and x['batch']==r['batch'] and x['mode']==r['mode'])
        summaries.append(dict(name=r['name'], plain_first_diff=[first_diff(x,y) for x,y in zip(plain['tokens'],r['tokens'])],
                              repeat_first_diff=[first_diff(x,y) for x,y in zip(cold['tokens'],r['tokens'])]))
    save(a.output/'matrix-summary.json', summaries)


def fixed(a, model, draft, cache, dcache, tok):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    data = json.loads((a.output.parent/'matrix/r0-identical-b4-plain.json').read_text())
    sequence = torch.tensor([data['prompt_ids'][0] + data['tokens'][0]])
    plen = len(data['prompt_ids'][0])
    layers = cache.get_all_recurrent_layers()
    states = [cache.get_new_state() for _ in range(4)]
    blocks = [m for m in model.modules if hasattr(m, 'attn')]
    bt = torch.arange(16, dtype=torch.int32).view(4,4)

    def forward(ids, pos, history=False, row=0):
        b = ids.shape[0]
        return model.forward(ids, dict(attn_mode='flash_attn', cache=cache,
            block_table=bt[row:row+b].contiguous(), cache_seqlens=torch.full((b,),pos,dtype=torch.int32),
            recurrent_states=states[row:row+b], recurrent_history=history))

    def snapshot():
        tensors = [(t,t.clone()) for l in layers.values() for t in (l.conv_state,l.recurrent_state)]
        tensors += [(t,t.clone()) for l in cache.layers.values() for t in l.get_tensors()]
        metadata = [{k:getattr(s,k) for k in ('position','parity','last_history','spec_row','spec_shape')} for s in states]
        return tensors,metadata

    def restore(snap):
        for t,c in snap[0]: t.copy_(c)
        for s,meta in zip(states,snap[1]):
            for k,v in meta.items():setattr(s,k,v)

    active = False
    captured = {}
    saved = []
    def wrap(obj, name, label, routing=False):
        fn = getattr(obj,name); saved.append((obj,name,fn))
        def call(*args,**kwargs):
            if active and not routing:
                x = args[0]
                captured[label+':input'] = x.reshape(-1,x.shape[-1])[0].detach().clone()
            out = fn(*args,**kwargs)
            if active:
                if routing:
                    captured[label+':selection'] = out[0][0].detach().clone()
                    captured[label+':weights'] = out[1][0].detach().clone()
                else:
                    captured[label+':output'] = out.reshape(-1,out.shape[-1])[0].detach().clone()
            return out
        setattr(obj,name,call)
    for m in model.modules: wrap(m,'forward',m.key)
    for b in blocks:
        if b.mlp:
            wrap(b.mlp,'forward',b.mlp.key)
            wrap(b.mlp,'routing_fn',b.mlp.key+':route',True)
        if b.attn:wrap(b.attn,'forward',b.attn.key)

    def compare(x,y):
        xf,yf=x.float(),y.float()
        return dict(exact=torch.equal(x,y), max_abs=(xf-yf).abs().max().item(),
                    rfn=((xf-yf).norm()/yf.norm().clamp_min(1e-20)).item())

    all_results=[]
    try:
        for d in ((14,) if a.phase=='state' else (14,50)):
            for s in states:
                for l in layers.values():l.clear(s.slot)
                s.position=0;s.parity=0;s.last_history=0;s.spec_row=None;s.spec_shape=None
            for i in range(4):forward(sequence[:,:plen-1],0,row=i)
            for pos in range(plen-1,plen+d-1):forward(sequence[:,pos:pos+1].repeat(4,1),pos)
            pos=plen+d-1
            base=snapshot()
            if a.phase=='state':
                state_checks(a,cache,states,sequence,pos,forward,snapshot,restore,base,compare)
                continue
            results={}
            for repeat in range(2):
                for b,q in ((4,1),(1,1),(1,4),(2,4),(3,1),(3,2),(3,4),(4,2),(4,4)):
                    restore(base);captured={};active=True
                    logits=forward(sequence[:,pos:pos+q].repeat(b,1),pos,q>1).detach().clone()
                    active=False
                    key=f'd{d}-r{repeat}-b{b}-q{q}'
                    output=logits[0,0,:model.config.vocab_size]
                    refkey=f'd{d}-r0-b4-q1'
                    if key==refkey: ref={k:v.clone() for k,v in captured.items()};refout=output.clone()
                    vals,ids=output.float().topk(5)
                    diffs={k:compare(v,ref[k]) for k,v in captured.items() if k in ref}
                    first=next((k for k,v in diffs.items() if not v['exact'] and 'embeddings:input' not in k),None)
                    record=dict(case=key,logits=compare(output,refout),top_ids=ids.tolist(),top_logits=vals.tolist(),
                                argmax=output.argmax().item(),first_difference=first,components=diffs)
                    if repeat:
                        record['repeat_exact']=all(torch.equal(v,results[(b,q)][k]) for k,v in captured.items())
                    else:results[(b,q)]={k:v.clone() for k,v in captured.items()}
                    all_results.append(record)
                    save(a.output/'fixed-shapes.json',all_results)
                    print('FIXED',key,record['argmax'],record['logits'],first,flush=True)
            if d==14:
                m=blocks[0].attn
                x=ref[m.key+':input'].half().view(1,1,-1)
                projection=[]
                for rows in (1,2,3,4,6,8,12,16):
                    y=m.in_proj.forward(x.repeat(1,rows,1),{})[0,0].clone()
                    if rows==1:p_ref=y.clone()
                    projection.append(dict(rows=rows,input_projection=compare(y,p_ref)))
                save(a.output/'mamba-input-projection.json',projection)
                for block in (blocks[1],blocks[39],blocks[-1]):
                    if block.mlp is None:continue
                    m=block.mlp
                    x=ref[m.key+':input'].half().view(1,1,-1)
                    moe_component(a,m,x,compare)
            restore(base)
            if d==14: settlement(a,model,cache,states,sequence,pos,forward,snapshot,restore,base,compare)
            del base,results
    finally:
        for obj,name,fn in reversed(saved):setattr(obj,name,fn)
        for s in states:s.free()


def state_checks(a,cache,states,sequence,pos,forward,snapshot,restore,base,compare):
    import torch
    layers=cache.get_all_recurrent_layers()
    def committed(b):
        return [t.clone() for l in layers.values() for s in states[:b]
                for t in (l.recurrent_state[s.base_row()],l.conv_state[s.slot,:,:l.module.conv_kernel_size])]
    def kv():return [t for l in cache.layers.values() for t in l.get_tensors()]
    records=[]
    for b,q in ((1,4),(2,4),(3,2),(3,4),(4,2),(4,4)):
        restore(base)
        ids=sequence[:,pos:pos+q].repeat(b,1)
        spec_logits=forward(ids,pos,True).clone()
        for s in states[:b]:s.rewind(0)
        spec_state=committed(b);spec_kv=[t.clone() for t in kv()]
        restore(base)
        plain_logits=forward(ids,pos,False).clone()
        record=dict(batch=b,query=q,history_logits=compare(spec_logits,plain_logits),
                    history_state_exact=all(torch.equal(x,y) for x,y in zip(spec_state,committed(b))),
                    history_kv_exact=all(torch.equal(x,y) for x,y in zip(spec_kv,kv())))
        assert record['history_logits']['exact'] and record['history_state_exact'] and record['history_kv_exact']
        restore(base)
        forward(ids,pos,True)
        accepted=[1+i%q for i in range(b)]
        for i,n in enumerate(accepted):states[i].rewind(q-n)
        settled=snapshot()
        tail_checks=[]
        for i,n in enumerate(accepted):
            next_id=sequence[:,pos+n:pos+n+1]
            restore(settled)
            clean=forward(next_id,pos+n,row=i).clone()
            clean_state=committed(b)
            restore(settled)
            for t in kv():
                # Each independent row owns four contiguous 256-token pages.
                t.view(4,1024,*t.shape[2:])[i,pos+n:].fill_(7)
            poisoned=forward(next_id,pos+n,row=i).clone()
            exact=torch.equal(clean,poisoned) and all(torch.equal(x,y) for x,y in zip(clean_state,committed(b)))
            assert exact
            tail_checks.append(dict(row=i,accepted=n,unused_kv_tail_invariant=exact))
        record['tail_checks']=tail_checks
        records.append(record)
        save(a.output/'state-checks.json',records)
        print('STATE',b,q,'same-shape history exact; rejected KV tail ignored',flush=True)


def moe_component(a,m,x,compare):
    import torch
    results=[]
    outputs=[]
    original=m.routing_fn
    sel,w=original(1,m.routing_cfg,x.view(1,-1),{})
    sel,w=sel.clone(),w.clone()
    reference=None
    for fixed_routes in (False,True):
        if fixed_routes:
            m.routing_fn=lambda rows,cfg,y,params:(sel.repeat(rows,1),w.repeat(rows,1))
        try:
            for rows in (1,2,3,4,6,8,12,16):
                y=x.repeat(1,rows,1)
                s,rw=m.routing_fn(rows,m.routing_cfg,y.view(rows,-1),{})
                selection=s[0].clone();weights=rw[0].clone()
                out=m.forward(y,{}).clone()[0,0]
                outputs.append(out.clone())
                if reference is None:reference=out.clone()
                results.append(dict(rows=rows,fixed_routes=fixed_routes,output=compare(out,reference),
                                    selection=selection.tolist(),weights=weights.tolist(),
                                    selected_set_equal=set(selection.tolist())==set(sel[0].tolist())))
        finally:m.routing_fn=original
    # Dense reconstruction reference for the SAME quantized weights and fixed route.
    latent=m.latent_in.forward(x.view(1,-1),{})
    routed=torch.zeros((1,m.expert_size),device=x.device,dtype=torch.float)
    for e,weight in zip(sel[0].tolist(),w[0]):
        u=m.ups[e].forward(latent,{'reconstruct':True})
        activation=torch.square(torch.relu(u)).half()
        down=m.downs[e].forward(activation,{'reconstruct':True})
        routed.add_(down.float()*weight.float())
    dense=m.latent_out.forward(routed.half(),{}) + m.shared_experts.forward(x,{})[0]
    for record,out in zip(results,outputs):
        if record['fixed_routes']:record['dense_reference']=compare(out,dense[0])
    save(a.output/(m.key+'-moe.json'),dict(cases=results,dense_reference=compare(reference,dense[0])))
    print('MOE',m.key,'dense reference',compare(reference,dense[0]),flush=True)


def settlement(a,model,cache,states,sequence,pos,forward,snapshot,restore,base,compare):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    layers=cache.get_all_recurrent_layers()
    records=[]
    for b,q in ((1,4),(2,4),(3,2),(3,4),(4,2),(4,4)):
        restore(base)
        ids=sequence[:,pos:pos+q].repeat(b,1)
        logits=forward(ids,pos,True).clone()
        spec=snapshot()
        # Per-row different accepted lengths exercise divergent settlement in one batch.
        for first in range(1,q+1):
            restore(spec)
            accepted=[1+(first-1+i)%q for i in range(b)]
            expected={}
            for key,l in layers.items():
                m=l.module
                conv,dt,g=(l.spec_conv[:b*q].view(b,q,-1),l.spec_beta[:b*q].view(b,q,-1),l.spec_g[:b*q].view(b,q,-1))
                expected[key]=[]
                for i,n in enumerate(accepted):
                    s=states[i]
                    pool=l.recurrent_state[s.base_row():s.base_row()+1].clone()
                    out=torch.empty((1,n,m.num_v_heads,m.v_head_dim),dtype=torch.bfloat16,device=pool.device)
                    ext.cuda_recurrent_mamba2(conv[i:i+1,:n].contiguous(),g[i:i+1,:n].contiguous(),
                        dt[i:i+1,:n].contiguous(),m.d_skip_f,pool,out,m.num_k_heads,m.num_v_heads,
                        m.k_head_dim,m.v_head_dim,None,False,None)
                    end=l.conv_state.shape[-1]-(q-n)
                    expected[key].append((pool[0].clone(),l.conv_state[s.slot,:,end-m.conv_kernel_size:end].clone()))
            kv_before=[t.clone() for l in cache.layers.values() for t in l.get_tensors()]
            for i,n in enumerate(accepted):states[i].rewind(q-n)
            errors=[]
            for key,l in layers.items():
                for i,n in enumerate(accepted):
                    rs,cs=expected[key][i];s=states[i]
                    if not torch.equal(rs,l.recurrent_state[s.base_row()]):errors.append([str(key),i,'scan',compare(l.recurrent_state[s.base_row()],rs)])
                    if not torch.equal(cs,l.conv_state[s.slot,:,:l.module.conv_kernel_size]):errors.append([str(key),i,'conv'])
                    assert s.position==pos+n and s.last_history==0
            kv_exact=all(torch.equal(old,t) for old,t in zip(kv_before,[t for l in cache.layers.values() for t in l.get_tensors()]))
            record=dict(batch=b,query=q,accepted=accepted,errors=errors,kv_unchanged=kv_exact)
            records.append(record)
            save(a.output/'settlement.json',records)
            print('SETTLE',b,q,accepted,'errors',len(errors),'kv',kv_exact,flush=True)
            assert not errors and kv_exact
        del spec


if __name__ == '__main__':
    main()
