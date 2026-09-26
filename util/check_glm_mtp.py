"""GLM MTP byte-exact event/state comparison through real rejection and repair."""
from collections import Counter
from contextlib import ExitStack
import json
import sys
from unittest.mock import patch
import torch
from util.check_glm_native import digest


class NativeCounts:
    def __init__(self,model,draft):
        self.models=[('target',model),('draft',draft)]
        self.counts=Counter(); self.scope=ExitStack()
    def __enter__(self):
        from exllamav3.modules import TransformerBlock,native_glm
        roles={}
        for role,mod in self.models:
            for b in mod.modules:
                if not isinstance(b,TransformerBlock): continue
                roles[id(b)]=role
                original=b.forward
                def forward(x,params,*a,_f=original,_r=role,**kw):
                    self.counts[f'{_r}/q{x.shape[1]}/all']+=1
                    return _f(x,params,*a,**kw)
                self.scope.enter_context(patch.object(b,'forward',forward))
        original=native_glm.get_native_block
        def choose(b,x,params):
            result=original(b,x,params)
            if result is not None: self.counts[f'{roles[id(b)]}/q{x.shape[1]}/native']+=1
            return result
        self.scope.enter_context(patch.object(native_glm,'get_native_block',choose))
        from exllamav3.util.device_copy import to_device
        def move(t,device,*a,**kw):
            if device is not None and t.device!=torch.device(device):
                self.counts[f'transfer/{t.device}->{torch.device(device)}/{tuple(t.shape)}/{t.dtype}']+=1
            return to_device(t,device,*a,**kw)
        for name,mod in list(sys.modules.items()):
            if name.startswith('exllamav3.') and getattr(mod,'to_device',None) is to_device:
                self.scope.enter_context(patch.object(mod,'to_device',move))
        return self
    def __exit__(self,*exc): self.scope.__exit__(*exc)
    def report(self): return dict(self.counts)


def recurrent(state):
    if state is None: return None
    out={}
    for key,layer in state.cache.get_all_recurrent_layers().items():
        conv,rec=layer.get_state_tensors()
        h=state.last_history
        out[str(key)]=dict(device=str(conv.device),
            conv=digest(conv[state.slot,:,:layer.module.conv_kernel_size+h]),
            recurrent=digest(rec[state.slot,:h+1]))
    return dict(position=state.position,history=state.last_history,layers=out)


def kv(cache,job,valid):
    pages=job.sequences[0].block_index_tensor[0,:len(job.sequences[0].allocated_pages)].long()
    out={}
    for key,layer in cache.layers.items():
        planes={}
        for name in ('k','v','k_idx','k_pool'):
            t=getattr(layer,name)
            n=valid//layer.kpool if name=='k_pool' else valid
            planes[name]=digest(t.index_select(0,pages.to(t.device)).flatten(0,1)[:n])
        out[str(key)]=dict(device=str(layer.device),valid=valid,planes=planes)
    return out


@torch.inference_mode()
def check_mtp(model,draft,cache,draft_cache,tokenizer,prompt,make_generator,output,tokens=128,force_rejections=False):
    from exllamav3 import Job,GreedySampler
    from exllamav3.modules import TransformerBlock,transformer
    from exllamav3.modules.gated_delta_net import GDNState
    flag=transformer._native_glm_enabled
    output.mkdir(parents=True,exist_ok=False)
    def run(enabled):
        transformer._native_glm_enabled=enabled
        # Equal cold index-cache state: otherwise the stock-first run alone
        # uploads buffered CPU aranges, obscuring transfer-count parity.
        from exllamav3.util.tensor import buffered_aranges
        buffered_aranges.clear()
        for c in (cache,draft_cache):
            for layer in c.layers.values():
                for t in layer.get_tensors(): t.zero_()
        gen=make_generator()
        job=Job(tokenizer.encode(prompt,encode_special_tokens=True),max_new_tokens=tokens,
                min_new_tokens=tokens,sampler=GreedySampler(),seed=1234,stop_conditions=[])
        records=[]; round_idx=0; phase=''; highwater={'target':0,'draft':0}
        rewinds=Counter(); forced=0; restored=0
        def record(kind,**data): records.append(dict(kind=kind,round=round_idx,**data))
        def snapshot(kind,transient=False):
            seq=job.sequences[0]
            pos=seq.kv_position
            record(kind,recurrent=recurrent(job.recurrent_state),
                target_kv=kv(cache,job,highwater['target'] if transient else pos),
                draft_kv=kv(draft_cache,job,highwater['draft'] if transient else min(pos,highwater['draft'])),
                position=pos,page_positions=[p.kv_position for p in seq.allocated_pages],
                ids=seq.sequence_ids.torch().tolist(),accepted=job.accepted_draft_tokens,rejected=job.rejected_draft_tokens,
                carry=digest(job.mtp_last_hidden) if job.mtp_last_hidden is not None else None)
        with NativeCounts(model,draft) as counts,ExitStack() as scope:
            for role,mod in [('target',model),('draft',draft)]:
                for name in ('forward','prefill'):
                    original=getattr(mod,name)
                    def call(input_ids,params=None,_f=original,_r=role,_n=name):
                        repair=_r=='draft' and _n=='prefill' and highwater['target']>0
                        if repair: snapshot('before_draft_repair')
                        y=_f(input_ids,params)
                        lengths=params.get('cache_seqlens')
                        past=int(lengths[0]) if lengths is not None else params.get('past_len',0)
                        highwater[_r]=past+input_ids.shape[1]
                        record('model',role=_r,method=_n,ids=input_ids.tolist(),result=digest(y) if torch.is_tensor(y) else None,
                            history=bool(params.get('recurrent_history')),past=past,
                            exports=[digest(t) for t in params.get('export_states',[])],
                            input_hidden=digest(params['target_hidden']) if 'target_hidden' in params else None)
                        if repair: snapshot('after_draft_repair')
                        return y
                    scope.enter_context(patch.object(mod,name,call))
                for b in mod.modules:
                    original=b.prepare_for_device
                    def prepare(x,params,_f=original,_k=b.key):
                        src=str(x.device); before=digest(x) if x.is_cuda else None
                        y=_f(x,params)
                        if src!=str(y.device): record('transfer',key=_k,source=src,dest=str(y.device),before=before,after=digest(y))
                        return y
                    scope.enter_context(patch.object(b,'prepare_for_device',prepare))
                    if isinstance(b,TransformerBlock):
                        original=b.forward
                        def block(x,params,*a,_f=original,_k=b.key,**kw):
                            y=_f(x,params,*a,**kw); record('block',key=_k,shape=list(x.shape),residual=digest(y)); return y
                        scope.enter_context(patch.object(b,'forward',block))
                        if hasattr(b.mlp,'routing_fn'):
                            original=b.mlp.routing_fn
                            def route(*a,_f=original,_k=b.key,**kw):
                                ids,weights=_f(*a,**kw)
                                record('routing',key=_k,ids=digest(ids),weights=digest(weights))
                                return ids,weights
                            scope.enter_context(patch.object(b.mlp,'routing_fn',route))
            original=draft.sample_from_state
            def sample(state,params,_f=original):
                nonlocal phase
                phase='draft'; ids=_f(state,params); phase=''
                record('draft_ids',ids=ids.cpu().tolist(),state=digest(state)); return ids
            scope.enter_context(patch.object(draft,'sample_from_state',sample))
            lm=model.modules[-1]; original=lm.forward
            def head(x,params,*a,**kw):
                y=original(x,params,*a,**kw)
                record('head',role=phase or 'target',hidden=digest(x),logits=digest(y)); return y
            scope.enter_context(patch.object(lm,'forward',head))
            original_propose=gen.iterate_draftmodel_mtp_gen
            def propose(results):
                nonlocal forced
                ids=original_propose(results)
                if ids is not None:
                    record('original_proposals',ids=ids.tolist())
                    if force_rejections:
                        col=round_idx%3
                        ids[0,col]=(ids[0,col]+1)%model.config.vocab_size
                        forced+=1; record('mutated_proposals',column=col,ids=ids.tolist())
                return ids
            scope.enter_context(patch.object(gen,'iterate_draftmodel_mtp_gen',propose))
            original_rewind=GDNState.rewind
            def rewind(state,n):
                nonlocal restored
                if state is not job.recurrent_state: return original_rewind(state,n)
                snapshot('before_rewind',True); rewinds[n]+=1
                # Independent byte-level oracle for the stock history selection, not
                # just agreement of two calls into the same rewind implementation.
                expected={}
                for k,l in cache.get_all_recurrent_layers().items():
                    c,r=l.get_state_tensors(); h=state.last_history
                    ri=h+1-n if n else 0
                    end=c.shape[-1]-n
                    expected[str(k)]=(digest(r[state.slot,ri]),digest(c[state.slot,:,end-l.module.conv_kernel_size:end]))
                previous=state.position
                original_rewind(state,n)
                assert state.position==previous-n and state.last_history==0
                for k,l in cache.get_all_recurrent_layers().items():
                    c,r=l.get_state_tensors()
                    assert expected[str(k)]==(digest(r[state.slot,0]),digest(c[state.slot,:,:l.module.conv_kernel_size])), (k,n)
                restored+=1; snapshot('after_rewind')
            scope.enter_context(patch.object(GDNState,'rewind',rewind))
            original_release=job.deallocate_pages
            def release():
                if job.recurrent_state is not None: snapshot('final_commit')
                return original_release()
            scope.enter_context(patch.object(job,'deallocate_pages',release))
            original_verify=gen.iterate_gen
            def verify(results,draft_tokens=None):
                nonlocal round_idx
                original_verify(results,draft_tokens)
                if job.recurrent_state is not None:
                    assert job.recurrent_state.position==job.sequences[0].kv_position
                    snapshot('committed')
                round_idx+=1
                if round_idx%16==0: print(f'GLM parity native={enabled} forced={force_rejections} round={round_idx}',flush=True)
            scope.enter_context(patch.object(gen,'iterate_gen',verify))
            try:
                gen.enqueue(job)
                while gen.num_remaining_jobs():
                    for result in gen.iterate():
                        if result.get('stage')=='error': raise result['error']
            finally:
                gen.clear_queue(); gen.filter_pool.shutdown(wait=True)
        assert job.accepted_draft_tokens>0 and job.rejected_draft_tokens>0
        if force_rejections:
            assert all(rewinds[n]>0 for n in (1,2,3)), f'Missing rejection depth: {rewinds}'
        else:
            assert rewinds[0]>0, 'Natural parity must cover a fully accepted window'
        (output/('native.json' if enabled else 'stock.json')).write_text(json.dumps(records))
        return records,dict(accepted=job.accepted_draft_tokens,rejected=job.rejected_draft_tokens,
            acceptance=job.accepted_draft_tokens/(job.accepted_draft_tokens+job.rejected_draft_tokens),
            rewinds=dict(rewinds),restoration_checks=restored,forced_windows=forced,
            calls=counts.report(),draft_stats=job.draft_stats)
    try:
        stock,ss=run(False); native,ns=run(True)
        (output/'stock-stats.json').write_text(json.dumps(ss,indent=2))
        (output/'native-stats.json').write_text(json.dumps(ns,indent=2))
        assert len(stock)==len(native),(len(stock),len(native))
        for i,(a,b) in enumerate(zip(stock,native)):
            if a!=b: raise AssertionError(f'GLM MTP parity event {i}, {a["kind"]}, round {a["round"]}: {[k for k in a if a[k]!=b.get(k)]}')
        assert ns['calls'].get('target/q4/native')==ns['calls'].get('target/q4/all')>0
        assert not any(k.startswith('draft/') and k.endswith('/native') for k in ns['calls'])
        transfers={k:(ss['calls'].get(k,0),ns['calls'].get(k,0)) for k in ss['calls'].keys() | ns['calls'].keys() if k.startswith('transfer/')}
        assert all(a==b for a,b in transfers.values()), {k:v for k,v in transfers.items() if v[0]!=v[1]}
        result=dict(exact=True,tokens=tokens,forced=force_rejections,events=len(stock),stock=ss,native=ns)
        (output/'parity.json').write_text(json.dumps(result,indent=2)); return result
    finally:
        transformer._native_glm_enabled=flag
