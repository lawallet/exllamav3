"""Process-local attention diagnostics over the unchanged installed release."""
import importlib.util,json,time,os
from pathlib import Path
spec=importlib.util.spec_from_file_location('attention_pipeline',Path(__file__).with_name('serve_deepseek_v41_pipeline.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
f=p.f;b=p.b;torch=b.torch
from exllamav3.modules.dsv4 import DSV4Attention
probe_spec=importlib.util.spec_from_file_location('projection_probe',Path(__file__).with_name('deepseek_v41_projection_probe.py'))
probe=importlib.util.module_from_spec(probe_spec);probe_spec.loader.exec_module(probe)
implemented=os.environ.get('DS41_ATTENTION_IMPLEMENTED')=='1'
if implemented:
    from exllamav3.modules import native_projection

def candidate_status():
    if not implemented:return probe.status()
    plans=[v[1] for a in stages() for v in a._native_projection_plans.values()]
    return dict(implementation=native_projection.__file__,enabled=native_projection.enabled,
                available=native_projection.available,plans=len(plans),
                captures=sum(p.graph_captures for p in plans),replays=sum(p.graph_replays for p in plans))
events=[];event_keys=set();samples={}
requested_seed=None;observed_seeds=[]
from exllamav3.generator.job import Job
original_job_init=Job.__init__
def job_init(job,*args,**kwargs):
    if requested_seed is not None:kwargs['seed']=requested_seed
    original_job_init(job,*args,**kwargs)
    observed_seeds.append(job.seed)
Job.__init__=job_init

original_trace=b.start_trace
def start_trace():
    original_trace()
    old=DSV4Attention._project_qkv
    b.trace_saved.append((DSV4Attention,'_project_qkv',old))
    def traced(a,*args,**kwargs):
        with torch.profiler.record_function('ds41.projections:'+a.key):
            return old(a,*args,**kwargs)
    DSV4Attention._project_qkv=traced
b.start_trace=start_trace

def stages():
    c=b.serving.container
    return [block.attn for block in c.model.modules if type(getattr(block,'attn',None)) is DSV4Attention]

def metadata():
    result=[]
    for a in stages():
        lins={}
        for name in ('q_a','q_b','wkv','wo_b','idx_wq_b','idx_wk','idx_weights_proj'):
            lin=getattr(a,name,None)
            if lin is None:continue
            inner=lin.inner
            lins[name]=dict(format=lin.quant_type,input=lin.in_features,output=lin.out_features,
                unpadded=lin.out_features_unpadded,K=getattr(inner,'K',None),mcg=getattr(inner,'mcg',None),
                mul1=getattr(inner,'mul1',None),bias=getattr(inner,'bias',None) is not None,
                dtype=str(lin.out_dtype),pre=lin.pre_scale,post=lin.post_scale,softcap=lin.softcap)
        result.append(dict(key=a.key,device=str(a.device),q_head_norm=a.q_head_norm,
            compressor=a.compressor is not None,indexer=a.indexer is not None,
            kv_source=a.kv_source,candidate_mode=a.candidate_mode,linears=lins))
    return result

def profile_stage(old,label):
    def run(a,x,*args,**kw):
        if not f.active:return old(a,x,*args,**kw)
        key=(label,a.layer_idx,tuple(x.shape))
        pair=None
        if key not in event_keys:
            event_keys.add(key)
            with torch.cuda.device(x.device):
                pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                pair[0].record()
        result=f.measured(f'Attention.{label}|{a.layer_idx}|{tuple(x.shape)}',old,a,x,*args,**kw)
        if pair:
            pair[1].record();events.append((key,str(x.device),pair))
            if label=='qkv':samples[(a.layer_idx,x.shape[1])]=(a,x.detach().clone())
        return result
    return run

def configure(profile=False):
    events.clear();event_keys.clear();samples.clear()
    f.install('host' if profile else 'off')
    if profile:
        if implemented:
            f.hook(native_projection,'get_plan',lambda old:lambda *args,**kw:
                f.measured('ProjectionPlan.guard',old,*args,**kw))
            f.hook(b.ext.BC_ProjectionPlan,'run',lambda old:lambda *args,**kw:
                f.measured('ProjectionPlan.run',old,*args,**kw))
        for name,label in (('_project_qkv','qkv'),('_project_o_grouped','output')):
            f.hook(DSV4Attention,name,lambda old,label=label:profile_stage(old,label))
        for a in stages():
            for name in ('q_a','q_norm','q_b','wkv','wo_b','idx_wq_b','idx_wk'):
                lin=getattr(a,name,None)
                if lin is not None:
                    f.hook(lin,'forward',lambda old,name=name,layer=a.layer_idx:
                        lambda *args,**kw:f.measured(f'Projection.{name}|{layer}',old,*args,**kw))

old_setup=b.server.setup_app
def setup(*args,**kwargs):
    app=old_setup(*args,**kwargs)
    @app.post('/__attention/control',dependencies=[b.Depends(b.check_admin_key)])
    async def control(body:dict):
        global requested_seed
        c=b.serving.container
        if c.active_job_ids or c.generator.jobs:raise b.HTTPException(409,'requests active')
        f.uninstall()
        requested_seed=body.get('seed');observed_seeds.clear()
        if implemented:
            native_projection.enabled=body.get('candidate',False)
            for a in stages():
                for _,plan in a._native_projection_plans.values():
                    plan.set_graphs(body.get('native_graphs',True))
        else:probe.set_enabled(body.get('candidate',False))
        configure(body.get('profile',False))
        return dict(profile=body.get('profile',False),layers=metadata(),pipeline=p.policy_status(),candidate=candidate_status())
    @app.get('/__attention/results',dependencies=[b.Depends(b.check_admin_key)])
    async def results():
        timings=[]
        for key,device,(start,end) in events:
            end.synchronize();timings.append(dict(key=key,device=device,gpu_ms=start.elapsed_time(end)))
        return dict(events=timings,rows=f.rows,layers=metadata(),pipeline=p.policy_status(),candidate=candidate_status(),
                    requested_seed=requested_seed,observed_seeds=observed_seeds)
    @app.post('/__attention/lifecycle',dependencies=[b.Depends(b.check_admin_key)])
    async def lifecycle(body:dict):
        c=b.serving.container
        if c.active_job_ids or c.generator.jobs:raise b.HTTPException(409,'requests active')
        from exllamav3 import Job,GreedySampler
        from exllamav3.generator.sampler import CustomSampler,SS_LogitBias,SS_Argmax
        await c.generator.close()
        gen=b.Generator(c.model,c.cache,c.tokenizer,draft_model=c.draft_model,draft_cache=c.draft_cache,
            num_draft_tokens=3,dynamic_draft_tokens=False,max_batch_size=1,max_chunk_size=c.chunk_size,
            ngram_match_min=0,record_draft_stats=True)
        records=[]
        def execute(label,texts,cancel=False,boundary=False):
            stops=c.model.config.eos_token_id_list
            jobs=[Job(c.tokenizer.encode(text,add_bos=False),max_new_tokens=64,
                min_new_tokens=7 if boundary else 64,stop_conditions=stops,
                sampler=CustomSampler([SS_LogitBias({stops[0]:1000.0}),SS_Argmax()]) if boundary else GreedySampler(),
                seed=42+i,identifier=str(i)) for i,text in enumerate(texts)]
            gen.enqueue(jobs);endings={};cancelled=False
            while gen.num_remaining_jobs():
                for event in gen.iterate():
                    if event.get('stage')=='error':raise event['error']
                    if event.get('eos'):endings[event['identifier']]=event
                if cancel and not cancelled and jobs[0].new_tokens>=12:
                    gen.cancel(jobs[0]);cancelled=True
            if cancel:assert cancelled and jobs[1].new_tokens==64
            if boundary:assert all(j.new_tokens==8 for j in jobs)
            records.append(dict(label=label,jobs=[dict(ids=j.sequences[0].sequence_ids.torch().tolist(),
                accepted=j.accepted_draft_tokens,rejected=j.rejected_draft_tokens,new_tokens=j.new_tokens,
                cached_tokens=endings.get(j.identifier,{}).get('cached_tokens'),
                eos_reason=endings.get(j.identifier,{}).get('eos_reason')) for j in jobs]))
        try:
            execute('initial',[body['prompt']])
            execute('prefix-repeat',[body['prompt']])
            if body.get('prefix_only'):
                assert records[1]['jobs'][0]['cached_tokens']>0,records[1]['jobs'][0]['cached_tokens']
                return dict(records=records,candidate=candidate_status(),max_batch_size=1,positive_prefix_hit=True)
            execute('queued-cancellation',[body['prompt'],'Explain how ocean tides arise from gravity.'])
            execute('cancel-and-drain',[body['prompt'],'Describe the roles of the Moon and Sun.'],cancel=True)
            execute('forced-eos-boundary',[body['prompt']],boundary=True)
            return dict(records=records,candidate=candidate_status(),max_batch_size=1)
        finally:
            gen.close()
            await c.create_generator()
    return app
b.server.setup_app=setup
if __name__=='__main__':
    import main
    main.entrypoint()
