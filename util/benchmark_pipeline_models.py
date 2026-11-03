"""Matched invariant stop-mask reuse cohorts using existing model configurations and installed code.

Run sequentially with Tabby's Python -I and CUDA_VISIBLE_DEVICES=2,0,1
(2,0 for the established two-rank GLM TP configuration).
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'native-results/deepseek-v41/generation-pipeline'
MODELS = {
    'glm': ('GLM-5.3-Flash-Uncensored-EXL3-3.05bpw', [90,44], 5120, 3, 256),
    'glm-tp': ('GLM-5.3-Flash-Uncensored-EXL3-3.05bpw', None, 65536, 3, 256),
    'qwen': ('Qwen3.8-Flash-Next-Uncensored-exl3-4bpw', None, 5120, 4, 2048),
    'nemotron': ('Nemotron-3.5-Super-VL-4.23bpw', None, 32768, 3, 2048),
    'deepseek4': ('turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw', [85,40], 8192, 5, 256),
    'gemma': ('turboderp_gemma-4-31b-it-exl3_6.00bpw', [20,40], 8192, 0, 256),
    'devstral': ('turboderp_Devstral-2-123B-Instruct-2512-exl3_4.00bpw', [55,22], 8192, 0, 256),
}


def runtime_policy(context=None, value=None):
    runtime = ctypes.CDLL('libiomp5md.dll')
    runtime.kmp_get_blocktime.restype = ctypes.c_int
    runtime.kmp_set_blocktime.argtypes = [ctypes.c_int]
    runtime.kmp_set_blocktime.restype = None
    before = runtime.kmp_get_blocktime()
    if value is not None:
        runtime.kmp_set_blocktime(value)
    actual = runtime.kmp_get_blocktime()
    if value is not None:
        assert actual == value, (value, actual)
    try:
        from exllamav3.util.openmp import openmp_status
        implementation=openmp_status()
    except ImportError:
        implementation=None
    return dict(pid=os.getpid(), device=context.get('device') if context else 'parent',
                before=before, milliseconds=actual, implementation=implementation)


def native_mode(context, enabled):
    from exllamav3.modules import native_plan, native_moe, transformer
    native_plan.enabled = enabled
    native_moe.enabled = enabled
    transformer._native_glm_enabled = enabled
    transformer._native_qwen_enabled = enabled


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=MODELS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--package-dir', type=Path)
    p.add_argument('--implemented', action='store_true', help='Use env-selected implementation, never the experimental setter')
    p.add_argument('--settings', type=int, nargs='+', default=[0,1])
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--no-telemetry', action='store_true')
    p.add_argument('--no-concurrency', action='store_true')
    p.add_argument('--tail-only',action='store_true',help='Complete parity/telemetry controls without repeating retained primary cohorts')
    p.add_argument('--concurrency-tail',action='store_true',help='Include unfinished concurrent cases with --tail-only')
    p.add_argument('--concurrent-only',action='store_true',help='Fresh repeated concurrency check for a suspected regression')
    p.add_argument('--short-only',action='store_true',help='Matched fresh-process short-decode check without long/concurrent cohorts')
    p.add_argument('--warmup-rounds',type=int,default=1)
    p.add_argument('--primary-only',action='store_true',help='Repeat only warmup and primary pairs after a harness correction')
    p.add_argument('--baseline-wheel',action='store_true',help='Allow the unchanged installed .omp1 package for disabled-only controls')
    p.add_argument('--mode-controls',action='store_true',help='Repeat ordinary/unmasked/512-token masked pairs to check noisy secondary observations')
    a = p.parse_args()
    if a.package_dir:
        sys.path.insert(0, str(a.package_dir.resolve()))
    assert os.environ.get('EXL3_OMP_BLOCKTIME_MS') == '0'
    os.environ.setdefault('EXL3_NATIVE_GLM_BLOCK','1')
    os.environ.setdefault('EXL3_NATIVE_QWEN_BLOCK','1')
    import torch
    import exllamav3
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules import native_plan, native_moe, transformer
    if a.baseline_wheel:
        from types import SimpleNamespace
        assert a.package_dir is None and a.settings==[0]
        stop_mask_cache=SimpleNamespace(enabled=False)
        draft_seed=SimpleNamespace(enabled=False)
    else:
        from exllamav3.generator import stop_mask_cache
        from exllamav3.util import draft_seed
    torch.set_grad_enabled(False)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'benchmark-source.py').write_bytes(Path(__file__).read_bytes())
    expected_root=a.package_dir.resolve() if a.package_dir else Path('D:/tabbyAPI/venv')
    assert Path(exllamav3.__file__).is_relative_to(expected_root),exllamav3.__file__
    def save(name, data):
        (a.output/name).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    name, budgets, capacity, drafts, chunk = MODELS[a.case]
    slots=4 if a.case=='nemotron' else 2
    path = Path('D:/tabbyAPI/models') / name
    assert path.is_dir(), path
    cfg = Config.from_directory(str(path))
    model = Model.from_config(cfg)
    draft = Model.from_config(cfg, component='mtp') if drafts else None
    tok = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens=capacity, max_batch_size=slots, max_history=drafts or 5)
    dc = Cache(draft, max_num_tokens=capacity, max_batch_size=slots) if draft else None
    monitor = None
    try:
        # These placements/chunks follow the existing model-specific validation scripts.
        if draft and a.case != 'deepseek4':
            draft.load(device='cuda:0', max_chunk_size=chunk, max_batch_size=slots, progressbar=True)
        if a.case == 'glm-tp':
            model.load(tensor_p=True, tp_backend='native', reserve_per_device=2.0,
                       max_chunk_size=chunk, max_batch_size=slots, progressbar=True)
        elif budgets:
            model.load(use_per_device=budgets, max_chunk_size=chunk, max_batch_size=slots, progressbar=True)
        else:
            model.load(device='cuda:0', max_chunk_size=chunk, max_batch_size=slots, progressbar=True)
        if draft:
            if a.case == 'deepseek4':
                draft.load(device='cuda:0', max_chunk_size=chunk, max_batch_size=slots, progressbar=True)
            draft.attach_to(model)
        def policy(value=None):
            if value is not None:
                assert value in (0,1)
                stop_mask_cache.enabled=bool(value)
                draft_seed.enabled=bool(value)
            states=[runtime_policy(value=None)]
            if model.loaded_tp:
                states += [model.tp_worker_dispatch_single(d,runtime_policy,(None,))
                           for d in model.active_devices+[-1] if d != model.tp_output_device]
            assert all(s['milliseconds']==0 for s in states),states
            return states
        save('metadata.json', dict(case=a.case, model=str(path), architecture=cfg.arch_string,
             package=exllamav3.__file__, extension=ext.__file__, extension_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
             torch=torch.__version__, threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
             omp=policy(), capacity=capacity, chunk=chunk, drafts=drafts, budgets=budgets,
             cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid(), implemented=a.implemented,
             environment={k:v for k,v in os.environ.items() if k.startswith(('EXL3_','CUDA_VISIBLE'))},
             devices=[dict(device=d,name=torch.cuda.get_device_name(d),capability=torch.cuda.get_device_capability(d)) for d in model.active_devices],
             placement={role:[dict(key=m.key,device=str(m.device),type=type(m).__name__) for m in mod.modules]
                        for role,mod in [('target',model),('draft',draft)] if mod},
             tp_plan=model.plan if model.loaded_tp else None,
             cpu_offload=[dict(key=m.key,mode=getattr(m,'cpu_offload',None)) for m in model
                          if getattr(m,'cpu_offload',None)],
             config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest()))
        prompts=[]
        for i,text in enumerate(('Write a Python function that computes prime numbers. Explain the sieve algorithm step by step.',
                                 'Explain how ocean tides arise from gravity. Describe the roles of the Moon and Sun.',
                                 'Describe how to bake a chocolate cake, including ingredients and preparation.')):
            if i == 0 and a.case in ('glm','glm-tp','qwen'):
                text=(ROOT/'doc'/('native_qwen_prompt.txt' if a.case=='qwen' else 'native_glm53_prompt.txt')).read_text(encoding='utf-8')
            else:
                text=model.default_chat_prompt(text)
            prompts.append(tok.encode(text, encode_special_tokens=True))
        body=tok.encode('The engineering notebook contains measurements and observations. '*1500)
        long_prompt=torch.cat((prompts[0][:,:-2],body[:,:4096-prompts[0].shape[1]],prompts[0][:,-2:]),dim=1)
        assert long_prompt.shape[1] == 4096
        save('prompts.json',dict(short=[t.tolist() for t in prompts],long=long_prompt.tolist()))
        references={};cache_references={}
        def run(label, value, context='short', count=1, tokens=None, native=True, mtp=True, eos_mask=True, oracle=None):
            actual=policy(value)
            native_plan.enabled=native
            transformer._native_glm_enabled=native
            transformer._native_qwen_enabled=native
            native_moe.enabled=native
            if model.loaded_tp:
                model.tp_dispatch_all(native_mode, (native,))
            use_draft=draft if mtp else None
            gen=Generator(model,cache,tok,draft_model=use_draft,draft_cache=dc if use_draft else None,
                num_draft_tokens=drafts if use_draft else 0,dynamic_draft_tokens=False,
                max_batch_size=slots,max_chunk_size=chunk,ngram_match_min=0,record_draft_stats=True)
            n=tokens or a.tokens
            ids=[long_prompt] if context=='long' else prompts[:count]
            stops=list(set(cfg.eos_token_id_list)) if eos_mask else []
            from exllamav3.generator.sampler import CategoricalSampler, CustomSampler, SS_LogitBias, SS_Argmax
            sampler=lambda: CategoricalSampler(temperature=0.8) if oracle=='seeded' else (
                CustomSampler([SS_LogitBias({stops[0]:1000.0}),SS_Argmax()]) if oracle=='boundary' else GreedySampler())
            jobs=[Job(x.clone(),max_new_tokens=n,min_new_tokens=7 if oracle=='boundary' else n,stop_conditions=stops,sampler=sampler(),
                      seed=1234+i,identifier=str(i)) for i,x in enumerate(ids)]
            for d in model.active_devices:torch.cuda.synchronize(d)
            t0=time.time_ns();cpu=time.process_time();start=time.perf_counter();endings={}
            try:
                gen.enqueue(jobs)
                while gen.num_remaining_jobs():
                    for r in gen.iterate():
                        if r.get('stage')=='error':raise r['error']
                        if r.get('eos'):endings[r['identifier']]=r
                elapsed=time.perf_counter()-start;cpu_s=time.process_time()-cpu;t1=time.time_ns()
                records=[]
                for j in jobs:
                    r=endings[j.identifier]
                    records.append(dict(identifier=j.identifier,ids=j.sequences[0].sequence_ids.torch().tolist(),
                        **{k:r[k] for k in ('prompt_tokens','cached_tokens','new_tokens','time_prefill','time_generate')},
                        accepted_draft_tokens=r.get('accepted_draft_tokens',0),rejected_draft_tokens=r.get('rejected_draft_tokens',0)))
                signature=[(r['ids'],r['accepted_draft_tokens'],r['rejected_draft_tokens']) for r in records]
                key=(context,count,n,native,mtp,eos_mask,oracle)
                exact=key not in references or signature==references[key]
                references.setdefault(key,signature)
                occupancy=[r['cached_tokens'] for r in records]
                assert key not in cache_references or occupancy==cache_references[key],(label,'cache occupancy changed')
                cache_references.setdefault(key,occupancy)
                if not native and (context,count,n,True,mtp,eos_mask,oracle) in references:
                    assert signature==references[(context,count,n,True,mtp,eos_mask,oracle)],(label,'stock/native mismatch')
                if oracle=='boundary':
                    assert all(r['new_tokens']==8 for r in records),records
                    assert all(r['eos_reason']=='stop_token' for r in endings.values())
                result=dict(name=label,candidate=bool(value),runtime=actual,native=native,mtp=bool(use_draft),context=context,
                    count=count,eos_mask=eos_mask,oracle=oracle,stop_conditions=stops,request_start_ns=t0,response_end_ns=t1,wall_s=elapsed,cpu_s=cpu_s,
                    cpu_core_equivalents=cpu_s/elapsed,aggregate_output_tps=sum(r['new_tokens'] for r in records)/elapsed,
                    exact=exact,jobs=records,placement_file='metadata.json',
                    memory=[dict(device=d,allocated=torch.cuda.memory_allocated(d),reserved=torch.cuda.memory_reserved(d),
                                 peak=torch.cuda.max_memory_allocated(d)) for d in model.active_devices])
                save(label+'.json',result)
                assert exact,(label,'token/MTP mismatch')
                # A new generator prevents previous-request caching. Concurrent
                # jobs can legitimately share a prefix within the same request.
                assert records[0]['cached_tokens']==0,(label,'previous-request prefix cache hit')
                print(json.dumps(dict(case=a.case,name=label,tps=[r['new_tokens']/r['time_generate'] for r in records],
                    prefill=[r['time_prefill'] for r in records],cpu=result['cpu_core_equivalents'],exact=exact)),flush=True)
            finally:
                gen.close()
        if a.concurrent_only:
            for value in a.settings:run(f'concurrent-warmup-cache{value}',value,count=2,tokens=128)
            for i in range(a.rounds):
                order=a.settings if i%2==0 else list(reversed(a.settings))
                for value in order:run(f'concurrent-r{i}-cache{value}',value,count=2,tokens=128)
            save('complete.json',dict(exact=True,settings=a.settings,concurrent_only=True))
            return
        if a.mode_controls:
            cases=[('ordinary',dict(tokens=128,mtp=False)),
                   ('unmasked',dict(tokens=256,eos_mask=False)),
                   ('masked512',dict(tokens=512))]
            for name,options in cases:
                for value in a.settings:run(f'{name}-warm-cache{value}',value,**options)
                for i in range(a.rounds):
                    order=a.settings if i%2==0 else list(reversed(a.settings))
                    for value in order:run(f'{name}-r{i}-cache{value}',value,**options)
            save('complete.json',dict(exact=True,settings=a.settings,mode_controls=True))
            return
        for warm in range(a.warmup_rounds):
            for value in a.settings:
                label=f'warmup-cache{value}' if a.warmup_rounds==1 else f'warmup-r{warm}-cache{value}'
                run(label,value)
        if not a.tail_only:
            for i in range(a.rounds):
                order=a.settings if i%2==0 else list(reversed(a.settings))
                for value in order:run(f'steady-r{i}-cache{value}',value)
            if a.primary_only:
                save('complete.json',dict(exact=True,settings=a.settings,primary_only=True))
                return
            if not a.short_only:
                run('long-warmup',a.settings[0],context='long',tokens=128)
                for i in range(2):
                    for value in (a.settings if i%2==0 else list(reversed(a.settings))):
                        run(f'long-r{i}-cache{value}',value,context='long',tokens=128)
        if not a.short_only and not a.no_concurrency and (not a.tail_only or a.concurrency_tail):
            run('concurrent-warmup',a.settings[0],count=3 if a.case=='nemotron' else 2,tokens=128)
            for i in range(2):
                for value in (a.settings if i%2==0 else list(reversed(a.settings))):
                    run(f'concurrent-r{i}-cache{value}',value,count=3 if a.case=='nemotron' else 2,tokens=128)
        # Minimum-length EOS suppression creates CPU mask work absent from no-stop requests.
        # Keep this serving-style case separate from the retained no-stop primary cohort.
        run('masked-warmup',a.settings[0],eos_mask=True)
        for i in range(2):
            for value in (a.settings if i%2==0 else list(reversed(a.settings))):
                run(f'masked-r{i}-cache{value}',value,eos_mask=True)
        # Stock and ordinary generation are separate parity controls, not pooled with MTP timing.
        run('native-parity-control',a.settings[0],tokens=128)
        for value in a.settings:
            run(f'unmasked-cache{value}',value,eos_mask=False)
            for oracle in ('boundary','seeded'):
                run(f'{oracle}-cache{value}',value,tokens=64,oracle=oracle)
        for value in dict.fromkeys((a.settings[0],a.settings[-1])):
            run(f'stock-cache{value}',value,tokens=128,native=False)
            if draft:run(f'ordinary-cache{value}',value,tokens=128,mtp=False)
        # Cancellation is a separate correctness cohort, never throughput data.
        cancellation_reference=None
        for value in a.settings:
            policy(value)
            native_mode({},True)
            if model.loaded_tp:model.tp_dispatch_all(native_mode,(True,))
            gen=Generator(model,cache,tok,draft_model=draft,draft_cache=dc,
                num_draft_tokens=drafts,dynamic_draft_tokens=False,max_batch_size=slots,
                max_chunk_size=chunk,ngram_match_min=0,record_draft_stats=True)
            jobs=[Job(x.clone(),max_new_tokens=64,min_new_tokens=64,
                      stop_conditions=cfg.eos_token_id_list,sampler=GreedySampler(),seed=42+i)
                  for i,x in enumerate(prompts[:2])]
            cancelled=False
            try:
                gen.enqueue(jobs)
                while gen.num_remaining_jobs():
                    events=gen.iterate()
                    for event in events:
                        if event.get('stage')=='error':raise event['error']
                    if not cancelled and jobs[0].new_tokens>=12:
                        gen.cancel(jobs[0]);cancelled=True
                assert cancelled and jobs[1].new_tokens==64
                signature=[dict(ids=j.sequences[0].sequence_ids.torch().tolist(),
                           accepted=j.accepted_draft_tokens,rejected=j.rejected_draft_tokens) for j in jobs]
                assert cancellation_reference is None or signature==cancellation_reference
                cancellation_reference=signature
                save(f'cancellation-cache{value}.json',dict(candidate=bool(value),exact=True,jobs=signature))
            finally:gen.close()
        if not a.no_telemetry:
            monitor=subprocess.Popen([sys.executable,'-I',str(ROOT/'util/deepseek_v41_telemetry.py'),
                '--pid',str(os.getpid()),'--exact-pid','--output',str(a.output/'telemetry')],creationflags=subprocess.CREATE_NO_WINDOW)
            time.sleep(3)
            for value in a.settings:run(f'monitored-cache{value}',value,tokens=max(a.tokens,512))
        bindings={}
        for role,mod in [('target',model),('draft',draft)]:
            if mod:
                for m in mod:
                    cached={k:len(v) if isinstance(v,(dict,list,tuple)) else 1 for k,v in vars(m).items()
                            if k.startswith('_native_') and isinstance(v,(dict,list,tuple)) and v}
                    if cached:bindings[role+':'+m.key]=cached
        save('complete.json',dict(exact=True,settings=a.settings,native_bindings=bindings))
        if model.loaded_tp:
            from exllamav3.modules.native_tp import diagnostics
            save('tp-native-evidence.json',model.tp_worker_dispatch_wait_multi(model.active_devices,diagnostics,()))
    finally:
        if monitor:
            (a.output/'telemetry/stop').touch();monitor.wait(timeout=20)
        if draft:draft.unload()
        model.unload()


if __name__ == '__main__':
    main()
