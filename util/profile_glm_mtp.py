"""GLM fixed-three-draft trace, exact parity gate and alternating benchmarks."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.decode_dispatch_profile import DispatchProfile


class MTPProfile(DispatchProfile):
    def _patch(self, obj, name, root=False):
        if not root:
            return super()._patch(obj, name)
        original = getattr(obj, name)
        def forward(ids=None, params=None, **kw):
            if ids is None:
                ids = kw.pop('input_ids')
            previous = self.active
            self.active = ids.shape[0] == 1 and ids.shape[1] <= 4
            if self.active:
                self.tokens += 1
            try:
                return original(ids, params, **kw)
            finally:
                self.active = previous
        self.saved.append((obj, name, obj.__dict__.get(name), name in obj.__dict__))
        setattr(obj, name, forward)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extension-dir', type=Path)
    p.add_argument('--package-dir', type=Path)
    p.add_argument('--trace-only', action='store_true')
    p.add_argument('--parity-only', action='store_true')
    p.add_argument('--compare-plan', action='store_true')
    p.add_argument('--tokens', type=int, default=1000)
    p.add_argument('--parity-tokens', type=int, default=128)
    p.add_argument('--forced-tokens', type=int, default=128)
    p.add_argument('--runs', type=int, default=3)
    a = p.parse_args()
    if a.compare_plan and (a.parity_only or a.trace_only or os.environ.get('EXL3_NATIVE_PLAN') != '1'):
        p.error('--compare-plan requires EXL3_NATIVE_PLAN=1 and validated benchmarking')
    if a.extension_dir and a.package_dir: p.error('Choose either --extension-dir or --package-dir')
    if a.tokens < 1000 or a.runs < 3:
        p.error('Measured benchmarks require >=1000 tokens and >=3 pairs')
    if a.extension_dir: sys.path.insert(1, str(a.extension_dir.resolve()))
    if a.package_dir: sys.path.insert(0, str(a.package_dir.resolve()))
    import torch
    import exllamav3
    import exllamav3_ext
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler
    from exllamav3.modules import transformer
    from exllamav3.version import __version__
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2,0'
    transformer._native_glm_enabled = False
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (a.output/name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    save('environment.json', dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        package=exllamav3.__file__, extension=exllamav3_ext.__file__, version=__version__,
        packages={n: importlib.metadata.version(n) for n in ('triton-windows','flash-linear-attention')},
        environment={k:v for k,v in os.environ.items() if k.startswith(('EXL','CUDA_'))},
        revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,power.limit,driver_version','--format=csv'], text=True),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    cfg = Config.from_directory(a.model)
    assert cfg.arch_string == 'Glm5NextForConditionalGeneration' and cfg.num_mtp_layers == 1
    model, draft = Model.from_config(cfg), Model.from_config(cfg, component='mtp')
    tokenizer = Tokenizer.from_config(cfg)
    prompt = (ROOT/'doc/native_glm53_prompt.txt').read_text(encoding='utf-8')
    (a.output/'prompt.txt').write_text(prompt, encoding='utf-8')
    save('model-config.json', json.loads((Path(a.model)/'config.json').read_text()))
    capacity = ((tokenizer.encode(prompt, encode_special_tokens=True).shape[-1] + max(a.tokens,a.parity_tokens)+259)//256)*256
    cache = Cache(model, max_num_tokens=capacity, max_batch_size=1, max_history=3)
    draft_cache = Cache(draft, max_num_tokens=capacity, max_batch_size=1)
    # Tabby loads the draft first using its stock autosplit policy, then the trunk.
    draft.load(progressbar=True, max_chunk_size=256)
    model.load(use_per_device=[90,44], tensor_p=False, max_chunk_size=256,
               max_batch_size=1, progressbar=True)
    draft.attach_to(model)
    save('placement.json', {r:[dict(key=m.key, type=type(m).__name__,device=str(m.device)) for m in mod.modules]
                           for r,mod in [('target',model),('draft',draft)]})
    save('configuration.json', dict(capacity=capacity, max_history=3, num_draft_tokens=3,
        default_draft_tokens=draft.caps['default_draft_size'], dynamic=False, split=[90,44], sampler='greedy',seed=1234))
    def make_generator():
        return Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft,
            draft_cache=draft_cache, num_draft_tokens=3, dynamic_draft_tokens=False,
            max_batch_size=1, max_chunk_size=256, ngram_match_min=0, record_draft_stats=True)
    def run(enabled, tokens):
        transformer._native_glm_enabled = enabled
        gen = make_generator()
        jobs=[]
        enqueue=gen.enqueue
        def remember(job,*args,**kwargs):
            jobs.extend(job if isinstance(job,list) else [job])
            return enqueue(job,*args,**kwargs)
        gen.enqueue=remember
        for d in model.active_devices: torch.cuda.synchronize(d)
        start_epoch, start = time.time(), time.perf_counter()
        try:
            completion,result = gen.generate(prompt,max_new_tokens=tokens,min_new_tokens=tokens,
                sampler=GreedySampler(),seed=1234,stop_conditions=[],encode_special_tokens=True,
                completion_only=True,return_last_results=True)
            for d in model.active_devices: torch.cuda.synchronize(d)
            accepted,rejected=result['accepted_draft_tokens'],result['rejected_draft_tokens']
            return dict(native=enabled,start_epoch=start_epoch,end_epoch=time.time(),total_seconds=time.perf_counter()-start,
                **{k:result[k] for k in ('new_tokens','prompt_tokens','time_prefill','time_generate','cached_tokens')},
                decode_tokens_per_second=(result['new_tokens']-1)/result['time_generate'],
                effective_output_tokens_per_second=result['new_tokens']/result['time_generate'],
                accepted=accepted,rejected=rejected,proposed=accepted+rejected,acceptance_rate=accepted/max(1,accepted+rejected),
                engine_first_token_epoch=jobs[0].time_first_token,
                engine_first_prefill_epoch=jobs[0].time_first_prefill,
                draft_stats=jobs[0].draft_stats,
                counts_derived_from_draft_stats=dict(target_verifications=len(jobs[0].draft_stats),
                    native_target_blocks=45*len(jobs[0].draft_stats) if enabled else 0,
                    native_mtp_blocks=0,stock_draft_forwards=sum(s[1] for s in jobs[0].draft_stats),
                    stock_repair_blocks=sum(s[2]>0 for s in jobs[0].draft_stats)),
                completion_sha256=hashlib.sha256(completion.encode()).hexdigest())
        finally:
            gen.clear_queue()
            gen.filter_pool.shutdown(wait=True)
    stopped = threading.Event()
    def telemetry():
        with (a.output/'telemetry.jsonl').open('w') as f:
            while not stopped.is_set():
                r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,power.draw,memory.used,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True)
                f.write(json.dumps(dict(time=time.time(),returncode=r.returncode,csv=r.stdout,error=r.stderr))+'\n'); f.flush()
                stopped.wait(1)
    worker=threading.Thread(target=telemetry,daemon=True); worker.start()
    try:
        if not a.trace_only:
            from util.check_glm_mtp import check_mtp
            for forced in (False,True):
                result=check_mtp(model,draft,cache,draft_cache,tokenizer,prompt,make_generator,
                    a.output/('forced' if forced else 'natural'),tokens=a.forced_tokens if forced else a.parity_tokens,force_rejections=forced)
                print(json.dumps(dict(exact=result['exact'],forced=forced,events=result['events'],
                    accepted=result['native']['accepted'],rejected=result['native']['rejected'],rewinds=result['native']['rewinds'])),flush=True)
            if a.parity_only: return
        save('warmup-stock.json',run(False,a.tokens))
        if not a.trace_only: save('warmup-native.json',run(True,a.tokens))
        results=[]
        for i in range(a.runs if not a.trace_only else 1):
            for enabled in ((False,True) if i%2==0 else (True,False)) if not a.trace_only else (False,):
                r=run(enabled,a.tokens); results.append(r); save('benchmark.json',results)
                print(json.dumps({k:v for k,v in r.items() if k!='draft_stats'}),flush=True)
        for enabled in (False,True) if not a.trace_only else (False,):
            shapes=Counter()
            with ExitStack() as scope:
                for role,mod in [('target',model),('draft',draft)]:
                    for name in ('forward','prefill'):
                        original=getattr(mod,name)
                        def call(input_ids=None,params=None,_f=original,_r=role,_n=name,**kw):
                            shapes[f'{_r}/{_n}/{tuple(input_ids.shape)}']+=1
                            return _f(input_ids,params,**kw)
                        scope.enter_context(patch.object(mod,name,call))
                with MTPProfile(model) as target_profile, MTPProfile(draft) as draft_profile:
                    from util.check_glm_mtp import NativeCounts
                    with NativeCounts(model,draft) as counts:
                        run(enabled,64)
            save(f'profile-{int(enabled)}.json',dict(shapes=dict(shapes),counts=counts.report(),target=target_profile.report(),draft=draft_profile.report()))
        if len({r['completion_sha256'] for r in results}) != 1:
            raise AssertionError('Benchmark completions differ')
        if a.compare_plan:
            from util.native_plan_validation import benchmark
            benchmark([model, draft], lambda n: run(True, n), a.tokens, a.runs, a.output, None, 64)
    finally:
        stopped.set(); worker.join()


if __name__ == '__main__': main()
