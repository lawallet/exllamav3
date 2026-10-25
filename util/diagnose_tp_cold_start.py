"""Isolated first-request reproduction using the installed WebUI engine."""
import argparse
import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--cold', action='store_true')
    p.add_argument('--prompt-repeats', type=int, default=32)
    p.add_argument('--no-mtp', action='store_true')
    p.add_argument('--no-ir', action='store_true')
    p.add_argument('--warmup', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.package.resolve()))
    if a.cold:
        os.environ['TRITON_CACHE_DIR'] = str(out/'triton-cache')
    os.environ['EXL3_NATIVE_PLAN'] = '0' if a.no_ir else '1'
    os.environ['EXL3_NATIVE_GLM_BLOCK'] = '0' if a.no_ir else '1'
    import torch
    import exllamav3
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.version import __version__
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler
    torch.set_grad_enabled(False)
    record = dict(python=sys.executable, package=exllamav3.__file__, extension=ext.__file__,version=__version__,
                  torch=torch.__version__,triton=importlib.metadata.version('triton-windows'),
                  visible=os.environ.get('CUDA_VISIBLE_DEVICES'),cold=a.cold,
                  max_context=65536,mtp=not a.no_mtp,ir=not a.no_ir)
    def save(): (out/'result.json').write_text(json.dumps(record,indent=2))
    def stage(name):
        print(time.strftime('%H:%M:%S'),name,flush=True)
        record['stage']=name
        save()
    assert Path(exllamav3.__file__).is_relative_to(a.package.resolve())
    assert torch.cuda.device_count() == 2
    cfg = Config.from_directory(r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw')
    model = Model.from_config(cfg)
    tokenizer = Tokenizer.from_config(cfg)
    vision = draft = generator = None
    try:
        if 'vision_config' in cfg.config_dict:
            stage('load vision')
            vision = Model.from_config(cfg,component='vision')
            vision.load(progressbar=True)
        if not a.no_mtp:
            stage('load draft')
            draft = Model.from_config(cfg,component='mtp')
            draft_cache = Cache(draft,max_num_tokens=65536,max_batch_size=4)
            draft.load(progressbar=True)
        else:
            draft_cache = None
        stage('load target')
        cache = Cache(model,max_num_tokens=65536,max_batch_size=4,max_history=3 if draft else 0)
        model.load(tensor_p=True,tp_backend='native',use_per_device=[90,46],progressbar=True)
        if a.warmup:
            stage('isolated warmup')
            import exllamav3.model.model_tp as tp_runtime
            timeout = tp_runtime.DISPATCH_TIMEOUT
            forward = model.forward_tp
            def compile_forward(x, params, last_kv_module_idx, modules):
                assert params.get('tp_warmup')
                x, reserve = model.prepare_inputs_for_tp(x,params)
                result = None
                try:
                    for device in model.active_devices:
                        model.tp_worker_dispatch(device,tp_runtime.mp_model_forward,
                            (x,params,last_kv_module_idx,False,None))
                        value = model.tp_worker_result(device)
                        if device == model.tp_output_device: result = value
                    return result
                finally:
                    model.restore_tp_params(params,reserve)
            try:
                tp_runtime.DISPATCH_TIMEOUT = 600
                model.forward_tp = compile_forward
                failures = model.warmup(cache=cache,max_chunk_size=2048,max_batch_size=4,
                                        max_q_len=6,long_context=True,verbose=True)
                assert not failures, failures
            finally:
                tp_runtime.DISPATCH_TIMEOUT = timeout
                model.forward_tp = forward
        stage('create generator')
        generator = Generator(model=model,cache=cache,tokenizer=tokenizer,draft_model=draft,
                              draft_cache=draft_cache,num_draft_tokens=3 if draft else 0,
                              max_batch_size=4)
        record['initial_diagnostics'] = model.native_plan_tp_diagnostics()
        prompt = 'Explain how tensor parallel inference divides work between GPUs.\n' * a.prompt_repeats
        record['prompt_tokens'] = tokenizer.encode(prompt).shape[-1]
        record['generator_chunk'] = generator.max_chunk_size
        stage('first generation')
        faulthandler.dump_traceback_later(80,repeat=True)
        start=time.monotonic()
        results = []
        for _ in range(2):
            completion, result = generator.generate(prompt,max_new_tokens=32,min_new_tokens=32,
                sampler=GreedySampler(),seed=1234,completion_only=True,return_last_results=True)
            results.append({k:v for k,v in result.items() if k in ('new_tokens','prompt_tokens',
                              'time_prefill','time_generate','accepted_draft_tokens','rejected_draft_tokens')})
        faulthandler.cancel_dump_traceback_later()
        record.update(passed=True,seconds=time.monotonic()-start,
                      results=results,
                      diagnostics=model.native_plan_tp_diagnostics())
        stage('passed')
    except Exception as exc:
        record.update(passed=False,error=repr(exc))
        save()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if generator is not None:
            generator.clear_queue()
            generator.filter_pool.shutdown(wait=True)
        if model.loaded_tp or model.output_device is not None: model.unload()
        if draft is not None: draft.unload()
        if vision is not None: vision.unload()

if __name__ == '__main__':
    main()
