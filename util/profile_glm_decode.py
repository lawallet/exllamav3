"""GLM5Next stock baseline. Run with the existing Tabby venv; no installation."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.glm_dispatch_profile import GlmDispatchProfile as DispatchProfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--split', type=float, nargs='+', required=True,
                   help='GiB budgets in CUDA_VISIBLE_DEVICES order')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=1000)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--profile-tokens', type=int, default=32)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--ab', action='store_true')
    p.add_argument('--compare-plan', action='store_true', help='Also benchmark bespoke versus plan on the same loaded model')
    p.add_argument('--smoke', action='store_true', help='Short development gate; not a release benchmark')
    p.add_argument('--extension-dir', type=Path)
    p.add_argument('--package-dir', type=Path, help='Extracted wheel directory; import its package and extension')
    args = p.parse_args()
    if args.compare_plan and (not args.ab or os.environ.get('EXL3_NATIVE_PLAN') != '1'):
        p.error('--compare-plan requires --ab and EXL3_NATIVE_PLAN=1')
    if (not args.smoke and (args.tokens < 1000 or args.runs < 3)) or min(args.tokens, args.runs, args.profile_tokens) < 1:
        p.error('baseline requires >=1000 tokens, >=3 runs and positive profile tokens')
    if not args.ab and os.environ.get('EXL3_NATIVE_GLM_BLOCK', '0') != '0':
        p.error('This harness measures stock only; native benchmarking requires parity first')
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    if args.extension_dir:
        sys.path.insert(1, str(args.extension_dir.resolve()))
    if args.package_dir:
        if args.extension_dir:
            p.error('--package-dir and --extension-dir are mutually exclusive')
        sys.path.insert(0, str(args.package_dir.resolve()))
    import torch
    import exllamav3
    import exllamav3_ext
    from exllamav3.version import __version__
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler
    from exllamav3.modules import transformer
    transformer._native_glm_enabled = False
    save('environment.json', dict(
        python=sys.version, packages={n: importlib.metadata.version(n) for n in
            ('torch', 'exllamav3', 'triton-windows', 'flash-linear-attention')},
        cuda=torch.version.cuda, package=exllamav3.__file__, extension=exllamav3_ext.__file__,
        runtime_version=__version__,
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        environment={k: v for k, v in os.environ.items() if k.startswith(('CUDA_', 'EXL3_'))},
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,power.limit',
                                      '--format=csv'], text=True),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    config = Config.from_directory(args.model)
    if config.arch_string != 'Glm5NextForConditionalGeneration':
        raise ValueError(config.arch_string)
    raw = json.loads((Path(args.model) / 'config.json').read_text(encoding='utf-8'))
    save('model-config.json', raw)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    prompt = model.default_chat_prompt('Explain how a computer executes a program, from source code to processor instructions. Include concrete examples and explain each stage in detail.')
    (args.output / 'prompt.txt').write_text(prompt, encoding='utf-8')
    prompt_tokens = tokenizer.encode(prompt, encode_special_tokens=True).shape[-1]
    # The parity gate decodes 256 tokens and warmup decodes 128, even in smoke
    # mode with a shorter timed run. Reserve space for those untimed paths too.
    cache_size = ((prompt_tokens + max(args.tokens, args.profile_tokens, 256 if args.ab else 128) + 255) // 256) * 256
    cache = Cache(model, max_num_tokens=cache_size)
    model.load(use_per_device=args.split, tensor_p=False, max_chunk_size=256,
               max_batch_size=1, progressbar=True, verbose=True)
    placement = [dict(key=m.key, type=type(m).__name__, device=str(m.device)) for m in model.modules]
    save('placement.json', placement)
    save('configuration.json', dict(cache_tokens=cache_size, cache_type=type(cache).__name__,
        prompt_tokens=prompt_tokens, mtp=False, tensor_parallel=model.loaded_tp,
        layers=Counter(config.layer_types[:config.num_hidden_layers]),
        transitions=[dict(before=a, after=b) for a, b in zip(placement, placement[1:]) if a['device'] != b['device']]))
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=1,
                          ngram_match_min=0)
    def sync():
        for device in model.active_devices:
            torch.cuda.synchronize(device)
    def run(tokens):
        sync()
        start_epoch, start = time.time(), time.perf_counter()
        completion, result = generator.generate(prompt, max_new_tokens=tokens, min_new_tokens=tokens,
            seed=1234, sampler=GreedySampler(), stop_conditions=[], encode_special_tokens=True,
            completion_only=True, return_last_results=True)
        sync()
        return dict(start_epoch=start_epoch, end_epoch=time.time(), total_seconds=time.perf_counter()-start,
            **{k: result[k] for k in ('new_tokens', 'prompt_tokens', 'time_prefill', 'time_generate', 'cached_tokens')},
            decode_tokens_per_second=(result['new_tokens']-1)/result['time_generate'],
            completion_sha256=hashlib.sha256(completion.encode()).hexdigest())
    stopped = threading.Event()
    def telemetry():
        with (args.output / 'telemetry.jsonl').open('w', encoding='utf-8') as f:
            while not stopped.is_set():
                r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,power.draw,memory.used,power.limit', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                f.write(json.dumps(dict(time=time.time(), returncode=r.returncode, csv=r.stdout, error=r.stderr))+'\n')
                f.flush()
                stopped.wait(1)
    worker = threading.Thread(target=telemetry, daemon=True)
    worker.start()
    try:
        if args.ab:
            from util.check_glm_native import check
            save('parity.json', check(model, cache, tokenizer, prompt, steps=256))
            print('Exact GLM parity passed', flush=True)
            transformer._native_glm_enabled = True
            save('warmup-native.json', run(128))
            transformer._native_glm_enabled = False
        save('warmup.json', run(128))
        results = []
        for i in range(args.runs):
            for enabled in ((False, True) if i % 2 == 0 else (True, False)) if args.ab else (False,):
                transformer._native_glm_enabled = enabled
                results.append(dict(run=i, native=enabled, **run(args.tokens)))
                save('benchmark.json', results)
                print(json.dumps(results[-1]), flush=True)
        for enabled in (False, True) if args.ab else (False,):
            transformer._native_glm_enabled = enabled
            with DispatchProfile(model) as dispatch:
                run(args.profile_tokens)
            save(f'dispatch-{int(enabled)}.json', dispatch.report())
        if args.compare_plan:
            from util.native_plan_validation import benchmark
            transformer._native_glm_enabled = True
            benchmark(model, run, args.tokens, args.runs, args.output, DispatchProfile, args.profile_tokens)
        from util.native_plan_validation import memory_report
        memory_report(model, args.output)
        if args.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                with DispatchProfile(model, ranges=True):
                    run(args.profile_tokens)
            prof.export_chrome_trace(str(args.output / 'trace.json'))
    finally:
        stopped.set()
        worker.join()


if __name__ == '__main__':
    main()
