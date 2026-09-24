"""Run with the Tabby venv. Imports checkout Python and the installed extension.

Baseline/profiling only: this script does not install or enable a native prototype.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.decode_dispatch_profile import DispatchProfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--prompt-file', required=True)
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--tokens', type=int, default=1000)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--profile-tokens', type=int, default=64)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--extension-dir', type=Path)
    p.add_argument('--installed', action='store_true', help='Import installed package instead of checkout')
    p.add_argument('--ab', action='store_true', help='Check exact parity, then alternate stock/native measured runs')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.tokens, args.runs, args.profile_tokens) < 1:
        p.error('token counts and runs must be positive')
    if args.installed:
        if args.extension_dir:
            p.error('--installed and --extension-dir are mutually exclusive')
        from util.check_qwen_native import check
        sys.path = [entry for entry in sys.path if Path(entry or '.').resolve() != ROOT]
    if args.extension_dir:
        sys.path.insert(1, str(args.extension_dir.resolve()))
    import torch
    import exllamav3
    import exllamav3_ext
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler
    from exllamav3.modules import transformer

    config = Config.from_directory(args.model)
    if config.arch_string != 'Qwen4ExpForConditionalGeneration':
        raise ValueError('This experiment requires Qwen4ExpForConditionalGeneration')
    prompt = Path(args.prompt_file).read_text(encoding='utf-8')
    tokenizer = Tokenizer.from_config(config)
    prompt_tokens = tokenizer.encode(prompt, encode_special_tokens=True).shape[-1]
    if not 160 <= prompt_tokens <= 300:
        raise ValueError(f'Expected 160–300 prompt tokens, got {prompt_tokens}')
    model = Model.from_config(config)
    cache_size = ((prompt_tokens + max(args.tokens, args.profile_tokens) + 255) // 256) * 256
    cache = Cache(model, max_num_tokens=cache_size)
    model.load(device=torch.device(args.device), progressbar=True)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          max_batch_size=1, ngram_match_min=0)
    args.output.mkdir(parents=True, exist_ok=True)

    def run(tokens):
        torch.cuda.synchronize(args.device)
        start_epoch = time.time()
        start = time.perf_counter()
        completion, result = generator.generate(
            prompt, max_new_tokens=tokens, min_new_tokens=tokens, seed=1234,
            sampler=GreedySampler(), stop_conditions=[], encode_special_tokens=True,
            completion_only=True, return_last_results=True)
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - start
        return dict(start_epoch=start_epoch, end_epoch=time.time(), total_seconds=elapsed, **{k: result[k] for k in (
            'new_tokens', 'prompt_tokens', 'time_prefill', 'time_generate', 'cached_tokens')},
            decode_tokens_per_second=(result['new_tokens'] - 1) / result['time_generate'],
            completion_sha256=hashlib.sha256(completion.encode()).hexdigest())

    metadata = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                    package=exllamav3.__file__, extension=exllamav3_ext.__file__,
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    device=torch.cuda.get_device_name(args.device), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    environment={k: v for k, v in os.environ.items() if k.startswith(('EXL3_', 'CUDA_'))})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if args.ab:
        if not args.installed:
            from util.check_qwen_native import check
        parity = check(model, cache, tokenizer, prompt)
        (args.output / 'parity.json').write_text(json.dumps(parity, indent=2), encoding='utf-8')
        print(json.dumps(parity), flush=True)
        for enabled in (False, True):
            transformer._native_qwen_enabled = enabled
            run(64)
    else:
        run(64)  # Warm BC graph replay, kernels, allocator, and prompt cache.
    results = []
    for i in range(args.runs):
        modes = (False, True) if i % 2 == 0 else (True, False)
        for enabled in modes if args.ab else (transformer._native_qwen_enabled,):
            transformer._native_qwen_enabled = enabled
            results.append(dict(native=enabled, **run(args.tokens)))
            print(json.dumps(results[-1]), flush=True)
            (args.output / 'benchmark.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for enabled in (False, True) if args.ab else (transformer._native_qwen_enabled,):
        transformer._native_qwen_enabled = enabled
        with DispatchProfile(model) as dispatch:
            run(args.profile_tokens)
        name = f'dispatch-{int(enabled)}.json' if args.ab else 'dispatch.json'
        (args.output / name).write_text(json.dumps(dispatch.report(), indent=2), encoding='utf-8')
    if args.trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            with DispatchProfile(model, ranges=True):
                run(args.profile_tokens)
        prof.export_chrome_trace(str(args.output / 'trace.json'))


if __name__ == '__main__':
    main()
