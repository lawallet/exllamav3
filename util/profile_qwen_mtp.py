"""Fixed-four-draft Qwen MTP parity and alternating benchmark, using the Tabby venv."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.check_qwen_native import check
from util.check_qwen_mtp import check_mtp, NativeCounts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt-file', default=str(ROOT / 'doc/native_qwen_prompt.txt'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--extension-dir', type=Path)
    parser.add_argument('--installed', action='store_true')
    parser.add_argument('--tokens', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--parity-tokens', type=int, default=128)
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--compare-plan', action='store_true')
    args = parser.parse_args()
    if args.compare_plan and (args.parity_only or os.environ.get('EXL3_NATIVE_PLAN') != '1'):
        parser.error('--compare-plan requires EXL3_NATIVE_PLAN=1 and benchmarking')
    if args.installed and args.extension_dir:
        parser.error('--installed and --extension-dir are mutually exclusive')
    if min(args.tokens, args.runs, args.parity_tokens) < 1:
        parser.error('Counts must be positive')
    if args.installed:
        sys.path = [s for s in sys.path if Path(s or '.').resolve() != ROOT]
    elif args.extension_dir:
        sys.path.insert(1, str(args.extension_dir.resolve()))
    import torch
    import exllamav3
    import exllamav3_ext
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler
    from exllamav3.modules import transformer
    device = torch.device('cuda:0')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,0,1':
        raise ValueError('Use the validated CUDA_VISIBLE_DEVICES=2,0,1 mapping')
    torch.cuda.set_device(device)
    config = Config.from_directory(args.model)
    assert config.arch_string == 'Qwen4ExpForConditionalGeneration'
    assert config.mtp_num_hidden_layers == 1
    model = Model.from_config(config)
    draft = Model.from_config(config, component='mtp')
    tokenizer = Tokenizer.from_config(config)
    prompt = Path(args.prompt_file).read_text(encoding='utf-8')
    prompt_len = tokenizer.encode(prompt, encode_special_tokens=True).shape[-1]
    assert 160 <= prompt_len <= 300
    capacity = ((prompt_len + max(args.tokens, args.parity_tokens, 128) + 5 + 255) // 256) * 256
    cache = Cache(model, max_num_tokens=capacity, max_batch_size=1, max_history=4)
    draft_cache = Cache(draft, max_num_tokens=capacity, max_batch_size=1)
    draft.load(device=device, progressbar=True)
    model.load(device=device, progressbar=True)
    draft.attach_to(model)
    args.output.mkdir(parents=True, exist_ok=True)

    def make_generator():
        return Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft,
                         draft_cache=draft_cache, num_draft_tokens=4, dynamic_draft_tokens=False,
                         max_batch_size=1, ngram_match_min=0, record_draft_stats=True)

    metadata = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                    package=exllamav3.__file__, extension=exllamav3_ext.__file__,
                    device=torch.cuda.get_device_name(device), cache_tokens=capacity, max_history=4,
                    draft_tokens=4, dynamic_draft=False, sampler='greedy', seed=1234,
                    revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    environment={k: v for k, v in os.environ.items() if k.startswith(('EXL3_', 'CUDA_'))})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Guard against silently loading the older checkpoint extension.
    assert getattr(exllamav3_ext, 'native_qwen_mtp_max_q_len', 0) == 5
    print('Checking non-MTP regression...', flush=True)
    parity_plain = check(model, cache, tokenizer, prompt)
    (args.output / 'non-mtp-parity.json').write_text(json.dumps(parity_plain, indent=2), encoding='utf-8')
    print('Checking natural MTP...', flush=True)
    print(json.dumps(check_mtp(model, draft, cache, draft_cache, tokenizer, prompt, make_generator,
                              args.output / 'natural', tokens=args.parity_tokens)), flush=True)
    print('Checking deliberately rejected MTP proposals...', flush=True)
    print(json.dumps(check_mtp(model, draft, cache, draft_cache, tokenizer, prompt, make_generator,
                              args.output / 'forced', tokens=64, force_rejections=True)), flush=True)
    if args.parity_only:
        return

    def run(enabled, tokens):
        transformer._native_qwen_enabled = enabled
        gen = make_generator()
        torch.cuda.synchronize(device)
        start_epoch, start = time.time(), time.perf_counter()
        try:
            with NativeCounts(model, draft, gen) as counts:
                completion, result = gen.generate(
                    prompt, max_new_tokens=tokens, min_new_tokens=tokens, sampler=GreedySampler(),
                    seed=1234, stop_conditions=[], encode_special_tokens=True,
                    completion_only=True, return_last_results=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            accepted, rejected = result['accepted_draft_tokens'], result['rejected_draft_tokens']
            assert counts.counts['draft_tokens_proposed'] == accepted + rejected
            return dict(native=enabled, start_epoch=start_epoch, end_epoch=time.time(), total_seconds=elapsed,
                        **{k: result[k] for k in ('new_tokens', 'prompt_tokens', 'time_prefill', 'time_generate', 'cached_tokens')},
                        decode_tokens_per_second=(result['new_tokens'] - 1) / result['time_generate'],
                        effective_output_tokens_per_second=result['new_tokens'] / result['time_generate'],
                        end_to_end_tokens_per_second=result['new_tokens'] / elapsed,
                        accepted_draft_tokens=accepted, rejected_draft_tokens=rejected,
                        proposed_draft_tokens=accepted + rejected, acceptance_rate=accepted / max(1, accepted + rejected),
                        calls=counts.report(), completion_sha256=hashlib.sha256(completion.encode()).hexdigest())
        finally:
            gen.clear_queue()
            gen.filter_pool.shutdown(wait=True)

    for enabled in (False, True):
        run(enabled, 64)
    results = []
    for i in range(args.runs):
        for enabled in ((False, True) if i % 2 == 0 else (True, False)):
            result = run(enabled, args.tokens)
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / 'benchmark.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    if len({r['completion_sha256'] for r in results}) != 1:
        raise AssertionError('Measured MTP completions differ')
    if len({(r['accepted_draft_tokens'], r['rejected_draft_tokens']) for r in results}) != 1:
        raise AssertionError('Measured MTP acceptance counts differ')
    if args.compare_plan:
        from util.native_plan_validation import benchmark
        benchmark([model, draft], lambda n: run(True, n), args.tokens, args.runs, args.output, None, 64)


if __name__ == '__main__':
    main()
