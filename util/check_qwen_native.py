"""Slow exact-parity checks, deliberately separate from timed generation."""
import hashlib
import torch


def digest(tensor):
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@torch.inference_mode()
def check(model, cache, tokenizer, prompt, steps=128):
    from exllamav3.modules import TransformerBlock
    from exllamav3.modules import transformer
    from exllamav3.modules.native_qwen import get_native_block
    from exllamav3 import GreedySampler
    blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
    original_flag = transformer._native_qwen_enabled
    original_methods = [(b, b.forward) for b in blocks]
    collected = []
    active = False

    def wrap(block, original):
        def forward(x, params, **kwargs):
            if active and transformer._native_qwen_enabled and get_native_block(block, x, params) is None:
                import sys
                lines = []
                def trace(frame, event, arg):
                    if frame.f_code is get_native_block.__code__ and event == 'return':
                        lines.append(frame.f_lineno)
                    return trace
                sys.settrace(trace)
                try:
                    get_native_block(block, x, params)
                finally:
                    sys.settrace(None)
                raise AssertionError(f'Unexpected fallback for {block.key}: native_qwen.py line {lines}; '
                                     f'input={x.shape}/{x.dtype}; params={list(params)}')
            y = original(x, params, **kwargs)
            if active:
                collected.append((block.key, digest(y)))
            return y
        return forward

    def run(enabled):
        nonlocal active
        transformer._native_qwen_enabled = enabled
        ids = tokenizer.encode(prompt, encode_special_tokens=True)
        length = ids.shape[-1]
        params = dict(attn_mode='flash_attn', cache=cache, past_len=0,
                      batch_shape=(1, cache.max_num_tokens))
        state = None
        records = []
        try:
            model.prefill(ids[:, :-1], params)
            state = params['recurrent_states']
            current = ids[:, -1:]
            sampler = GreedySampler()
            for step in range(steps):
                collected.clear()
                params = dict(attn_mode='flash_attn', cache=cache, past_len=length - 1 + step,
                              batch_shape=(1, cache.max_num_tokens), recurrent_states=state)
                active = True
                logits = model.forward(current, params)
                active = False
                current = sampler.forward(logits, tokenizer=tokenizer).cpu()
                record = dict(logits=digest(logits), token=int(current.item()), blocks=list(collected))
                if step in (0, 1, 7, 31, steps - 1):
                    record['recurrent'] = {
                        str(k): [digest(t) for t in layer.stash(state[0].slot)]
                        for k, layer in cache.get_all_recurrent_layers().items()}
                    valid = length + step
                    record['kv'] = {}
                    for key, layer in cache.layers.items():
                        # Rectangular batch maps logical pages consecutively. Compare only
                        # initialized K/V and raw/complete pooled QSA entries, never padding.
                        for name in ('k', 'v', 'raw_k', 'pooled'):
                            t = getattr(layer, name, None)
                            if t is None:
                                raise AssertionError(f'Expected QSA FP16 cache field {name}')
                            n = valid // layer.compress_ratio if name == 'pooled' else valid
                            record['kv'][f'{key}:{name}'] = digest(t.flatten(0, 1)[:n])
                records.append(record)
        finally:
            active = False
            for rs in state or params.get('recurrent_states', []):
                rs.free()
        return records

    try:
        for block, original in original_methods:
            block.forward = wrap(block, original)
        stock = run(False)
        native = run(True)
        used = sum(getattr(b, '_native_qwen', None) is not None for b in blocks)
        if used != len(blocks):
            raise AssertionError(f'Native path constructed for only {used}/{len(blocks)} blocks')
        for step, (a, b) in enumerate(zip(stock, native)):
            if a != b:
                fields = [k for k in a if a[k] != b[k]]
                raise AssertionError(f'Exact parity failed at step {step}: {fields}')
        x = torch.zeros((1, 1, 4, model.config.hidden_size), device=blocks[0].device)
        for params in ({'recurrent_history': True}, {'capture': None}, {'prefill': True},
                       {'backend': None}, {'export_state_layers': []}, {'quant_preserve': False}):
            assert get_native_block(blocks[0], x, params) is None, params
        assert get_native_block(blocks[0], x.expand(2, -1, -1, -1), {}) is None
        assert get_native_block(blocks[0], x.expand(-1, 2, -1, -1), {}) is None
        return dict(steps=steps, native_blocks=used, exact_logits=True, exact_tokens=True,
                    exact_block_residuals=True, exact_recurrent_and_ple=True, exact_kv_and_qsa=True,
                    state_checkpoint_steps=[0, 1, 7, 31, steps - 1], fallback_checks=True)
    finally:
        transformer._native_qwen_enabled = original_flag
        for block, original in original_methods:
            del block.forward
