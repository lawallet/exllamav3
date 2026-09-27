"""Byte-exact GLM parity diagnostics, intentionally outside timed generation."""
import hashlib
import torch


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def check(model, cache, tokenizer, prompt, steps=128):
    from exllamav3.modules import TransformerBlock, transformer
    from exllamav3.modules.native_glm import get_native_block
    from exllamav3 import GreedySampler
    if steps < 128:
        raise ValueError('Use at least 128 decode steps')
    blocks = [b for b in model.modules if isinstance(b, TransformerBlock)]
    flag = transformer._native_glm_enabled
    saved = []
    active = False
    collected = {}
    placement = [(m.key, str(m.device)) for m in model.modules]
    checkpoints = {0, 1, 7, 31, 63, steps - 1}

    def patch(obj, name, fn):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, fn)

    for block in blocks:
        original = block.forward
        def forward(x, params, original=original, block=block, **kwargs):
            if active and transformer._native_glm_enabled:
                if get_native_block(block, x, params) is None:
                    raise AssertionError(f'Unexpected native fallback: {block.key}')
            y = original(x, params, **kwargs)
            if active:
                collected[block.key] = digest(y)
                cfg = getattr(block.mlp, 'routing_cfg', None)
                if cfg is not None:
                    collected[block.key + ':routing'] = {
                        name: digest(getattr(cfg, name)) for name in
                        ('selected_experts_bsz1', 'routing_weights_bsz1')}
            return y
        patch(block, 'forward', forward)
    for m in model.modules:
        original = m.prepare_for_device
        def prepare(x, params, original=original, m=m):
            transfer = active and x.device != m.device
            if transfer:
                collected[m.key + ':transfer-in'] = (str(x.device), digest(x))
            y = original(x, params)
            if transfer:
                collected[m.key + ':transfer-out'] = (str(y.device), digest(y))
            return y
        patch(m, 'prepare_for_device', prepare)
    norm = model.modules[-2]
    original_norm = norm.forward
    def norm_forward(x, params, **kwargs):
        if active:
            collected['final-hidden-before-norm'] = digest(x)
        y = original_norm(x, params, **kwargs)
        if active:
            collected['final-hidden-after-norm'] = digest(y)
        return y
    patch(norm, 'forward', norm_forward)

    def run(enabled):
        nonlocal active
        transformer._native_glm_enabled = enabled
        ids = tokenizer.encode(prompt, encode_special_tokens=True)
        length = ids.shape[-1]
        params = dict(attn_mode='flash_attn', cache=cache, past_len=0,
                      batch_shape=(1, cache.max_num_tokens))
        states = None
        records = []
        try:
            model.prefill(ids[:, :-1], params)
            states = params['recurrent_states']
            current = ids[:, -1:]
            for step in range(steps):
                collected.clear()
                params = dict(attn_mode='flash_attn', cache=cache, past_len=length-1+step,
                              batch_shape=(1, cache.max_num_tokens), recurrent_states=states)
                active = True
                logits = model.forward(current, params)
                active = False
                current = GreedySampler().forward(logits, tokenizer=tokenizer).cpu()
                record = dict(logits=digest(logits), token=int(current.item()), tensors=dict(collected))
                record['positions'] = {k: digest(v) for k, v in params.items()
                                       if torch.is_tensor(v) and k in
                                       ('block_table', 'cache_seqlens', 'positions', 'recurrent_slots')}
                record['recurrent_positions'] = [s.position for s in states]
                if step in checkpoints:
                    record['recurrent'] = {str(k): [digest(t) for t in layer.stash(states[0].slot)]
                        for k, layer in cache.get_all_recurrent_layers().items()}
                    valid = length + step
                    record['kv'] = {}
                    for key, layer in cache.layers.items():
                        for name in ('k', 'v', 'k_idx', 'k_pool'):
                            t = getattr(layer, name, None)
                            if t is None:
                                raise AssertionError(f'Missing GLM cache plane {key}:{name}')
                            n = valid // layer.kpool if name == 'k_pool' else valid
                            record['kv'][f'{key}:{name}'] = digest(t.flatten(0, 1)[:n])
                records.append(record)
        finally:
            active = False
            for s in states or params.get('recurrent_states', []):
                s.free()
        return records
    try:
        stock = run(False)
        from util.native_plan_validation import three_way_reference
        bespoke = three_way_reference(model, run)
        native = run(True)
        if bespoke is not None and bespoke != native:
            raise AssertionError('Exact bespoke/plan GLM parity failed')
        for step, (a, b) in enumerate(zip(stock, native)):
            if a != b:
                raise AssertionError(f'Byte-exact GLM parity failed at step {step}: '
                                     f'{[k for k in a if a[k] != b[k]]}')
        assert placement == [(m.key, str(m.device)) for m in model.modules]
        x = torch.zeros((1, 1, 4, 4096), device=blocks[0].device)
        for params in ({'recurrent_history': True}, {'capture': None}, {'prefill': True},
                       {'backend': None}, {'export_state_layers': []}, {'quant_preserve': False},
                       {'export_state_norm_keys': set()}, {'loras': []}):
            assert get_native_block(blocks[0], x, params) is None, params
        assert get_native_block(blocks[0], x.expand(2, -1, -1, -1), {}) is None
        assert get_native_block(blocks[0], x.expand(-1, 2, -1, -1), {}) is None
        return dict(steps=steps, native_blocks=len(blocks), exact=True, exact_bespoke_plan=bespoke is not None,
                    compared=['tokens', 'logits', 'all block residuals', 'routing selections/weights',
                              'final hidden', 'transfer inputs/outputs', 'positions',
                              'convolution/recurrent state', 'latent KV', 'indexer keys', 'pooled keys'],
                    state_checkpoint_steps=sorted(checkpoints), placement=placement,
                    stock=stock, native=native)
    finally:
        transformer._native_glm_enabled = flag
        for obj, name, owned, value in reversed(saved):
            if owned:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
