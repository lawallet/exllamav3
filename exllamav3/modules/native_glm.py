"""Opt-in GLM5Next ordinary mHC orchestration, with stock attention/MLP."""
import torch
from ..ext import exllamav3_ext as ext
from .hyperconnections import HyperConnection
from .rmsnorm import RMSNorm
from . import native_plan


def get_native_block(block, x, params):
    if 'export_state_norm_keys' in params:
        from .native_glm_mtp import get_native_mtp_block
        return get_native_mtp_block(block, x, params)
    ref = getattr(block, '_native_glm_owner', None)
    owner = ref() if ref else None
    if (owner is None or owner.loaded_tp or (native_plan.enabled and not native_plan.available) or not hasattr(ext, 'BC_GlmBlock') or
        owner.config.arch_string != 'Glm5NextForConditionalGeneration' or
        (len(x.shape) != 4 or not 1 <= x.shape[0] <= 4 or tuple(x.shape[1:]) != (1, 4, 4096)) or x.dtype != torch.float32 or
        (x.shape[0] > 1 and getattr(ext, "native_glm_max_batch", 0) != 4) or
        not x.is_cuda or not x.is_contiguous() or torch.device(block.device) != x.device or
        any(k in params for k in ('capture', 'quant_preserve', 'backend', 'export_state_layers',
                                  'export_state_norm_keys', 'target_hidden', 'loras', 'activate_all_experts')) or
        any(params.get(k) for k in ('prefill', 'tp_warmup', 'autosplit_measure', 'layer_instance',
                                    'recurrent_history'))):
        return None
    cache = params.get('cache')
    states = params.get('recurrent_states')
    if (cache is None or cache.max_history != 0 or getattr(owner, 'caps', {}).get('mtp_draft') or
        not states or len(states) != x.shape[0] or
        any(s.cache is not cache or s.exported for s in states) or
        (len(states) > 1 and len({s.slot for s in states}) != len(states))):
        return None
    # Only the measured model shape and ordered Pro 6000 -> 4090D split. Do not
    # infer residency from a CUDA ordinal alone (CUDA_VISIBLE_DEVICES remaps it).
    expected_device = 0 if block.layer_idx < 35 else 1
    if not native_plan.enabled and x.device.index != expected_device:
        return None
    a, m, an, mn = block.attn_hc, block.mlp_hc, block.attn_norm, block.mlp_norm
    if (type(a) is not HyperConnection or type(m) is not HyperConnection or
        type(an) is not RMSNorm or type(mn) is not RMSNorm or
        block.attn_resid_scalar is not None or block.mlp_resid_scalar is not None or
        block.attn_post_norm is not None or block.mlp_post_norm is not None):
        return None
    # Reloads replace these tensors; retain owners alongside native scratch and
    # compare identity so no old weight bundle can survive unload/reload.
    if any(n.weight is None or tuple(n.weight.shape) != (4096,) or
           n.weight.dtype not in (torch.float16, torch.bfloat16) or
           n.weight.device != x.device or not n.weight.is_contiguous() for n in (an, mn)):
        return None
    signature = tuple(id(t) for h, n in ((a, an), (m, mn)) for t in (h.fn, h.fn_h, h.base, h.scale, n.weight))
    signature += tuple(v for h, n in ((a, an), (m, mn)) for v in
        (h.hc_mult, h.hidden_size, h.sinkhorn_iters, h.hc_eps, h.rms_eps,
         n.rms_norm_eps, n.constant_bias, n.constant_scale, n.groups, n.span_heads, n.unweighted))
    signature += (x.device, tuple(x.shape[:2]), x.dtype, native_plan.enabled)
    batch = x.shape[0]
    cached = getattr(block, '_native_glm', None) if batch == 1 else getattr(block, '_native_glm_batch', {}).get(batch)
    if cached is not None and cached[0] == signature:
        return cached[1]
    if native_plan.enabled:
        if not native_plan.glm_placement(owner):
            return None
    elif ('RTX PRO 6000' not in torch.cuda.get_device_name(0) or
        '4090 D' not in torch.cuda.get_device_name(1) or owner.config.num_hidden_layers != 45):
        return None
    for b in owner.modules:
        if not native_plan.enabled and getattr(b, '_native_glm_owner', None) is not None:
            if torch.device(b.device).index != (0 if b.layer_idx < 35 else 1):
                return None
    weights = []
    for h, n in ((a, an), (m, mn)):
        if (h.hc_mult != 4 or h.hidden_size != 4096 or h.sinkhorn_iters != 20 or
            h.hc_eps != 1e-6 or h.rms_eps != 1e-5 or n.rms_norm_eps != 1e-5 or
            n.constant_bias != 0 or n.constant_scale != 1 or n.groups != 1 or
            n.span_heads or n.unweighted or
            any(t is None or t.device != x.device or not t.is_contiguous()
                for t in (h.fn, h.base, h.scale, n.weight))):
            return None
        if (tuple(h.fn.shape) != (24, 16384) or h.base.numel() != 24 or h.scale.numel() != 3 or
            h.base.dtype != torch.float32 or h.scale.dtype != torch.float32):
            return None
        if h.fn_h is None:
            h.fn_h = h.fn.half()
        weights.extend((h.fn_h, h.base, h.scale, n.weight))
    native = native_plan.glm(weights, batch, 1)
    if batch == 1:
        block._native_glm = (signature, native)
    else:
        if not hasattr(block, '_native_glm_batch'): block._native_glm_batch = {}
        block._native_glm_batch[batch] = (signature, native)
    return native
