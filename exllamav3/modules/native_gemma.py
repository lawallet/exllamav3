"""Single-stream norm plumbing; Python retains attention, MLP, state and scale."""
import math
import weakref
import torch
from . import native_plan
from .rmsnorm import RMSNorm

_FLOAT_MAX = torch.finfo(torch.float32).max
_FLOAT_TINY = torch.finfo(torch.float32).tiny


def register(model):
    from .transformer import TransformerBlock
    for block in model.modules:
        if type(block) is TransformerBlock:
            block._native_plan_get = get_native_block
            block._native_plan_owner = weakref.ref(model)


def get_native_block(block, x, params):
    if not native_plan.enabled or not native_plan.v2_available:
        return None
    ref = getattr(block, '_native_plan_owner', None)
    owner = ref() if ref else None
    if (owner is None or owner.loaded_tp or x.ndim != 3 or
        not 1 <= x.shape[0] <= 4 or not 1 <= x.shape[1] <= 6 or x.shape[0]*x.shape[1] > 12 or
        x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous() or
        any(k in params for k in ('capture', 'quant_preserve', 'backend', 'loras', 'ovr',
                                  'reconstruct', 'export_state_norm_keys', 'activate_all_experts')) or
        any(params.get(k) for k in ('prefill', 'tp_warmup', 'autosplit_measure', 'layer_instance'))):
        return None
    from .attn import Attention
    from .sliding_attn import SlidingAttention
    from .mlp import GatedMLP
    if (type(block.attn) not in (Attention, SlidingAttention) or type(block.mlp) is not GatedMLP or
        getattr(block.attn, 'tp_mode', False) or block.mlp.tp_reduce or
        block.attn.out_dtype not in (None, torch.float16) or
        block.attn.o_proj.out_dtype not in (None, torch.float16) or
        block.mlp.out_dtype != torch.float32 or x.shape[2] != block.mlp.hidden_size or
        x.shape[2] <= 0 or x.shape[2] % 4 or
        any(getattr(block, k) is not None for k in
            ('attn_hc', 'mlp_hc', 'attn_resid_scalar', 'mlp_resid_scalar')) or
        (block.layer_scalar_f is not None and not math.isfinite(block.layer_scalar_f))):
        return None
    norms = (block.attn_norm, block.attn_post_norm, block.mlp_norm, block.mlp_post_norm)
    if any(type(n) is not RMSNorm for n in norms):
        return None
    if any(m.device is None or torch.device(m.device) != x.device for m in
           (block, block.attn, block.mlp, *norms)):
        return None
    signature = [id(block.attn), id(block.mlp), block.layer_scalar_f]
    for n in norms:
        w = n.weight
        if (n.groups != 1 or n.span_heads or n.unweighted or
            not _FLOAT_TINY <= n.rms_norm_eps <= _FLOAT_MAX or
            not abs(n.constant_bias) <= _FLOAT_MAX or
            not abs(n.constant_scale) <= _FLOAT_MAX or
            w is None or w.device != x.device or not w.is_contiguous() or
            w.shape != (x.shape[2],) or w.dtype not in (torch.float16, torch.bfloat16)):
            return None
        signature.extend((id(n), id(w), w.data_ptr(), w.dtype, n.rms_norm_eps,
                          n.constant_bias, n.constant_scale))
    signature = tuple(signature)
    key = (x.device, tuple(x.shape), x.dtype)
    cached = getattr(block, '_native_gemma', {}).get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    native = native_plan.gemma(norms, x.shape[0], x.shape[1])
    if not hasattr(block, '_native_gemma'):
        block._native_gemma = {}
    block._native_gemma[key] = (signature, native)
    return native
