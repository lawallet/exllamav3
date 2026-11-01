"""DeepSeek HC plan bindings; attention, routing and DSpark state stay in Python."""
import math
import torch
from . import native_plan
from .hyperconnections import HyperConnection
from .rmsnorm import RMSNorm


def register(model):
    """Register only this architecture's trunk/draft blocks, without a new flag."""
    import weakref
    from .transformer import TransformerBlock
    for block in model.modules:
        if type(block) is TransformerBlock and not any(
            getattr(h, 'carry_pre', False) for h in (block.attn_hc, block.mlp_hc)
        ):
            block._native_plan_get = get_native_block
            block._native_plan_owner = weakref.ref(model)


def get_native_block(block, x, params):
    if not native_plan.enabled or not native_plan.available:
        return None
    ref = getattr(block, '_native_plan_owner', None)
    owner = ref() if ref else None
    if (owner is None or owner.loaded_tp or
        x.ndim != 4 or tuple(x.shape[2:]) != (4, 4096) or
        not 1 <= x.shape[0] <= 4 or not 1 <= x.shape[1] <= 6 or x.shape[0] * x.shape[1] > 12 or
        x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous() or
        any(k in params for k in ('capture', 'quant_preserve', 'backend', 'loras', 'ovr', 'reconstruct',
                                  'export_state_norm_keys', 'activate_all_experts')) or
        any(params.get(k) for k in ('prefill', 'tp_warmup', 'autosplit_measure', 'layer_instance'))):
        return None
    from .dsv4 import DSV4Attention
    from .arch_specific.dspark import DSparkAttention
    from .block_sparse_mlp import BlockSparseMLP
    if (type(block.attn) not in (DSV4Attention, DSparkAttention) or
        type(block.mlp) is not BlockSparseMLP or block.attn.tp_mode or
        block.attn.out_dtype != torch.float32 or
        any(getattr(block, k) is not None for k in
            ('attn_post_norm', 'mlp_post_norm', 'attn_resid_scalar', 'mlp_resid_scalar', 'layer_scalar_f'))):
        return None
    # No state is bound by the plan. Existing Python attention validates/owns
    # slots, pools, verification and rewind, exactly as on the stock path.
    sites = ((block.attn_hc, block.attn_norm), (block.mlp_hc, block.mlp_norm))
    if any(type(h) is not HyperConnection or type(n) is not RMSNorm for h, n in sites):
        return None
    # V4.1 collapses with the previous site's weights; the existing IR encodes
    # only same-site collapse. Check before looking up an already cached plan.
    if any(getattr(h, 'carry_pre', False) for h, _ in sites):
        return None
    if any(m.device is None or torch.device(m.device) != x.device for m in
           (block, block.attn, block.mlp, *(m for site in sites for m in site))):
        return None
    signature = [x.device, tuple(x.shape[:2]), x.dtype]
    for h, n in sites:
        if (h.hc_mult != 4 or h.hidden_size != 4096 or
            not isinstance(h.sinkhorn_iters, int) or h.sinkhorn_iters < 1 or
            not math.isfinite(h.hc_eps) or h.hc_eps <= 0 or
            not math.isfinite(h.rms_eps) or h.rms_eps <= 0 or h.rms_eps != n.rms_norm_eps or
            n.constant_bias != 0 or n.constant_scale != 1 or n.groups != 1 or n.span_heads or n.unweighted):
            return None
        tensors = (h.fn, h.base, h.scale, n.weight)
        if any(t is None or t.device != x.device or not t.is_contiguous() for t in tensors):
            return None
        if (h.fn.shape != (24, 16384) or h.fn.dtype != torch.float32 or
            h.base.shape != (24,) or h.base.dtype != torch.float32 or
            h.scale.shape != (3,) or h.scale.dtype != torch.float32 or
            n.weight.shape != (4096,) or n.weight.dtype not in (torch.float16, torch.bfloat16)):
            return None
        if h.fn_h is not None and (h.fn_h.shape != h.fn.shape or h.fn_h.dtype != torch.float16 or
                                  h.fn_h.device != x.device or not h.fn_h.is_contiguous()):
            return None
        signature.extend(id(t) for t in (*tensors, h.fn_h))
        signature.extend((h.rms_eps, h.hc_eps, h.sinkhorn_iters))
    signature = tuple(signature)
    key = (x.device, x.shape[0], x.shape[1], x.dtype)
    cached = getattr(block, '_native_deepseek', {}).get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    # All rejection checks precede allocation and native execution. fn_h is the
    # same immutable loaded-weight conversion used by stock HC small-row decode.
    for h, _ in sites:
        if h.fn_h is None:
            h.fn_h = h.fn.half()
    native = native_plan.deepseek(sites, x.shape[0], x.shape[1])
    # fn_h was materialized above: store the post-construction identity.
    signature = list(signature)
    for i, (h, _) in enumerate(sites):
        signature[3 + i * 8 + 4] = id(h.fn_h)
    if not hasattr(block, '_native_deepseek'):
        block._native_deepseek = {}
    block._native_deepseek[key] = (tuple(signature), native)
    return native
