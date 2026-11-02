"""Guarded V4.1 carried-HC bindings. Python owns attention, Engram and MTP state."""
import math
import weakref
import torch
from . import native_plan
from .hyperconnections import HyperConnection
from .rmsnorm import RMSNorm

available = getattr(native_plan.ext, 'native_block_plan_carry_abi', None) == 1


def register(model):
    from .transformer import TransformerBlock
    if model.config.arch_string != 'DeepseekV41ForCausalLM':
        return
    for block in model.modules:
        if type(block) is TransformerBlock and all(
            getattr(h, 'carry_pre', False) for h in (block.attn_hc, block.mlp_hc)
        ):
            block._native_plan_get = get_native_block
            block._native_plan_owner = weakref.ref(model)
            block._native_carry_pre = True


def get_native_block(block, x, params):
    if not native_plan.enabled or not available:
        return None
    ref = getattr(block, '_native_plan_owner', None)
    owner = ref() if ref else None
    if (owner is None or owner.loaded_tp or
        x.ndim != 4 or x.shape[0] != 1 or not 1 <= x.shape[1] <= 5 or
        tuple(x.shape[2:]) != (4, 5120) or x.dtype != torch.float32 or
        not x.is_cuda or not x.is_contiguous() or
        any(k in params for k in ('capture', 'quant_preserve', 'backend', 'loras', 'ovr',
            'reconstruct', 'export_state_norm_keys', 'activate_all_experts')) or
        any(params.get(k) for k in ('prefill', 'tp_warmup', 'autosplit_measure', 'layer_instance'))):
        return None
    from .dsv4 import DSV4Attention
    from .arch_specific.dspark import DSparkAttention
    from .block_sparse_mlp import BlockSparseMLP
    if (type(block.attn) not in (DSV4Attention, DSparkAttention) or
        type(block.mlp) is not BlockSparseMLP or block.attn.tp_mode or
        block.attn.out_dtype != torch.float32 or
        any(getattr(block, k) is not None for k in ('attn_post_norm', 'mlp_post_norm',
            'attn_resid_scalar', 'mlp_resid_scalar', 'layer_scalar_f'))):
        return None
    sites = ((block.attn_hc, block.attn_norm), (block.mlp_hc, block.mlp_norm))
    if any(type(h) is not HyperConnection or type(n) is not RMSNorm or not h.carry_pre for h,n in sites):
        return None
    if any(m.device is None or torch.device(m.device) != x.device for m in
           (block, block.mlp, *(m for site in sites for m in site))):
        return None
    # Shared-pool attention may live on another GPU; TransformerBlock preserves
    # the stock input/output transfers. The residual never moves for that call.
    if block.attn.device is None or torch.device(block.attn.device).type != 'cuda':
        return None
    prev = params.get('hc_pre')
    if prev is not None and (prev.shape != x.shape[:3] or prev.dtype != torch.float32 or
                            not prev.is_cuda or not prev.is_contiguous()):
        return None
    signature = [x.device, tuple(x.shape[:2]), torch.device(block.attn.device)]
    for h,n in sites:
        if (h.hc_mult != 4 or h.hidden_size != 5120 or
            not isinstance(h.sinkhorn_iters,int) or h.sinkhorn_iters < 1 or
            not math.isfinite(h.hc_eps) or h.hc_eps <= 0 or
            not math.isfinite(h.rms_eps) or h.rms_eps <= 0 or h.rms_eps != n.rms_norm_eps or
            n.constant_bias != 0 or n.constant_scale != 1 or n.groups != 1 or n.span_heads or n.unweighted):
            return None
        tensors = (h.fn,h.base,h.scale,n.weight)
        if any(t is None or t.device != x.device or not t.is_contiguous() for t in tensors):
            return None
        if (h.fn.shape != (24,20480) or h.fn.dtype != torch.float32 or
            h.base.shape != (24,) or h.base.dtype != torch.float32 or
            h.scale.shape != (3,) or h.scale.dtype != torch.float32 or
            n.weight.shape != (5120,) or n.weight.dtype not in (torch.float16,torch.bfloat16)):
            return None
        if h.fn_h is not None and (h.fn_h.shape != h.fn.shape or h.fn_h.dtype != torch.float16 or
                                  h.fn_h.device != x.device or not h.fn_h.is_contiguous()):
            return None
        signature.extend((id(t),t.data_ptr()) for t in (*tensors,h.fn_h) if t is not None)
        signature.extend((h.rms_eps,h.hc_eps,h.sinkhorn_iters))
    key = (x.device,x.shape[0],x.shape[1])
    cached = block._native_deepseek.get(key)
    if cached is not None and cached[0] == tuple(signature):
        return cached[1]
    # Use the identical loaded-weight conversion as the stock small-row path.
    if any(h.fn_h is None for h,_ in sites):
        for h,_ in sites:
            if h.fn_h is None: h.fn_h = h.fn.half()
        return get_native_block(block,x,params)
    plan = native_plan.deepseek_carry(sites,x.shape[0],x.shape[1])
    block._native_deepseek[key] = (tuple(signature),plan)
    return plan
