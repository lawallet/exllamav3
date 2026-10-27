"""GLM attached three-draft protocol. Ordinary decode retains its own guard/cache."""
import torch
from ..ext import exllamav3_ext as ext
from .hyperconnections import HyperConnection
from .rmsnorm import RMSNorm
from . import native_plan


def get_native_mtp_block(block, x, params):
    owner_ref = getattr(block, '_native_glm_owner', None)
    owner = owner_ref() if owner_ref else None
    draft_ref = getattr(owner, '_native_glm_mtp_draft', None)
    draft = draft_ref() if draft_ref else None
    target_ref = getattr(draft, 'attached_model', None)
    if (owner is None or (native_plan.enabled and not native_plan.available) or draft is None or target_ref is None or target_ref() is not owner or
        owner.loaded_tp or draft.loaded_tp or not draft.caps.get('mtp_draft') or
        owner.config.arch_string != 'Glm5NextForConditionalGeneration' or
        owner.config.num_hidden_layers != 45 or owner.config.num_mtp_layers != 1 or
        getattr(ext, 'native_glm_mtp_max_q_len', 0) != 4 or
        x.ndim != 4 or not 1 <= x.shape[0] <= 3 or x.shape[1] not in (1,4) or tuple(x.shape[2:]) != (4,4096) or
        (x.shape[0] > 1 and getattr(ext, "native_glm_mtp_max_batch", 0) != 3) or
        not x.is_cuda or not x.is_contiguous() or x.dtype != torch.float32 or
        torch.device(block.device) != x.device or (not native_plan.enabled and x.device.index != (0 if block.layer_idx < 35 else 1)) or
        any(k in params for k in ('capture','quant_preserve','backend','export_state_layers','target_hidden','loras','activate_all_experts')) or
        any(params.get(k) for k in ('prefill','tp_warmup','autosplit_measure','layer_instance'))):
        return None
    if params.get('export_state_norm_keys') != {owner.modules[owner.logit_layer_idx-1].key}:
        return None
    if bool(params.get('recurrent_history')) != (x.shape[1] == 4):
        return None
    cache, states = params.get('cache'), params.get('recurrent_states')
    if (cache is None or cache.max_history != 3 or not states or len(states) != x.shape[0] or
        any(s.cache is not cache or s.exported for s in states) or
        (len(states)>1 and len({s.slot for s in states})!=len(states))):
        return None
    a,m,an,mn = block.attn_hc,block.mlp_hc,block.attn_norm,block.mlp_norm
    if (type(a) is not HyperConnection or type(m) is not HyperConnection or
        type(an) is not RMSNorm or type(mn) is not RMSNorm or
        any(v is not None for v in (block.attn_resid_scalar,block.mlp_resid_scalar,block.attn_post_norm,block.mlp_post_norm))):
        return None
    if any(n.weight is None or tuple(n.weight.shape) != (4096,) or
           n.weight.dtype not in (torch.float16, torch.bfloat16) or
           n.weight.device != x.device or not n.weight.is_contiguous() for n in (an, mn)):
        return None
    signature = (id(draft),) + tuple(id(t) for h,n in ((a,an),(m,mn)) for t in (h.fn,h.fn_h,h.base,h.scale,n.weight))
    signature += tuple(v for h,n in ((a,an),(m,mn)) for v in
        (h.hc_mult,h.hidden_size,h.sinkhorn_iters,h.hc_eps,h.rms_eps,
         n.rms_norm_eps,n.constant_bias,n.constant_scale,n.groups,n.span_heads,n.unweighted))
    signature += (x.device, tuple(x.shape[:2]), x.dtype, native_plan.enabled)
    slots = getattr(block, '_native_glm_mtp', None)
    if slots is None:
        slots = block._native_glm_mtp = {}
    key = x.shape[1] if x.shape[0] == 1 else tuple(x.shape[:2])
    cached = slots.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    if native_plan.enabled:
        if not native_plan.glm_placement(owner, draft):
            return None
    elif ('RTX PRO 6000' not in torch.cuda.get_device_name(0) or
        '4090 D' not in torch.cuda.get_device_name(1) or
        any(torch.device(mod.device) != torch.device('cuda:0') for mod in draft.modules)):
        return None
    for b in owner.modules:
        if not native_plan.enabled and getattr(b, '_native_glm_owner', None) is not None and torch.device(b.device).index != (0 if b.layer_idx < 35 else 1):
            return None
    weights=[]
    for h,n in ((a,an),(m,mn)):
        if (h.hc_mult != 4 or h.hidden_size != 4096 or h.sinkhorn_iters != 20 or
            h.hc_eps != 1e-6 or h.rms_eps != 1e-5 or n.rms_norm_eps != 1e-5 or
            n.constant_bias != 0 or n.constant_scale != 1 or n.groups != 1 or n.span_heads or n.unweighted or
            any(t is None or t.device != x.device or not t.is_contiguous() for t in (h.fn,h.base,h.scale,n.weight))):
            return None
        if (tuple(h.fn.shape) != (24,16384) or h.base.numel() != 24 or h.scale.numel() != 3 or
            h.base.dtype != torch.float32 or h.scale.dtype != torch.float32):
            return None
        if h.fn_h is None: h.fn_h=h.fn.half()
        weights.extend((h.fn_h,h.base,h.scale,n.weight))
    native=native_plan.glm(weights,x.shape[0],x.shape[1])
    slots[key]=(signature,native)
    return native
