"""Opt-in stateless V4.1 projection boundary in the native IR layer.

Only loaded K3 mul1 EXL3 operands are supported initially. Positional math and
all attention/cache/index/MTP state remain in DSV4Attention. Output storage is
borrowed until the next invocation of the same shape/stream specialization.
"""
import math
import os
import torch
from . import native_plan
from .linear import Linear
from .quant.exl3 import LinearEXL3
from .rmsnorm import RMSNorm

enabled = os.environ.get('EXL3_NATIVE_DSV41_PROJECTIONS', '0') == '1'
available = (getattr(native_plan.ext, 'native_projection_plan_abi', None) == 1
             and torch.version.hip is None)
graphs_enabled = os.environ.get('EXL3_GRAPHS', '1') != '0'


def get_plan(a, x, params):
    if not (enabled and available and native_plan.enabled and graphs_enabled):
        return None
    from .dsv4 import DSV4Attention
    if (type(a) is not DSV4Attention or a.config.arch_string != 'DeepseekV41ForCausalLM'
        or a.q_head_norm or a.tp_mode or x.ndim != 3 or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 5 or x.shape[2] != 5120
        or x.dtype != torch.half or not x.is_cuda or not x.is_contiguous()
        or any(k in params for k in ('capture', 'quant_preserve', 'ovr', 'reconstruct', 'loras', 'backend'))
        or any(params.get(k) for k in ('prefill', 'autosplit_measure', 'tp_warmup', 'layer_instance'))):
        return None
    n = a.q_norm
    if (type(n) is not RMSNorm or n.span_heads or n.groups != 1 or n.unweighted
        or n.constant_bias != 0 or n.constant_scale != 1
        or not math.isfinite(n.rms_norm_eps) or n.rms_norm_eps <= 0
        or n.key in params.get('export_state_norm_keys', ())):
        return None
    weights = []
    signature = [n.rms_norm_eps]
    for lin, dims in zip((a.q_a, a.q_b, a.wkv), ((5120,1280),(1280,32768),(5120,512))):
        if (type(lin) is not Linear or type(lin.inner) is not LinearEXL3
            or (lin.in_features,lin.out_features) != dims
            or (lin.in_features_unpadded,lin.out_features_unpadded) != dims
            or lin.is_sliced or lin.lora_a_tensors or lin.lora_b_tensors
            or lin.pre_scale != 1 or lin.post_scale != 1 or lin.softcap != 0
            or lin.out_dtype not in (None,torch.half)):
            return None
        inner = lin.inner
        if (inner.K != 3 or inner.mcg or not inner.mul1 or inner.bias is not None
            or inner.default_out_dtype != torch.half or inner.swap_device is not None):
            return None
        weights.extend((inner.trellis,inner.suh,inner.svh))
        signature.append(id(inner))
    weights.append(n.weight)
    if any(t is None or t.device != x.device or not t.is_contiguous() for t in weights):
        return None
    signature.extend((id(t),t.data_ptr()) for t in weights)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    key = (x.device,x.shape[1],stream)
    cached = a._native_projection_plans.get(key)
    if cached is not None and cached[0] == tuple(signature):
        return cached[1]
    # Validate newly bound operands before invoking the native constructor.
    if torch.cuda.get_device_capability(x.device) not in ((8,9),(12,0)):
        return None
    for i,(din,dout) in enumerate(((5120,1280),(1280,32768),(5120,512))):
        t,su,sv = weights[3*i:3*i+3]
        if (t.dtype != torch.int16 or t.shape != (din//16,dout//16,48)
            or su.dtype != torch.half or su.shape != (din,)
            or sv.dtype != torch.half or sv.shape != (dout,)):
            return None
    if n.weight.dtype not in (torch.half,torch.bfloat16) or n.weight.shape != (1280,):
        return None
    plan = native_plan.projections(weights,n.rms_norm_eps,x.shape[1])
    a._native_projection_plans[key] = (tuple(signature),plan)
    return plan
