"""Rank-local bindings for existing native plans; no TP math or collectives.

Only registrations already attached to a supported model can cross the export
boundary. Attention/MLP remain upstream modules, including on empty-shard ranks.
"""
import math
import os
import warnings
from collections import Counter

import torch
from . import native_plan
from .rmsnorm import RMSNorm
from .attn import Attention
from .sliding_attn import SlidingAttention
from .mlp import GatedMLP
from .hyperconnections import GatedResidual, HyperConnection

_FLOAT_TINY = torch.finfo(torch.float32).tiny
_FLOAT_MAX = torch.finfo(torch.float32).max
_OP_COUNTS = dict(qwen=4, glm=6, deepseek=6, gemma=4, devstral=3)
# Resolved at import; only used when constructing an HC plan. Older extensions
# continue to use eager stages, and operators can opt out before startup.
hc_graphs_enabled = os.environ.get('EXL3_NATIVE_TP_GRAPHS', '1') == '1'
_hc_graphs_available = hasattr(getattr(native_plan.ext, 'BC_BlockPlan', object), 'set_graphs')


def _hc_plan(sites, batch, query, *, kind):
    plan = native_plan.deepseek(sites, batch, query)
    # GLM has a repeatable sustained-generation gain. DeepSeek's existing long
    # MTP repeatability issue prevents certifying that workload; keep it eager.
    if kind == 'glm' and hc_graphs_enabled and _hc_graphs_available:
        plan.set_graphs(True)
    return plan


def export_kind(block):
    if any(getattr(getattr(block, name, None), 'carry_pre', False)
           for name in ('attn_hc', 'mlp_hc')):
        return None
    for attr, kind in (('_native_qwen_owner', 'qwen'), ('_native_glm_owner', 'glm'),
                       ('_native_plan_owner', None)):
        ref = getattr(block, attr, None)
        owner = ref() if ref else None
        if owner is None or not owner.caps.get('supports_tp'):
            continue
        if kind is None:
            get = getattr(block, '_native_plan_get', None)
            kind = getattr(get, '__module__', '').rsplit('.', 1)[-1].removeprefix('native_')
        if kind in _OP_COUNTS:
            return kind
    return None


def bind(block, kind, context):
    if kind is None:
        return
    if kind not in _OP_COUNTS:
        raise ValueError(f'Unknown native TP plan registration: {kind}')
    block._native_tp_kind = kind
    block._native_tp_device = torch.device('cuda', context['device'])
    block._native_tp_backend = context['backend']
    block._native_tp_rank = context['rank']
    block._native_tp_world_size = context['world_size']
    block._native_tp_notices = context.setdefault('native_plan_notices', set())
    block._native_plan_get = get_native_block
    clear(block)


def clear(block):
    block._native_tp_plans = {}
    block._native_tp_stats = dict(executions=0, operations=0, builds=0,
                                  fallbacks=Counter())


def fallback(block, reason):
    block._native_tp_stats['fallbacks'][reason] += 1
    notices = block._native_tp_notices
    if reason not in notices:
        notices.add(reason)
        warnings.warn(f'Native IR TP rank {block._native_tp_rank} '
                      f'({block._native_tp_device}): stock fallback: {reason}',
                      RuntimeWarning, stacklevel=3)
    return None


def _tensor_key(t):
    if t is None:
        return None
    return (id(t), t.data_ptr(), None if t.is_inference() else t._version,
            tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _valid(t, device, shape, dtypes):
    return (t is not None and t.device == device and t.is_contiguous() and
            tuple(t.shape) == shape and t.dtype in dtypes)


def _norm(n, device, hidden):
    return (type(n) is RMSNorm and n.groups == 1 and not n.span_heads and
            not n.unweighted and math.isfinite(n.rms_norm_eps) and
            _FLOAT_TINY <= n.rms_norm_eps <= _FLOAT_MAX and
            abs(n.constant_bias) <= _FLOAT_MAX and
            abs(n.constant_scale) <= _FLOAT_MAX and
            _valid(n.weight, device, (hidden,), (torch.float16, torch.bfloat16)))


def get_native_block(block, x, params):
    if not native_plan.enabled:
        return None
    reject = lambda reason: fallback(block, reason)
    kind = block._native_tp_kind
    if block._native_tp_world_size < 2:
        return reject('native TP requires multiple ranks')
    # Null-backend warmup must not populate the real execution counters or plans.
    if params.get('tp_warmup'):
        return None
    if params.get('backend') is not block._native_tp_backend:
        return reject('missing or changed upstream TP backend')
    if (x.device != block._native_tp_device or x.dtype != torch.float32 or
            not x.is_contiguous()):
        return reject('residual device, dtype or layout')
    if (x.ndim != (4 if kind in ('qwen', 'glm', 'deepseek') else 3) or
            not 1 <= x.shape[0] <= 4 or not 1 <= x.shape[1] <= 6 or
            x.shape[0] * x.shape[1] > (5 if kind == 'qwen' else 12)):
        return reject('shape outside existing small-row IR domain')
    if kind == 'qwen' and (x.shape[1] not in (1, 5) or (x.shape[1] == 5 and x.shape[0] != 1)):
        return reject('shape outside existing Qwen IR domain')
    if (any(k in params for k in ('capture', 'quant_preserve', 'loras', 'ovr',
                                  'reconstruct', 'activate_all_experts', 'capture_sens')) or
            any(params.get(k) for k in ('prefill', 'autosplit_measure', 'layer_instance'))):
        return reject('prefill or execution override')
    # RMSNorm.forward owns this optional side effect. Final norm/mixer exports
    # used by MTP remain valid, but block-local exports must execute stock norms.
    exports = params.get('export_state_norm_keys')
    if exports and any(getattr(getattr(block, name, None), 'key', None) in exports
                       for name in ('attn_norm', 'mlp_norm', 'attn_post_norm', 'mlp_post_norm')):
        return reject('block-local norm state export')
    if (block.attn is None or block.mlp is None or
            not block.attn.tp_reduce or not block.mlp.tp_reduce or
            any(getattr(block, k) is not None for k in ('attn_resid_scalar', 'mlp_resid_scalar'))):
        return reject('unsupported distributed block boundary')
    if getattr(block.mlp, 'cpu_offload', False) or getattr(block.mlp, 'cpu_split_first', None) is not None:
        return reject('expert CPU offload')
    if kind in ('qwen', 'glm', 'deepseek'):
        if not native_plan.available:
            return reject('extension lacks BC_BlockPlan')
        if any(getattr(block, k) is not None for k in ('attn_post_norm', 'mlp_post_norm', 'layer_scalar_f')):
            return reject('unsupported stream residual modifiers')
        return _streams(block, x, reject)
    if not native_plan.v2_available or (kind != 'gemma' and not native_plan.v2_residual_available):
        return reject('extension lacks required v2 instructions')
    return _dense(block, x, reject)


def _cached(block, x, signature, build):
    key = (x.device, tuple(x.shape), x.dtype)
    cached = block._native_tp_plans.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    plan = build()
    block._native_tp_plans[key] = (signature, plan)
    block._native_tp_stats['builds'] += 1
    return plan


def _lookup(block, x, signature):
    cached = block._native_tp_plans.get((x.device, tuple(x.shape), x.dtype))
    return cached[1] if cached is not None and cached[0] == signature else None


def _norm_key(n):
    if type(n) is not RMSNorm:
        return type(n)
    return (type(n), _tensor_key(n.weight), n.rms_norm_eps, n.constant_bias,
            n.constant_scale, n.groups, n.span_heads, n.unweighted)


def _dense(block, x, reject):
    kind = block._native_tp_kind
    norms = ((block.attn_norm, block.attn_post_norm, block.mlp_norm, block.mlp_post_norm)
             if kind == 'gemma' else (block.attn_norm, block.mlp_norm))
    signature = (type(block.attn), type(block.mlp), block.attn.out_dtype,
                 getattr(block.mlp, 'out_dtype', None), block.mlp.hidden_size,
                 block.attn_hc, block.mlp_hc, block.layer_scalar_f,
                 block.attn_post_norm, block.mlp_post_norm,
                 tuple(_norm_key(n) for n in norms))
    cached = _lookup(block, x, signature)
    if cached is not None:
        return cached
    if (type(block.attn) not in (Attention, SlidingAttention) or
            type(block.mlp) is not GatedMLP or
            block.attn_hc is not None or block.mlp_hc is not None):
        return reject('unsupported dense block')
    hidden = x.shape[-1]
    if hidden <= 0 or hidden % 4 or hidden != block.mlp.hidden_size:
        return reject('hidden dimension')
    if kind == 'gemma':
        norms = (block.attn_norm, block.attn_post_norm, block.mlp_norm, block.mlp_post_norm)
        if block.attn.out_dtype not in (None, torch.float16) or block.mlp.out_dtype != torch.float32:
            return reject('Gemma boundary dtype')
        if block.layer_scalar_f is not None and not math.isfinite(block.layer_scalar_f):
            return reject('layer scalar')
        emit = native_plan.gemma
    else:
        norms = (block.attn_norm, block.mlp_norm)
        if (block.attn.out_dtype != torch.float32 or
                (type(block.mlp) is GatedMLP and block.mlp.out_dtype != torch.float32) or
                any(getattr(block, k) is not None for k in ('attn_post_norm', 'mlp_post_norm', 'layer_scalar_f'))):
            return reject('pre-norm boundary dtype or modifiers')
        emit = native_plan.devstral
    if not all(_norm(n, x.device, hidden) for n in norms):
        return reject('RMS norm operands')
    return _cached(block, x, signature, lambda: emit(norms, *x.shape[:2]))


def _streams(block, x, reject):
    kind = block._native_tp_kind
    sites = ((block.attn_hc, block.attn_norm), (block.mlp_hc, block.mlp_norm))
    if kind == 'qwen':
        if getattr(native_plan.ext, 'native_qwen_unfolded_gr', 0) != 1:
            return reject('extension lacks unfolded GR ABI')
        if any(type(h) is not GatedResidual for h, _ in sites):
            return reject('gated residual geometry')
        signature = tuple((_tensor_key(h.proj_h), _tensor_key(h.upx_h), _tensor_key(h.w_h),
                           h.proj_m, h.rms_eps, h.use_combine, h.hidden_size, h.hc_mult, n)
                          for h, n in sites)
        cached = _lookup(block, x, signature)
        if cached is not None:
            return cached
        bundles, eps = [], []
        for h, n in sites:
            if (type(h) is not GatedResidual or n is not None or not h.use_combine or
                    tuple(x.shape[2:]) != (4, h.hidden_size) or h.hidden_size % 8 or
                    not math.isfinite(h.rms_eps) or h.rms_eps <= 0):
                return reject('gated residual geometry')
            weights = (h.proj_h, h.upx_h, h.w_h)
            if any(t is None or t.device != x.device or t.dtype != torch.float16 or not t.is_contiguous() for t in weights):
                return reject('gated residual operands')
            bundles.append([h.proj_h[:h.proj_m], h.upx_h, h.w_h])
            eps.append(h.rms_eps)
        def build():
            # Both projections return fully reduced activations through Python.
            return native_plan.ext.BC_BlockPlan(
                bundles, [native_plan.GR_MIX] * 2, eps, [0., 0.], [0, 0],
                [[(native_plan.GR_MIX, 0)],
                 [(native_plan.APPLY, 0), (native_plan.GR_MIX, 1)],
                 [(native_plan.APPLY, 1)]], [], None, *x.shape[:2])
    else:
        if tuple(x.shape[2:]) != (4, 4096):
            return reject('HC residual geometry')
        if any(type(h) is not HyperConnection for h, _ in sites):
            return reject('HC operands')
        if any(getattr(h, 'carry_pre', False) for h, _ in sites):
            return reject('carried HC pre weights require stock execution')
        def signature_for_sites():
            return tuple((_tensor_key(h.fn), _tensor_key(h.fn_h), _tensor_key(h.base),
                          _tensor_key(h.scale), _norm_key(n), h.rms_eps, h.hc_eps,
                          h.sinkhorn_iters, h.hc_mult, h.hidden_size) for h, n in sites)
        signature = signature_for_sites()
        cached = _lookup(block, x, signature)
        if cached is not None:
            return cached
        for h, n in sites:
            if (type(h) is not HyperConnection or h.hc_mult != 4 or h.hidden_size != 4096 or
                    not _norm(n, x.device, 4096) or n.constant_bias != 0 or n.constant_scale != 1 or
                    h.rms_eps != n.rms_norm_eps or not math.isfinite(h.hc_eps) or h.hc_eps <= 0 or
                    not isinstance(h.sinkhorn_iters, int) or h.sinkhorn_iters < 1 or
                    not _valid(h.fn, x.device, (24, 16384), (torch.float32,)) or
                    not _valid(h.base, x.device, (24,), (torch.float32,)) or
                    not _valid(h.scale, x.device, (3,), (torch.float32,))):
                return reject('HC operands')
        for h, _ in sites:
            if h.fn_h is None:
                h.fn_h = h.fn.half()
            if not _valid(h.fn_h, x.device, (24, 16384), (torch.float16,)):
                return reject('HC half projection')
        signature = signature_for_sites()
        build = lambda: _hc_plan(sites, *x.shape[:2], kind=kind)
    return _cached(block, x, signature, build)


def record_execution(block):
    """Called only after all three native stages and upstream sublayers return."""
    stats = block._native_tp_stats
    stats['executions'] += 1
    stats['operations'] += _OP_COUNTS[block._native_tp_kind]


def diagnostics(context):
    """Explicit worker RPC: synchronize actual launches before reporting evidence."""
    torch.cuda.synchronize(context['device'])
    blocks = []
    for block in context['modules']:
        if getattr(block, '_native_tp_kind', None) is not None:
            plans = [p[1] for p in block._native_tp_plans.values()]
            blocks.append(dict(key=block.key, kind=block._native_tp_kind,
                               plans=len(block._native_tp_plans),
                               plan_types=sorted({type(p).__name__ for p in plans}),
                               graph_captures=sum(getattr(p, 'graph_captures', 0) for p in plans),
                               graph_replays=sum(getattr(p, 'graph_replays', 0) for p in plans),
                               attention_heads=getattr(block.attn, 'num_q_heads', None),
                               mlp_channels=getattr(block.mlp, 'intermediate_size', None),
                               **(block._native_tp_stats | {'fallbacks':dict(block._native_tp_stats['fallbacks'])})))
    return dict(rank=context['rank'], world_size=context['world_size'],
                device=context['device'], device_name=torch.cuda.get_device_name(context['device']),
                device_uuid=str(torch.cuda.get_device_properties(context['device']).uuid),
                backend=type(context['backend']).__name__,
                peak_allocated=torch.cuda.max_memory_allocated(context['device']),
                blocks=blocks)
