"""Load/shape-time plan emitters. No Python IR interpretation during execution.

EXL3_NATIVE_PLAN=1 selects registered new-architecture plans directly. Qwen and
GLM retain their existing architecture flags for bespoke/reference selection.
Absent/older extension support falls back to stock before any native mutation.
"""
import os
import torch
from ..ext import exllamav3_ext as ext

enabled = os.environ.get('EXL3_NATIVE_PLAN', '0') == '1'
GR_MIX, HC_MIX, APPLY, RMS_NORM, ROUTE, MOE = range(6)
# Extension identity is fixed for the process. Resolve capability once, not
# through an extra Python function call for every block/token.
available = hasattr(ext, 'BC_BlockPlan')
v2_available = (getattr(ext, 'native_block_plan_v2_abi', None) == 1 and
                hasattr(ext, 'BC_BlockPlanV2'))
NORM_ADD = 6
RES_ADD_NORM, RES_ADD = 7, 8
v2_residual_available = v2_available and getattr(ext, 'native_block_plan_v2_residual_ops', None) == 1


def devstral(norms, batch, query):
    """Pre-norm residual plumbing around two Python sublayers."""
    return ext.BC_BlockPlanV2(
        [(0, 1), (1, 1), (2, 1), (-1, 0), (-1, 0)],
        [n.weight for n in norms],
        [[n.rms_norm_eps, n.constant_bias, n.constant_scale] for n in norms],
        [[(RMS_NORM, 0, 3, 0)], [(RES_ADD_NORM, 1, 4, 1)], [(RES_ADD, 2, 0, -1)]],
        [3, 4, -1], batch, query, norms[0].weight.numel())


def gemma(norms, batch, query):
    """Single residual, two Python sublayers, explicit independent norm operands."""
    return ext.BC_BlockPlanV2(
        [(0, 1), (1, 0), (2, 1), (-1, 0), (-1, 0)],
        [n.weight for n in norms],
        [[n.rms_norm_eps, n.constant_bias, n.constant_scale] for n in norms],
        [[(RMS_NORM, 0, 3, 0)],
         [(NORM_ADD, 1, 0, 1), (RMS_NORM, 0, 4, 2)],
         [(NORM_ADD, 2, 0, 3)]],
        [3, 4, -1], batch, query, norms[0].weight.numel())


def glm_placement(owner, draft=None):
    """Resolve the measured 35/10 split from participants, never visible count.

    Called only on specialization construction. Load/unload clears block plans.
    Hardware restrictions remain until other hardware has its own parity gates.
    """
    blocks = [m for m in owner.modules if getattr(m, '_native_glm_owner', None) is not None]
    by_layer = {m.layer_idx: m for m in blocks}
    if len(blocks) != 45 or set(by_layer) != set(range(45)):
        return False
    first, second = (torch.device(by_layer[i].device) for i in (0, 35))
    if first.type != 'cuda' or second.type != 'cuda' or first == second:
        return False
    if ('RTX PRO 6000' not in torch.cuda.get_device_name(first.index) or
            '4090 D' not in torch.cuda.get_device_name(second.index)):
        return False
    for i, block in by_layer.items():
        expected = first if i < 35 else second
        if any(torch.device(m.device) != expected for m in
               (block, block.attn, block.mlp, block.attn_hc, block.mlp_hc,
                block.attn_norm, block.mlp_norm)):
            return False
    tail = owner.modules[owner.modules.index(by_layer[44])+1:owner.logit_layer_idx+1]
    if not tail or any(m.device is None or torch.device(m.device) != second for m in tail):
        return False
    if draft is not None and any(torch.device(m.device) != first for m in draft.modules):
        return False
    return True


def qwen(weights, moe, attn_eps, mlp_eps, batch, query, backend=ext):
    if not enabled:
        return backend.BC_QwenBlock(weights, moe, attn_eps, mlp_eps, batch * query)
    return ext.BC_BlockPlan(
        [weights[:3], weights[3:6]], [GR_MIX, GR_MIX], [attn_eps, mlp_eps],
        [0., 0.], [0, 0],
        [[(GR_MIX, 0)], [],
         [(APPLY, 0), (GR_MIX, 1), (ROUTE, 1), (MOE, 1), (APPLY, 1)]],
        weights[6:], moe, batch, query)


def glm(weights, batch, query):
    if not enabled:
        return ext.BC_GlmBlock(weights, 1e-5, 1e-6, 20, query, batch)
    return ext.BC_BlockPlan(
        [weights[:4], weights[4:]], [HC_MIX, HC_MIX], [1e-5, 1e-5],
        [1e-6, 1e-6], [20, 20],
        [[(HC_MIX, 0), (RMS_NORM, 0)],
         [(APPLY, 0), (HC_MIX, 1), (RMS_NORM, 1)], [(APPLY, 1)]],
        [], None, batch, query)


def deepseek(sites, batch, query):
    """Two HC sites around explicit Python attention and MoE boundaries."""
    return ext.BC_BlockPlan(
        [[h.fn_h, h.base, h.scale, n.weight] for h, n in sites],
        [HC_MIX, HC_MIX], [h.rms_eps for h, _ in sites],
        [h.hc_eps for h, _ in sites], [h.sinkhorn_iters for h, _ in sites],
        [[(HC_MIX, 0), (RMS_NORM, 0)],
         [(APPLY, 0), (HC_MIX, 1), (RMS_NORM, 1)], [(APPLY, 1)]],
        [], None, batch, query)
