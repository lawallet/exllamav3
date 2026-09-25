"""Experimental native residual/MoE orchestration; no new math or cache path."""
import torch
from ..ext import exllamav3_ext as ext
from .hyperconnections import GatedResidual
from .block_sparse_mlp import BlockSparseMLP
from .block_sparse_mlp_routing import _gate_t


def _mtp_eligible(owner, x, params):
    """Recognize the attached Qwen4Exp MTP protocol, not arbitrary state exports.

    Only the final mixer exports the trunk tap. Block-local exports remain stock.
    Native scratch is stateless; GDN history, rewind and accepted-token commits
    still run through the original attention/model/generator implementation.
    """
    if getattr(ext, 'native_qwen_mtp_max_q_len', 0) < 5:
        return False
    is_draft = owner.caps.get('mtp_draft', False)
    draft_ref = getattr(owner, '_native_qwen_mtp_draft', None)
    draft = owner if is_draft else draft_ref() if draft_ref else None
    target_ref = getattr(draft, 'attached_model', None)
    target = target_ref() if target_ref else None
    if (target is None or (not is_draft and target is not owner) or target.loaded_tp or
        draft.loaded_tp or draft.config.mtp_num_hidden_layers != 1 or
        target.config.arch_string != 'Qwen4ExpForConditionalGeneration'):
        return False
    # Both components must live on the same GPU. In particular, an individually
    # single-device draft on a second GPU is outside this prototype's scope.
    if any(torch.device(mod.device) != x.device
           for mod in draft.modules if mod.device is not None and not mod.caps.get('prefer_cpu')):
        return False
    if is_draft and any(torch.device(mod.device) != x.device
                        for mod in target.modules if mod.device is not None and not mod.caps.get('prefer_cpu')):
        return False
    export = params.get('export_state_norm_keys')
    history = params.get('recurrent_history', False)
    if is_draft:
        hidden = params.get('target_hidden')
        return (x.shape[1] == 1 and not history and export is None and hidden is not None and
                tuple(hidden.shape) == (1, 1, x.shape[2] * x.shape[3]))
    expected = {target.modules[target.logit_layer_idx - 1].key}
    # The supported generator configuration drafts four tokens: a one-token
    # bootstrap or a five-token verification. Shortened/different windows fall back.
    if export != expected or x.shape[1] not in (1, 5):
        return False
    if x.shape[1] > 1 and not history:
        return False
    states = params.get('recurrent_states')
    cache = params.get('cache')
    return (bool(states) and len(states) == 1 and states[0].cache is cache and
            not states[0].exported and cache.max_history >= x.shape[1] - 1)


def get_native_block(block, x, params):
    owner_ref = getattr(block, '_native_qwen_owner', None)
    owner = owner_ref() if owner_ref else None
    if (owner is None or owner.loaded_tp or
        not hasattr(ext, 'BC_QwenBlock') or x.ndim != 4 or x.shape[0] != 1 or
        x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous() or
        any(k in params for k in ('capture', 'quant_preserve', 'backend', 'export_state_layers',
                                  'activate_all_experts', 'loras')) or
        any(params.get(k) for k in ('prefill', 'tp_warmup', 'autosplit_measure', 'layer_instance'))):
        return None
    mtp = (x.shape[1] != 1 or params.get('recurrent_history') or
           'export_state_norm_keys' in params or getattr(owner, 'caps', {}).get('mtp_draft'))
    if mtp and not _mtp_eligible(owner, x, params):
        return None
    if getattr(block, '_native_qwen', None) is None and not getattr(block, '_native_qwen_mtp', None):
        # Explicit single-device load does not populate Model.active_devices.
        # Check the actual placement, including the head and PLE, instead.
        devices = {torch.device(m.device) for m in owner.modules
                   if m.device is not None and not m.caps.get('prefer_cpu')}
        if devices != {x.device}:
            return None
    a, h, m = block.attn_hc, block.mlp_hc, block.mlp
    if not (type(a) is GatedResidual and type(h) is GatedResidual and type(m) is BlockSparseMLP):
        return None
    if (not a.use_combine or not h.use_combine or torch.device(a.device) != x.device or torch.device(h.device) != x.device or
        torch.device(m.device) != x.device or torch.device(block.attn.device) != x.device or
        any(getattr(block, k) is not None for k in ('attn_norm', 'mlp_norm', 'attn_post_norm',
            'mlp_post_norm', 'layer_scalar_f', 'attn_resid_scalar', 'mlp_resid_scalar')) or
        m.bc is None or not m.is_quantized or m.router_type != 'std' or
        m.cpu_offload or m.cpu_split_first is not None or m.tp_reduce or m.num_slices != 1 or
        m.routing_device is not None or m.alt_residual_channel or m.config.infer_params.no_reconstruct or
        any(getattr(m, k) is not None for k in ('router_pre_norm', 'routed_pre_norm', 'routed_post_norm',
            'shared_experts_post_norm', 'latent_in', 'latent_out', 'per_expert_scale')) or
        (m.shared_experts is not None and not m.bc_sh_exp) or
        m.intermediate_size == 0 or m.num_local_experts != m.num_experts):
        return None
    rows = x.shape[1]
    native = (getattr(block, '_native_qwen', None) if rows == 1 else
              getattr(block, '_native_qwen_mtp', {}).get(rows))
    if native is None:
        cfg = m.routing_cfg
        _gate_t(cfg)  # Same lazy router materialization as routing_std.
        if cfg.gate_i8 is None or cfg.router_bias is not None:
            return None
        native = ext.BC_QwenBlock([
            a.fn_h, a.upx_h, a.w_h, h.fn_h, h.upx_h, h.w_h,
            cfg.gate_tensor, cfg.gate_tensor_t, cfg.gate_i8, cfg.gate_sb,
            cfg.router_logits_bsz1, cfg.selected_experts_bsz1, cfg.routing_weights_bsz1,
        ], m.bc, a.rms_eps, h.rms_eps, *([rows] if rows > 1 else []))
        if rows == 1:
            block._native_qwen = native
        else:
            if not hasattr(block, '_native_qwen_mtp'):
                block._native_qwen_mtp = {}
            block._native_qwen_mtp[rows] = native
    return native
