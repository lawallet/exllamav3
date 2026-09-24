"""Experimental native residual/MoE orchestration; no new math or cache path."""
import torch
from ..ext import exllamav3_ext as ext
from .hyperconnections import GatedResidual
from .block_sparse_mlp import BlockSparseMLP
from .block_sparse_mlp_routing import _gate_t

def get_native_block(block, x, params):
    owner_ref = getattr(block, '_native_qwen_owner', None)
    owner = owner_ref() if owner_ref else None
    if (owner is None or owner.loaded_tp or
        not hasattr(ext, 'BC_QwenBlock') or x.ndim != 4 or tuple(x.shape[:2]) != (1, 1) or
        x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous() or
        any(k in params for k in ('capture', 'quant_preserve', 'backend', 'export_state_layers',
                                  'export_state_norm_keys', 'activate_all_experts', 'loras')) or
        any(params.get(k) for k in ('prefill', 'recurrent_history', 'tp_warmup', 'autosplit_measure', 'layer_instance'))):
        return None
    if getattr(block, '_native_qwen', None) is None:
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
    native = getattr(block, '_native_qwen', None)
    if native is None:
        cfg = m.routing_cfg
        _gate_t(cfg)  # Same lazy router materialization as routing_std.
        if cfg.gate_i8 is None or cfg.router_bias is not None:
            return None
        native = ext.BC_QwenBlock([
            a.fn_h, a.upx_h, a.w_h, h.fn_h, h.upx_h, h.w_h,
            cfg.gate_tensor, cfg.gate_tensor_t, cfg.gate_i8, cfg.gate_sb,
            cfg.router_logits_bsz1, cfg.selected_experts_bsz1, cfg.routing_weights_bsz1,
        ], m.bc, a.rms_eps, h.rms_eps)
        block._native_qwen = native
    return native
