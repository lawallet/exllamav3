"""Bounded native IR routed-MoE stage, opt-in and independent of native plans.

Only assignment preparation is new. Expert arithmetic and fixed-k-order gather
remain the stock extension's. Nemotron Omni is the first validated consumer;
the assignment ABI itself is model independent. No capture/replay in this stage.
"""
import os
import threading
import weakref

import torch

from ..ext import exllamav3_ext as ext
from .block_sparse_mlp_routing import routing_dots

enabled = os.environ.get("EXL3_NATIVE_MOE_ASSIGN", "0") == "1"
_arenas = weakref.WeakValueDictionary()
_devices = {}


def clear(module):
    """Release stage ownership before expert unload/reload (no weight references)."""
    module.__dict__.pop("_native_moe_stages", None)


def eligible(m, y, selected, weights, params):
    # Identity of the unmodified top-k router proves distinct, in-range picks:
    # at most one assignment per token per expert, hence occupancy <= R <= 16.
    # No device-to-host validation belongs on the decode path.
    if not enabled or y.ndim != 2 or y.shape[0] not in (12, 16) or y.shape[1] != 1024:
        return False
    if getattr(ext, "native_moe_assignment_abi", None) != 1:
        return False
    if threading.current_thread() is not threading.main_thread():
        return False
    if (getattr(m.config, "arch_string", None) != "NemotronH_Omni_Reasoning_V3"
        or m.router_type != "dots" or m.routing_fn is not routing_dots
        or m.num_experts != 512 or m.num_local_experts != 512
        or m.num_experts_per_tok != 22 or m.expert_size != 1024
        or m.gated or m.activation_fn_idx != 2 or m.latent_in is None
        or m.latent_out is None or m.cpu_offload or m.cpu_split_first is not None
        or m.routing_device is not None or m.routing_first not in (None, 0)
        or m.routing_gate is None or m.tid2eid is not None
        or not m.is_quantized or not m.support_quant_paths
        or m.config.infer_params.no_reconstruct or m.fused_mode_buffers is None
        or m.fused_rows < 16):
        return False
    from .block_sparse_mlp import FUSED_DET
    if not FUSED_DET or any(k in params for k in (
        "backend", "tp_warmup", "autosplit_measure", "activate_all_experts",
        "capture_qmaps", "capture_sens", "capture_h", "calibration", "reconstruct")):
        return False
    if (y.dtype != torch.half or not y.is_cuda or not y.is_contiguous()
        or selected.shape != (y.shape[0], 22) or weights.shape != selected.shape
        or selected.dtype != torch.long or weights.dtype != torch.half
        or selected.device != y.device or weights.device != y.device
        or not selected.is_contiguous() or not weights.is_contiguous()):
        return False
    for multi in (m.multi_up, m.multi_down):
        if multi is None or not multi.mul1 or multi.mcg or multi.K != 4:
            return False
        for table in (multi.ptrs_trellis, multi.ptrs_suh, multi.ptrs_svh):
            if (table is None or table.ndim != 1 or table.device != y.device or table.dtype != torch.long
                or table.numel() != 512 or not table.is_contiguous()):
                return False
    if y.device not in _devices:
        p = torch.cuda.get_device_properties(y.device)
        _devices[y.device] = (p.major == 12 and p.minor == 0
                              and p.multi_processor_count == 188
                              and "RTX PRO 6000" in p.name)
    if not _devices[y.device]:
        return False
    # Serial default-stream execution only: all consuming projection work is
    # enqueued before the next module can reuse this arena. Other threads/streams
    # and graph capture fall back before any stage allocation or mutation.
    if (torch.cuda.current_stream(y.device) != torch.cuda.default_stream(y.device)
        or torch.cuda.is_current_stream_capturing()):
        return False
    return True


class AssignmentWorkspace:
    """Fixed-capacity metadata and expert scratch; no model/cache pointers."""
    def __init__(self, rows, topk, experts, hidden, intermediate, device):
        a = rows * topk
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        self.counts = empty((experts + 1,), torch.long)
        self.tables = empty((3, experts + 1), torch.long)
        self.base, self.starts, self.kind = self.tables.unbind()
        self.gather_tables = tuple(t[:experts] for t in (self.starts, self.base, self.kind))
        self.tokens = empty((a,), torch.long)
        self.weights = empty((a,), torch.half)
        self.inverse = empty((a,), torch.long)
        self.output = empty((rows, hidden), torch.float)
        self.scratch = empty((a, hidden), torch.float)
        c = ext.exl3_moe_max_concurrency(device.index)
        self.buffers = [empty((c, 16, d), torch.half)
                        for d in (hidden, hidden, intermediate, intermediate)]

    def prepare(self, selected, weights):
        ext.native_moe_assign(selected, weights, self.counts, self.tables,
                              self.tokens, self.weights, self.inverse, self.output)


def run(m, y, selected, weights, params):
    """Return borrowed routed output, or None without mutation when unsupported.

    The caller consumes the result immediately on the same stream. Exceptions
    after dispatch propagate; retrying stock after a partial launch is unsafe.
    """
    if not eligible(m, y, selected, weights, params):
        return None
    key = (y.device, y.shape[0], 22, 512, y.shape[1], m.intermediate_size_padded)
    stages = m.__dict__.setdefault("_native_moe_stages", {})
    ws = stages.get(key)
    if ws is None:
        ws = _arenas.get(key)
        if ws is None:
            ws = AssignmentWorkspace(*key[1:], key[0])
            _arenas[key] = ws
        stages[key] = ws
    ws.prepare(selected, weights)
    up, down = m.multi_up, m.multi_down
    # With >=22 active experts and 188 SMs, known-active and unknown-active
    # stock launches both use 8 CTAs per group: preserve quantized reduction order.
    ext.exl3_moe(y, ws.output, ws.counts, ws.tokens, ws.weights, *ws.buffers,
                 m.activation_fn_idx, up.K, up.K, down.K,
                 up.ptrs_trellis, up.ptrs_suh, up.ptrs_svh,
                 up.ptrs_trellis, up.ptrs_suh, up.ptrs_svh,
                 down.ptrs_trellis, down.ptrs_suh, down.ptrs_svh,
                 up.mcg, up.mul1, up.mcg, up.mul1, down.mcg, down.mul1,
                 m.act_limit, -1, ws.scratch, ws.base, 1, 16, 16)
    ext.exl3_moe_gather(ws.output, ws.scratch, selected.view(-1), ws.inverse,
                        *ws.gather_tables, ws.weights)
    return ws.output
