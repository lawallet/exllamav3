"""The grouped atomic fallback must accumulate each routed contribution once."""
from types import SimpleNamespace as NS

import pytest
import torch

from exllamav3.modules import block_sparse_mlp as moe

pytestmark = pytest.mark.nogpu


@pytest.mark.parametrize('rows', [4, 20])
def test_atomic_grouped_dispatch_once(monkeypatch, rows):
    # Four experts keep each occupancy below the 16-row fused capacity, with
    # both the device-only and host-planned assignment branches exercised.
    x = torch.ones(1, rows, 4)
    selected = (torch.arange(rows) % 4).view(rows, 1)
    multi = NS(K=4, ptrs_trellis=None, ptrs_suh=None, ptrs_svh=None, mcg=False, mul1=True)
    layer = NS(
        alt_residual_channel=False, hidden_size=4, expert_size=4, bc=None,
        router_pre_norm=None, routed_pre_norm=None, latent_in=None, latent_out=None,
        routing_gate=object(), routing_cfg=None,
        routing_fn=lambda *args: (selected, torch.ones(rows, 1)),
        routing_device=None, routing_first=None, cpu_split_first=None, cpu_offload=False,
        intermediate_size=4, num_local_experts=4, num_experts=4, f_threshold=1,
        is_quantized=True, support_quant_paths=True, gated=False, activation_fn_idx=2,
        config=NS(infer_params=NS(no_reconstruct=False)), fused_rows=16,
        fused_mode_buffers=NS(temp_state_g=None, temp_state_u=None,
                              temp_intermediate_g=None, temp_intermediate_u=None),
        multi_up=multi, multi_down=multi, act_limit=0., mtile_ok=False,
        _batch_recon_layer=lambda y: None,
        cpu_split_combine=lambda result, *args: result,
        tp_reduce=False, shared_experts=None, shared_experts_post_norm=None,
        routed_post_norm=None, shared_gate=None,
    )
    calls = []
    def accumulate(y, out, *args):
        calls.append(1)
        out.add_(1)
    monkeypatch.setattr(moe, 'FUSED_DET', False)
    monkeypatch.setattr(moe.native_moe, 'enabled', False)
    monkeypatch.setattr(moe.ext, 'exl3_moe', accumulate)
    actual = moe.BlockSparseMLP.forward(layer, x, {})
    assert calls == [1]
    assert torch.equal(actual, torch.ones_like(actual))
