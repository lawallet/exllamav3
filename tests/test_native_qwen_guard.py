"""Eligibility tests require the Tabby Torch environment, but allocate no VRAM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import torch
from exllamav3.modules import native_qwen as n


class Owner:
    loaded_tp = False


class Input:
    ndim = 4
    shape = (1, 1, 4, 2560)
    dtype = torch.float32
    is_cuda = True
    device = torch.device('cuda:0')
    def is_contiguous(self):
        return True


class TestGuard(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.owner.modules = [SimpleNamespace(device='cpu', caps={'prefer_cpu': True}),
                              SimpleNamespace(device='cuda:0', caps={})]
        a, h = n.GatedResidual.__new__(n.GatedResidual), n.GatedResidual.__new__(n.GatedResidual)
        for site in (a, h):
            site.use_combine = True
            site.device = torch.device('cuda:0')
        m = n.BlockSparseMLP.__new__(n.BlockSparseMLP)
        for key in ('router_pre_norm', 'routed_pre_norm', 'routed_post_norm', 'shared_experts_post_norm',
                    'latent_in', 'latent_out', 'per_expert_scale', 'cpu_split_first', 'routing_device', 'shared_experts'):
            setattr(m, key, None)
        for key in ('cpu_offload', 'tp_reduce', 'alt_residual_channel'):
            setattr(m, key, False)
        m.device = torch.device('cuda:0')
        m.bc, m.is_quantized, m.router_type, m.num_slices = object(), True, 'std', 1
        m.config = SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=False))
        m.intermediate_size, m.num_local_experts, m.num_experts = 512, 512, 512
        self.block = SimpleNamespace(_native_qwen_owner=weakref.ref(self.owner),
                                     attn_hc=a, mlp_hc=h, mlp=m, attn=SimpleNamespace(device=m.device))
        for key in ('attn_norm', 'mlp_norm', 'attn_post_norm', 'mlp_post_norm', 'layer_scalar_f',
                    'attn_resid_scalar', 'mlp_resid_scalar'):
            setattr(self.block, key, None)
        self.sentinel = object()
        m.routing_cfg = SimpleNamespace(gate_i8=object(), router_bias=None)
        for key in ('fn_h', 'upx_h', 'w_h', 'rms_eps'):
            setattr(a, key, None)
            setattr(h, key, None)
        for key in ('gate_tensor', 'gate_tensor_t', 'gate_sb', 'router_logits_bsz1',
                    'selected_experts_bsz1', 'routing_weights_bsz1'):
            setattr(m.routing_cfg, key, None)

    def get(self, params=None, x=None):
        with patch.object(n, 'ext', SimpleNamespace(BC_QwenBlock=lambda *args: self.sentinel)), \
             patch.object(n, '_gate_t'):
            return n.get_native_block(self.block, x or Input(), params or {})

    def test_single_gpu_with_cpu_embedding(self):
        self.assertIs(self.get(), self.sentinel)
        self.assertIs(self.get(), self.sentinel)

    def test_tp_and_multigpu_fallback(self):
        self.owner.loaded_tp = True
        self.assertIsNone(self.get())
        self.owner.loaded_tp = False
        self.owner.modules.append(SimpleNamespace(device='cuda:1', caps={}))
        self.assertIsNone(self.get())

    def test_unsupported_params(self):
        for params in ({'capture': None}, {'quant_preserve': False}, {'recurrent_history': True},
                       {'prefill': True}, {'backend': None}, {'export_state_layers': []}):
            self.assertIsNone(self.get(params))

    def test_shapes_and_offload(self):
        for shape in ((2, 1, 4, 2560), (1, 2, 4, 2560)):
            x = Input()
            x.shape = shape
            self.assertIsNone(self.get(x=x))
        self.block.mlp.cpu_offload = True
        self.assertIsNone(self.get())

    def test_old_extension(self):
        with patch.object(n, 'ext', SimpleNamespace()):
            self.assertIsNone(n.get_native_block(self.block, Input(), {}))


if __name__ == '__main__':
    unittest.main()
