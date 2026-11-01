"""Opt-in, capacity proof, unsupported-mode and arena-lifetime contract."""
from types import SimpleNamespace as NS
import threading
import weakref
import gc
import pytest
import torch
from exllamav3.modules import native_moe as nm
from exllamav3.modules import block_sparse_mlp as stock


def fixture_module(device):
    table = torch.zeros(512, dtype=torch.long, device=device)
    multi = NS(mul1=True, mcg=False, K=4, ptrs_trellis=table, ptrs_suh=table, ptrs_svh=table)
    return NS(config=NS(arch_string='NemotronH_Omni_Reasoning_V3', infer_params=NS(no_reconstruct=False)),
        router_type='dots', routing_fn=nm.routing_dots, num_experts=512, num_local_experts=512,
        num_experts_per_tok=22, expert_size=1024, gated=False, activation_fn_idx=2,
        latent_in=object(), latent_out=object(), cpu_offload=False, cpu_split_first=None,
        routing_device=None, routing_first=None, routing_gate=object(), tid2eid=None,
        is_quantized=True, support_quant_paths=True, fused_mode_buffers=object(),
        fused_rows=256, multi_up=multi, multi_down=multi)


def test_guards(device, monkeypatch):
    monkeypatch.setattr(nm,'enabled',True)
    monkeypatch.setattr(nm.ext,'native_moe_assignment_abi',1,raising=False)
    monkeypatch.setitem(nm._devices,device,True)
    monkeypatch.setattr(stock,'FUSED_DET',True)
    m = fixture_module(device)
    y = torch.empty((12,1024),dtype=torch.half,device=device)
    sel = torch.empty((12,22),dtype=torch.long,device=device)
    w = torch.empty((12,22),dtype=torch.half,device=device)
    assert nm.eligible(m,y,sel,w,{})
    cases = dict(router_type='sqrtsp_hash', routing_fn=lambda *x:None, num_experts=256,
                 num_local_experts=511, num_experts_per_tok=21, expert_size=4096,
                 gated=True, activation_fn_idx=0, latent_in=None, latent_out=None,
                 cpu_offload=True, cpu_split_first=256, routing_device=0, routing_first=1,
                 routing_gate=None, tid2eid=object(), is_quantized=False,
                 support_quant_paths=False, fused_mode_buffers=None, fused_rows=8, multi_up=None)
    for name,value in cases.items():
        old = getattr(m,name); setattr(m,name,value)
        assert nm.run(m,y,sel,w,{}) is None, name
        assert '_native_moe_stages' not in m.__dict__
        setattr(m,name,old)
    for key in ('backend','tp_warmup','autosplit_measure','activate_all_experts',
                'capture_qmaps','capture_sens','capture_h','calibration'):
        assert not nm.eligible(m,y,sel,w,{key:False}), key
    for r in (1,4,8,9,13,17):
        assert not nm.eligible(m,torch.empty((r,1024),dtype=torch.half,device=device),sel,w,{})
    assert not nm.eligible(m,y.float(),sel,w,{})
    assert not nm.eligible(m,y,sel.int(),w,{})
    assert not nm.eligible(m,y,sel,w.float(),{})
    assert not nm.eligible(m,y,sel,w.t(),{})
    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        assert not nm.eligible(m,y,sel,w,{})
    result=[]
    t=threading.Thread(target=lambda:result.append(nm.eligible(m,y,sel,w,{})))
    t.start();t.join();assert result==[False]
    monkeypatch.setattr(nm,'enabled',False)
    assert not nm.eligible(m,y,sel,w,{})
    monkeypatch.setattr(nm,'enabled',True)
    monkeypatch.setattr(stock,'FUSED_DET',False)
    assert not nm.eligible(m,y,sel,w,{})
    monkeypatch.setattr(stock,'FUSED_DET',True)
    m.config.arch_string='Qwen3_8MoeForCausalLM'
    assert not nm.eligible(m,y,sel,w,{})
    m.config.arch_string='NemotronH_Omni_Reasoning_V3'
    m.multi_up.mul1=False
    assert not nm.eligible(m,y,sel,w,{})
    m.multi_up.mul1=True
    m.multi_up.K=3.5
    assert not nm.eligible(m,y,sel,w,{})
    m.multi_up.K=4
    monkeypatch.setattr(nm.ext,'native_moe_assignment_abi',0)
    assert not nm.eligible(m,y,sel,w,{})


def test_arena_release(device):
    m1,m2 = NS(),NS()
    ws = nm.AssignmentWorkspace(12,22,512,1024,512,device)
    ref = weakref.ref(ws)
    key = ('test_lifetime',device)
    nm._arenas[key]=ws
    m1._native_moe_stages={key:ws};m2._native_moe_stages={key:ws}
    del ws
    nm.clear(m1);gc.collect();assert ref() is not None
    nm.clear(m2);gc.collect();assert ref() is None
    assert key not in nm._arenas
