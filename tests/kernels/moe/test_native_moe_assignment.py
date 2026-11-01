"""Assignment ABI, bounded scratch safety and unchanged fused arithmetic."""
import math
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.native_moe import AssignmentWorkspace
from testlib.exl3 import rand_experts
from testlib.moe import routing_layout, experts_to, expert_ptr_tables, moe_buffers

pytestmark = pytest.mark.skipif(getattr(ext, "native_moe_assignment_abi", None) != 1,
                                reason="native assignment ABI 1 required")


@pytest.mark.parametrize("rows", [1, 8, 9, 12, 16])
@pytest.mark.parametrize("experts,topk", [(1, 1), (23, 22), (512, 22), (512, 32)])
@pytest.mark.parametrize("case", ["unique", "duplicate", "sentinel", "empty", "zero_weights", "permuted"])
def test_metadata(device, rows, experts, topk, case):
    g = torch.Generator().manual_seed(71)
    sel = torch.stack([torch.randperm(experts, generator=g)[:topk] for _ in range(rows)]).to(device)
    if case == "duplicate": sel.fill_(0)
    if case == "sentinel": sel[:, ::2] = -17; sel[:, 1::4] = experts + 9
    if case == "empty": sel.fill_(experts)
    w = torch.randn(sel.shape, generator=g).half().to(device)
    if case == "zero_weights": w.zero_()
    if case == "permuted": sel, w = sel.flip(0), w.flip(0)
    ws = AssignmentWorkspace(rows, topk, experts, 256, 256, device)
    ref = routing_layout(sel, w, experts)
    for _ in range(3):
        ws.output.fill_(float("nan")); ws.tables.fill_(-99); ws.inverse.fill_(-99)
        ws.prepare(sel, w)
        assert torch.equal(ws.counts, ref.expert_count)
        assert torch.equal(ws.starts, ref.expert_start)
        assert torch.equal(ws.base, ref.expert_start)
        kind = (ref.expert_count > 0).long(); kind[-1] = 0
        assert torch.equal(ws.kind, kind)
        assert torch.equal(ws.tokens, ref.token_sorted)
        assert torch.equal(ws.weights, ref.weight_sorted)
        assert torch.equal(ws.inverse, ref.inv)
        assert not torch.count_nonzero(ws.output)
    # Gather must skip invalid assignments and preserve original k order.
    contributions = torch.randn((rows * topk, 256), generator=g).to(device)
    ws.scratch[ws.inverse] = contributions
    ext.exl3_moe_gather(ws.output, ws.scratch, sel.flatten(), ws.inverse,
                        *ws.gather_tables, ws.weights)
    expected = torch.zeros_like(ws.output)
    for k in range(topk):
        expected += contributions.reshape(rows, topk, 256)[:, k] * ((sel[:, k] >= 0) & (sel[:, k] < experts))[:, None]
    assert torch.equal(ws.output, expected)


@pytest.mark.parametrize("rows", [8, 9, 12, 16])
@pytest.mark.parametrize("distribution", ["random", "identical", "duplicate", "zero_weights"])
def test_exact_grouped_experts(device, rows, distribution):
    # Match production occupancy and geometry (>=22 active experts, tile 16).
    E, K, H, I, topk = 32, 4, 1024, 1024, 22
    g = torch.Generator().manual_seed(42)
    ex = experts_to(rand_experts(E, H, I, K, g), device)
    tabs = expert_ptr_tables(ex, device)
    y = (torch.randn((rows, H), generator=g) * .05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(rows)]).to(device)
    if distribution == "identical": sel[:] = sel[0].clone()
    if distribution == "duplicate": sel[0,0] = sel[0,1]
    w = torch.rand((rows, topk), generator=g).half().to(device)
    w[:, ::3] = 0
    if distribution == "zero_weights": w.zero_()
    ws = AssignmentWorkspace(rows, topk, E, H, I, device)
    ws.prepare(sel, w)
    # Stock uses unstable argsort; compare in original assignment order.
    flat = sel.flatten(); order = flat.argsort()
    inv = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=device))
    count = torch.bincount(flat, minlength=E+1)
    assert count.max() <= 16  # Duplicate metadata is safe only with a proved bound.
    starts = count.cumsum(0) - count
    token = torch.arange(rows, device=device).repeat_interleave(topk)[order]
    weight = w.flatten()[order]
    scratch = torch.full_like(ws.scratch, float("nan"))
    output = torch.zeros_like(ws.output)
    oldbuffers = moe_buffers(H, I, 256, device)
    def launch(out, counts, tokens, weights, bufs, active, slots, base, cap):
        ext.exl3_moe(y, out, counts, tokens, weights, *bufs, 2, K, K, K,
                     *tabs["u"], *tabs["u"], *tabs["d"], False, True, False, True, False, True,
                     0., active, slots, base, 1, cap, 16)
    launch(output, count, token, weight, oldbuffers, int((count[:-1] > 0).sum()), scratch, starts, 256)
    launch(ws.output, ws.counts, ws.tokens, ws.weights, ws.buffers, -1, ws.scratch, ws.base, 16)
    assert torch.equal(scratch[inv], ws.scratch[ws.inverse]), "per-assignment expert arithmetic changed"
    kind = (count > 0).long(); kind[-1] = 0
    ext.exl3_moe_gather(output, scratch, flat, inv, starts[:E], starts[:E], kind[:E], weight)
    ext.exl3_moe_gather(ws.output, ws.scratch, flat, ws.inverse, *ws.gather_tables, ws.weights)
    assert torch.equal(output, ws.output)


def test_reject_bad_buffers_before_write(device):
    sel = torch.zeros((12, 22), dtype=torch.long, device=device)
    w = torch.ones_like(sel, dtype=torch.half)
    ws = AssignmentWorkspace(12, 22, 512, 256, 256, device)
    ws.output.fill_(123)
    args = [sel, w, ws.counts, ws.tables, ws.tokens, ws.weights, ws.inverse, ws.output]
    bad = [(0, sel.float()), (1, w.float()), (2, ws.counts[:-1]),
           (3, ws.tables.t()), (4, ws.inverse), (5, w.flatten()),
           (6, ws.inverse[:-1]), (7, ws.output.half())]
    for index, value in bad:
        changed = args.copy(); changed[index] = value
        with pytest.raises(RuntimeError): ext.native_moe_assign(*changed)
        assert torch.all(ws.output == 123)
    oversized = torch.zeros((17, 22), dtype=torch.long, device=device)
    with pytest.raises(RuntimeError): ext.native_moe_assign(oversized, w, *args[2:])


def test_workspace_canaries(device):
    # Views address exact logical ranges inside larger allocations. Detect writes
    # outside either end, including the maximum A=512 and the sentinel bucket.
    for rows in (8,9,12,16,12):
        k,e=32,512
        sel=torch.zeros((rows,k),dtype=torch.long,device=device)
        w=torch.ones((rows,k),dtype=torch.half,device=device)
        storages=[];args=[]
        for shape,dtype in (((e+1,),torch.long),((3,e+1),torch.long),
                            ((rows*k,),torch.long),((rows*k,),torch.half),
                            ((rows*k,),torch.long),((rows,1024),torch.float)):
            storage=torch.full((math.prod(shape)+16,),73,dtype=dtype,device=device)
            storages.append(storage);args.append(storage[8:-8].reshape(shape))
        for expert in (0,e,-1):
            sel.fill_(expert)
            ext.native_moe_assign(sel,w,*args)
            for storage in storages:
                assert torch.all(storage[:8]==73) and torch.all(storage[-8:]==73)
