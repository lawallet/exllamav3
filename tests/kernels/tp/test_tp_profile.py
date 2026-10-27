"""The opt-in observer must preserve reduction outputs and bounded storage."""
import threading

import pytest
import torch

from testlib.tp_native import run_ranks


def reduce_program(ctx):
    outputs = []
    # Exact small integers avoid arrival-order rounding ambiguity at three ranks.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for n in (4096, 16384, 65536, 65544, 65536 * 18):
            x = torch.full((n,), ctx.rank + 1, dtype=dtype, device=ctx.device)
            ctx.backend.all_reduce(x, True)
            outputs.append(x.cpu())
    return outputs


def test_api_lifetime_and_thread_isolation():
    from exllamav3.ext import exllamav3_ext as ext
    assert ext.tp_profile_end() == []
    for capacity in (0, 1048577):
        with pytest.raises(RuntimeError): ext.tp_profile_begin(capacity)
    ext.tp_profile_begin(2)
    with pytest.raises(RuntimeError): ext.tp_profile_begin(2)
    errors = []
    def other_thread():
        try:
            ext.tp_profile_begin(1)
            assert ext.tp_profile_end() == []
        except Exception as exc: errors.append(exc)
    t = threading.Thread(target=other_thread)
    t.start(); t.join()
    assert not errors
    assert ext.tp_profile_end() == []
    assert ext.tp_profile_dropped() == 0


@pytest.mark.parametrize('world', [2, 3])
@pytest.mark.parametrize('capacity', [2, 128])
def test_profile_preserves_collectives(devices, tmp_path, world, capacity):
    import json
    if len(devices) < world: pytest.skip('insufficient GPUs')
    path = tmp_path / 'helper.json'
    outputs = run_ranks(reduce_program, [d.index for d in devices[:world]],
                        cpu_profile=dict(path=str(path), capacity=capacity))
    for rank in outputs:
        for tensor in rank:
            assert torch.equal(tensor, torch.full_like(tensor, world*(world+1)//2))
    data = json.loads(path.read_text())
    assert len(data['rows']) == min(capacity, 16)  # 15 jobs + end marker
    assert data['dropped'] == max(16-capacity, 0)
    for row in data['rows']:
        assert row[0] <= row[1] <= row[2]
        if row[3] == 0: continue
        assert row[6] > 0 and row[1] <= row[7] <= row[2]
        assert row[11] in (0, 1, 2)
        assert row[10] == (row[3]+131071)//131072
        for d in devices[:world]: assert row[1] <= row[12+d.index] <= row[7]
