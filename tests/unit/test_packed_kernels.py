from __future__ import annotations

import pytest
import torch

import brute
from brute.tensor import Tensor as BruteTensor


@pytest.mark.parametrize("shape", [(3, 17), (2, 5, 96)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_pack_sign_matches_threshold(shape, dtype, device):
    x = torch.randint(-5, 6, shape, dtype=torch.int32, device=device)
    if dtype.is_floating_point:
        x = x.to(dtype) + 0.25
    else:
        x = x.to(dtype)
    packed = torch.ops.brute.pack_sign(x)
    got = BruteTensor._make_bit1_from_packed(packed, list(shape))
    assert torch.equal(got.bool().as_subclass(torch.Tensor), x >= 0)


def _bsr_reference(q_bool, assoc_bool, shifts):
    B, n, D = q_bool.shape
    groups = len(shifts)
    A = torch.zeros(B, D, dtype=torch.int32, device=q_bool.device)
    read = torch.zeros_like(q_bool)
    state = torch.zeros_like(q_bool)
    gate = torch.zeros_like(q_bool)
    shift_by_dim = torch.empty(D, dtype=torch.int32, device=q_bool.device)
    idx = (torch.arange(D, device=q_bool.device) * groups) // D
    for g, s in enumerate(shifts):
        shift_by_dim[idx == g] = int(s)

    for t in range(n):
        state_t = A >= 0
        gate_t = state_t != assoc_bool[:, t]
        read[:, t] = q_bool[:, t] == state_t
        state[:, t] = state_t
        gate[:, t] = gate_t
        decayed = A.clone()
        for s in sorted(set(int(x) for x in shifts)):
            if s > 0:
                mask = shift_by_dim == s
                decayed[:, mask] = A[:, mask] - (A[:, mask] >> s)
        update = torch.where(gate_t, torch.where(assoc_bool[:, t], 1, -1), 0).to(torch.int32)
        A = decayed + update
    return read, state, gate


@pytest.mark.parametrize("D", [31, 64, 97])
def test_bsr_scan_matches_reference(D, device):
    B, n = 2, 9
    q_bool = torch.randint(0, 2, (B, n, D), dtype=torch.bool, device=device)
    assoc_bool = torch.randint(0, 2, (B, n, D), dtype=torch.bool, device=device)
    q = brute.as_tensor(q_bool, dtype=brute.bit1)
    assoc = brute.as_tensor(assoc_bool, dtype=brute.bit1)
    shifts = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)

    read_p, state_p, gate_p = torch.ops.brute.bsr_scan(
        q._packed_buf, assoc._packed_buf, shifts, D)
    read = BruteTensor._make_bit1_from_packed(read_p, [B, n, D])
    state = BruteTensor._make_bit1_from_packed(state_p, [B, n, D])
    gate = BruteTensor._make_bit1_from_packed(gate_p, [B, n, D])
    ref_read, ref_state, ref_gate = _bsr_reference(q_bool, assoc_bool, [1, 2, 0])

    assert torch.equal(read.bool().as_subclass(torch.Tensor), ref_read)
    assert torch.equal(state.bool().as_subclass(torch.Tensor), ref_state)
    assert torch.equal(gate.bool().as_subclass(torch.Tensor), ref_gate)
