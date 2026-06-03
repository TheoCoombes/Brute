from __future__ import annotations

import math
import pytest
import torch

import brute
import brute.fast as bfast
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


# ── xnor_popcount_matmul_sign ─────────────────────────────────────────────────

@pytest.mark.parametrize("M,N,K", [(4, 8, 64), (3, 33, 128), (7, 64, 96)])
def test_matmul_sign_matches_sign_of_matmul(M, N, K, device):
    """xnor_popcount_matmul_sign output == sign(xnor_popcount_matmul) packed."""
    torch.manual_seed(0)
    a_bool = torch.randint(0, 2, (M, K), dtype=torch.bool, device=device)
    b_bool = torch.randint(0, 2, (N, K), dtype=torch.bool, device=device)
    a = brute.as_tensor(a_bool, dtype=brute.bit1)
    b = brute.as_tensor(b_bool, dtype=brute.bit1)

    # Reference: int32 matmul → threshold → pack
    z_ref = bfast.matmul(a, b)                        # (M, N) int32
    ref   = (z_ref >= 0).bool().as_subclass(torch.Tensor)

    # Fused kernel
    out   = bfast.matmul_sign(a, b, K)
    got   = out.bool().as_subclass(torch.Tensor)

    assert out.shape == (M, N)
    assert got.dtype == torch.bool
    assert torch.equal(got, ref), f"first mismatch at {(got != ref).nonzero()[0]}"


def test_matmul_sign_pad_bits_are_zero(device):
    """Pad bits in the last packed word of each row must remain 0."""
    M, N, K = 2, 33, 64   # N=33: last word uses only 1 bit (33 % 64 = 33 bits)
    a = brute.as_tensor(torch.ones(M, K, dtype=torch.bool, device=device), dtype=brute.bit1)
    b = brute.as_tensor(torch.ones(N, K, dtype=torch.bool, device=device), dtype=brute.bit1)
    out = bfast.matmul_sign(a, b, K)
    Kp = int(out._packed_buf.shape[-1])
    last_word = out._packed_buf[:, -1]
    pad_bits = N % 64
    if pad_bits != 0:
        mask = (1 << pad_bits) - 1
        assert torch.equal(last_word & ~mask, torch.zeros_like(last_word))


# ── packed_majority ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 3, 5, 7, 15])
@pytest.mark.parametrize("D", [64, 97, 128])
def test_packed_majority_matches_pm1_vote(k, D, device):
    """packed_majority output == sign(sum of ±1 rows)."""
    torch.manual_seed(42)
    batch = 4
    rows_bool = torch.randint(0, 2, (batch, k, D), dtype=torch.bool, device=device)
    rows_bit  = brute.as_tensor(rows_bool, dtype=brute.bit1)

    # Reference: unpack to ±1, sum, sign
    rows_pm1 = rows_bit.unpack_pm1()                  # (batch, k, D) float ±1
    vote_ref  = rows_pm1.sum(dim=1)                   # (batch, D) integer-valued
    ref_bool  = (vote_ref >= 0).bool().as_subclass(torch.Tensor)

    # Fused kernel
    out = bfast.majority(rows_bit, k, D)
    got = out.bool().as_subclass(torch.Tensor)

    assert out.shape == (batch, D)
    assert torch.equal(got, ref_bool), (
        f"k={k}, D={D}: mismatch at {(got != ref_bool).nonzero()[0]}")


def test_packed_majority_k1_is_identity(device):
    """For k=1, majority is trivially the single row."""
    D, batch = 128, 3
    rows_bool = torch.randint(0, 2, (batch, 1, D), dtype=torch.bool, device=device)
    rows_bit  = brute.as_tensor(rows_bool, dtype=brute.bit1)
    out  = bfast.majority(rows_bit, 1, D)
    ref  = rows_bit[:, 0, :]                          # (batch, D) bit1
    assert torch.equal(out.bool().as_subclass(torch.Tensor),
                       ref.bool().as_subclass(torch.Tensor))


# ── episodic_causal_search ─────────────────────────────────────────────────────

def _episodic_search_reference(qc_bool, qp_bool, kc_bool, pos_bool, pay_bool, cnt, D):
    """Python reference for episodic_causal_search."""
    B  = qc_bool.shape[0]
    read_ref   = torch.zeros(B, D, dtype=torch.bool, device=qc_bool.device)
    idx_ref    = torch.full((B,), -1, dtype=torch.int32, device=qc_bool.device)
    score_ref  = torch.full((B,), -(D * 2 + 2), dtype=torch.int32, device=qc_bool.device)
    for b in range(B):
        best_score = -(D * 2 + 2)
        best_i     = -1
        for i in range(int(cnt[b])):
            c_h = int((qc_bool[b] != kc_bool[b, i]).sum())
            p_h = int((qp_bool[b] != pos_bool[b, i]).sum())
            sc  = (D - 2 * c_h) + (D - 2 * p_h)
            if sc > best_score:
                best_score = sc
                best_i     = i
        if best_i >= 0:
            read_ref[b]  = pay_bool[b, best_i]
            idx_ref[b]   = best_i
            score_ref[b] = best_score
    return read_ref, idx_ref, score_ref


@pytest.mark.parametrize("B,N,D", [(1, 5, 64), (3, 12, 96), (2, 7, 128)])
def test_episodic_causal_search_matches_reference(B, N, D, device):
    torch.manual_seed(7)
    qc_bool  = torch.randint(0, 2, (B, D),    dtype=torch.bool, device=device)
    qp_bool  = torch.randint(0, 2, (B, D),    dtype=torch.bool, device=device)
    kc_bool  = torch.randint(0, 2, (B, N, D), dtype=torch.bool, device=device)
    pos_bool = torch.randint(0, 2, (B, N, D), dtype=torch.bool, device=device)
    pay_bool = torch.randint(0, 2, (B, N, D), dtype=torch.bool, device=device)
    cnt      = torch.randint(1, N + 1, (B,),  dtype=torch.int32, device=device)

    # Pack inputs
    qc  = brute.as_tensor(qc_bool,  dtype=brute.bit1)
    qp  = brute.as_tensor(qp_bool,  dtype=brute.bit1)
    kc  = brute.as_tensor(kc_bool,  dtype=brute.bit1)
    pos = brute.as_tensor(pos_bool, dtype=brute.bit1)
    pay = brute.as_tensor(pay_bool, dtype=brute.bit1)

    read, idx, score = bfast.episodic_causal_search(
        qc, kc._packed_buf, qp, pos._packed_buf, pay._packed_buf, cnt, D)

    # Reference
    ref_read, ref_idx, ref_score = _episodic_search_reference(
        qc_bool, qp_bool, kc_bool, pos_bool, pay_bool, cnt, D)

    assert torch.equal(idx, ref_idx), f"idx mismatch: {idx} vs {ref_idx}"
    assert torch.equal(score, ref_score), f"score mismatch"
    assert torch.equal(read.bool().as_subclass(torch.Tensor), ref_read)
