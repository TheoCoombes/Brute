"""Unit tests for the binary attention module."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn import BinaryAttention, alibi_slopes


def test_alibi_slopes_negative_log_spaced():
    slopes = alibi_slopes(8)
    assert slopes.shape == (8,)
    assert (slopes < 0).all(), "ALiBi slopes are negative by convention"
    # Standard ALiBi: head 0 has the largest magnitude, magnitudes shrink
    # geometrically with head index.
    mags = slopes.abs()
    assert (mags[1:] < mags[:-1]).all(), "ALiBi magnitudes should be monotone decreasing"


def test_attention_runs_and_returns_bit1():
    attn = BinaryAttention(dim=256, n_heads=4, max_context=64)
    x = brute.randint(0, 2, (2, 8, 256), dtype=brute.bit1)
    out, diag, _ = attn(x)
    assert out.shape == (2, 8, 256)
    assert getattr(out, "_is_bit1", False)
    assert diag is None


def test_attention_with_diagnostics():
    attn = BinaryAttention(dim=256, n_heads=4, max_context=64)
    x = brute.randint(0, 2, (2, 8, 256), dtype=brute.bit1)
    out, diag, _ = attn(x, return_diag=True)
    assert diag is not None
    assert "att_dec" in diag and "att_sparsity" in diag
    assert diag["att_dec"].shape == (4,)
    assert (diag["att_dec"] >= 0).all() and (diag["att_dec"] <= 1).all()
    # Sparsity = 1 - decisiveness
    assert torch.allclose(diag["att_dec"] + diag["att_sparsity"], torch.ones(4))


def test_attention_causal_mask_applied():
    """No future-token leakage: changing the LAST token's input shouldn't
    affect the FIRST token's attention output."""
    torch.manual_seed(0)
    attn = BinaryAttention(dim=256, n_heads=4, max_context=64)
    x1 = brute.randint(0, 2, (1, 8, 256), dtype=brute.bit1)
    out1, _, _ = attn(x1)
    # Flip every bit in the last position.
    x2_bool = x1.bool().clone()
    x2_bool[0, -1, :] ^= True
    x2 = brute.as_tensor(x2_bool, dtype=brute.bit1, device=x1.device)
    out2, _, _ = attn(x2)
    # The first position's output must be identical because it can't attend
    # to position 7.
    assert torch.equal(out1.bool()[0, 0], out2.bool()[0, 0]), (
        "causal mask leaked: position 0 saw position 7"
    )


def test_attention_rejects_oversized_context():
    attn = BinaryAttention(dim=256, n_heads=4, max_context=64)
    x = brute.randint(0, 2, (1, 65, 256), dtype=brute.bit1)
    with pytest.raises(ValueError, match="exceeds max_context"):
        attn(x)


def test_attention_invalid_n_heads():
    with pytest.raises(ValueError, match="not divisible"):
        BinaryAttention(dim=255, n_heads=4, max_context=64)


def test_attention_zero_band_at_tau_zero():
    """At tau=0, the only zero-band cells are scores exactly equal to 0."""
    attn = BinaryAttention(dim=256, n_heads=4, max_context=64, tau_init=0)
    x = brute.randint(0, 2, (1, 8, 256), dtype=brute.bit1)
    out, diag, _ = attn(x, return_diag=True)
    # Most cells should be decisive (very low sparsity at tau=0).
    assert (diag["att_sparsity"] < 0.5).all(), (
        f"sparsity too high at tau=0: {diag['att_sparsity'].tolist()}"
    )


def test_stage2_exact_matches_ternary_matmul():
    """The packed (Y1 - Y2)/2 trick must match a ground-truth ternary
    int8 matmul for any (A_pos, A_neg, V) where A_pos and A_neg are
    disjoint bit1 masks. Requires C and d_h to be multiples of 64
    (pack-aligned)."""
    import torch
    from brute.nn.binary_attention import stage2_aggregate

    torch.manual_seed(0)
    C, dh = 64, 64
    a_pos = brute.randint(0, 2, (C, C), dtype=brute.bit1)
    a_neg_raw = brute.randint(0, 2, (C, C), dtype=brute.bit1)
    a_neg = brute.bitwise_and(a_neg_raw, brute.bitwise_not(a_pos))
    v = brute.randint(0, 2, (C, dh), dtype=brute.bit1)

    # Ground truth: ternary int matmul.
    ap_int = a_pos.bool().to(torch.int32)
    an_int = a_neg.bool().to(torch.int32)
    A_pm1 = ap_int - an_int
    V_pm1 = v.bool().to(torch.int32) * 2 - 1
    Y_ref = A_pm1 @ V_pm1

    a_active = brute.bitwise_or(a_pos, a_neg)
    Y_fast = stage2_aggregate(a_active, a_pos, v)
    assert torch.equal(Y_fast, Y_ref), "stage-2 (Y1-Y2)/2 trick mismatch"


def test_tau_feedback_moves_threshold():
    """tau_feedback_step should raise tau when att_dec exceeds target."""
    attn = BinaryAttention(dim=256, n_heads=4, max_context=64, tau_init=0,
                            att_dec_target=0.5, tau_accum_threshold=4)
    # Pretend forward saw fully-decisive heads — att_dec = 1.0 for all heads.
    att_dec = brute.tensor([1.0, 1.0, 1.0, 1.0])
    # Each step the accumulator gains ~500; after 1 step it crosses 4 → tick.
    attn.tau_feedback_step(att_dec)
    assert (attn.tau > 0).all(), f"tau should rise; got {attn.tau.tolist()}"
