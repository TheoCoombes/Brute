"""Tests for :class:`layers.BooleanLinear`, :class:`layers.DiagBind`, and core
:mod:`bep` primitives (BepParam, linear_backward, signed_batch_sum, combine_desired,
mux, pm1_int, random_bit_param).

Every test verifies an exact algebraic or mechanical property, not a smoke-test:

* BooleanLinear.forward returns (sign(a·Wᵀ), z) — output equals sign_to_bit1(z)
  and z matches an independent brute.fast.matmul.
* linear_backward ΔH sign is correct: toward pm1(a*_out) ⊗ pm1(a_in).
* linear_backward returns the correct upstream desired sign(Wᵀ a*_out).
* set_active masking: all-False mask → no H update; single-row mask → only that row.
* signed_batch_sum equals Σ ±1 per coordinate, respects the active mask.
* combine_desired / mux select correctly.
* pm1_int maps True → +1, False → −1.
* DiagBind.forward = c ⊗ m; backward updates m and returns a* ⊗ m.
* BepParam: bit == sign(H); accumulate, step, state_dict round-trip, param_bytes.
* random_bit_param: balanced, right shape.
"""

from __future__ import annotations

import torch
import pytest
import brute
import bep
from bep import random_bit_param
import vsa
import layers
from layers import BooleanLinear, DiagBind


def _rand(rows: int, D: int, gen: torch.Generator) -> brute.Tensor:
    return vsa.random_hypervectors(rows, D, generator=gen)


# ── BooleanLinear.forward ────────────────────────────────────────────────────────

def test_boolean_linear_forward_output_equals_sign_z(gen):
    """BooleanLinear.forward returns (sign_to_bit1(z), z); output bit == sign(z)."""
    M, in_dim, out_dim = 6, 64, 32
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    out, z = bl.forward(x)

    assert out.dtype == brute.bit1
    assert z.dtype == torch.int32
    assert list(out.shape) == [M, out_dim]
    assert list(z.shape) == [M, out_dim]

    expected = vsa.sign_to_bit1(z)
    assert torch.equal(out.bool(), expected.bool()), (
        "BooleanLinear output bit must equal sign_to_bit1(z)"
    )


def test_boolean_linear_z_matches_manual_matmul(gen):
    """Pre-activation z matches brute.fast.matmul(a, W) independently computed."""
    M, in_dim, out_dim = 5, 128, 64
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    _, z = bl.forward(x)

    W = bl.W.bit              # (out_dim, in_dim)
    z_manual = brute.fast.matmul(x, W)   # (M, out_dim)
    assert torch.equal(z, z_manual), (
        "z from forward must match brute.fast.matmul(x, W)"
    )


def test_boolean_linear_output_dtype_shape(gen):
    """BooleanLinear output is bit1 of shape (M, out_dim), z is int32."""
    M, in_dim, out_dim = 3, 64, 16
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    out, z = bl.forward(x)
    assert out.dtype == brute.bit1
    assert z.dtype == torch.int32
    assert list(out.shape) == [M, out_dim]
    assert list(z.shape) == [M, out_dim]


# ── linear_backward: ΔH sign ────────────────────────────────────────────────────

def test_linear_backward_delta_H_sign_is_outer_product(gen):
    """ΔH sign matches pm1(a*_out)^T @ pm1(a_in): the binary outer-product update."""
    M, in_dim, out_dim = 5, 64, 32
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    bl.forward(x)  # populate cache

    a_star = _rand(M, out_dim, gen)
    H_before = bl.W.H.clone()
    bl.backward(a_star)
    dH = (bl.W.H - H_before).int()

    a_pm1 = a_star.unpack_pm1()    # (M, out)
    x_pm1 = x.unpack_pm1()         # (M, in)
    expected_dH = 2 * (a_pm1.T @ x_pm1).int()   # (out, in); factor 2 from accumulate(2*lr*dH)

    assert torch.equal(dH, expected_dH), (
        "ΔH must equal 2 * (pm1(a*_out)^T @ pm1(a_in))"
    )


def test_linear_backward_H_moves_toward_desired(gen):
    """A desired a*_out that disagrees with the current output moves H toward it.

    For a single agreeing (or disagreeing) desired, H at position (j,k) should
    move in the direction sign(a*_out[j]) * sign(a_in[k]).
    """
    M, in_dim, out_dim = 1, 8, 4
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    out, _ = bl.forward(x)

    # Inverted desired: deliberately disagrees with current output everywhere
    a_star_inv = vsa.to_bit1(~out.bool())

    H_before = bl.W.H.clone()
    bl.backward(a_star_inv)
    dH = (bl.W.H - H_before).int()

    # dH_sign should be: sign( outer(pm1(a_star_inv), pm1(x)) )
    a_pm1 = a_star_inv.unpack_pm1()
    x_pm1 = x.unpack_pm1()
    expected_sign = (a_pm1.T @ x_pm1).sign().int()
    nonzero = expected_sign != 0
    assert torch.equal(dH.sign()[nonzero], expected_sign[nonzero]), (
        "ΔH must have the same sign as pm1(a*_out)^T @ pm1(a_in)"
    )


def test_linear_backward_upstream_desired_is_sign_Wt_astar(gen):
    """linear_backward returns upstream desired a*_in = sign(Wᵀ a*_out)."""
    M, in_dim, out_dim = 6, 64, 32
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    bl.forward(x)

    a_star = _rand(M, out_dim, gen)
    x_star = bl.backward(a_star)

    # Independent reference: sign(Wᵀ a*_out)
    Wt = bl.W.bit.transpose(0, 1)         # (in_dim, out_dim)
    z_ref = brute.fast.matmul(a_star, Wt) # (M, in_dim)
    expected = vsa.sign_to_bit1(z_ref)

    assert torch.equal(x_star.bool(), expected.bool()), (
        "Upstream desired must equal sign_to_bit1(Wᵀ a*_out)"
    )


# ── set_active masking ────────────────────────────────────────────────────────────

def test_set_active_all_false_no_H_update(gen):
    """With an all-False active mask, linear_backward makes NO change to H."""
    M, in_dim, out_dim = 5, 64, 32
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    bl.forward(x)

    a_star = _rand(M, out_dim, gen)
    mask_false = torch.zeros(M, dtype=torch.bool)
    bep.set_active(mask_false)
    try:
        H_before = bl.W.H.clone()
        bl.backward(a_star)
        H_after = bl.W.H.clone()
    finally:
        bep.set_active(None)

    assert torch.equal(H_before, H_after), (
        "All-False active mask must prevent any H update"
    )


def test_set_active_single_row_only_that_row_contributes(gen):
    """With one active row, ΔH equals 2 × outer product of that single row."""
    M, in_dim, out_dim = 5, 64, 32
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    bl.forward(x)

    a_star = _rand(M, out_dim, gen)
    active_idx = 2
    mask_one = torch.zeros(M, dtype=torch.bool)
    mask_one[active_idx] = True
    bep.set_active(mask_one)
    try:
        H_before = bl.W.H.clone()
        bl.backward(a_star)
        dH = (bl.W.H - H_before).int()
    finally:
        bep.set_active(None)

    a_pm1 = a_star[active_idx].unpack_pm1().unsqueeze(0)  # (1, out)
    x_pm1 = x[active_idx].unpack_pm1().unsqueeze(0)       # (1, in)
    expected_dH = 2 * (a_pm1.T @ x_pm1).int()
    assert torch.equal(dH, expected_dH), (
        "With one active row, ΔH must equal 2 × outer product of that row"
    )


def test_set_active_none_updates_all_rows(gen):
    """After set_active(None) all rows contribute normally."""
    M, in_dim, out_dim = 5, 64, 32
    bl = BooleanLinear(in_dim, out_dim, name="bl", generator=gen)
    x = _rand(M, in_dim, gen)
    bl.forward(x)

    a_star = _rand(M, out_dim, gen)
    bep.set_active(None)
    H_before = bl.W.H.clone()
    bl.backward(a_star)
    dH = (bl.W.H - H_before).int()

    a_pm1 = a_star.unpack_pm1()
    x_pm1 = x.unpack_pm1()
    expected_dH = 2 * (a_pm1.T @ x_pm1).int()
    assert torch.equal(dH, expected_dH), (
        "With no active mask all rows must contribute to ΔH"
    )


# ── signed_batch_sum ─────────────────────────────────────────────────────────────

def test_signed_batch_sum_equals_pm1_column_sum(gen):
    """signed_batch_sum(x) equals the per-coordinate Σ ±1 over rows."""
    M, D = 8, 64
    x = _rand(M, D, gen)

    result = bep.signed_batch_sum(x)
    manual = bep.pm1_int(x, torch.int32).sum(0)

    assert torch.equal(result, manual), (
        "signed_batch_sum must equal column-wise sum of ±1 values"
    )


def test_signed_batch_sum_respects_active_mask(gen):
    """signed_batch_sum with active mask sums only the active rows."""
    M, D = 6, 64
    x = _rand(M, D, gen)

    mask = torch.zeros(M, dtype=torch.bool)
    mask[1] = True
    mask[4] = True
    bep.set_active(mask)
    try:
        result = bep.signed_batch_sum(x)
    finally:
        bep.set_active(None)

    manual = bep.pm1_int(x[mask], torch.int32).sum(0)
    assert torch.equal(result, manual), (
        "signed_batch_sum must sum only active rows when a mask is set"
    )


# ── combine_desired and mux ───────────────────────────────────────────────────────

def test_combine_desired_agree_keep_agree_disagree_use_tie(gen):
    """combine_desired: where a==b keep that value; where a!=b fall back to tie."""
    D = 64
    a = _rand(1, D, gen)
    b = _rand(1, D, gen)
    tie = _rand(1, D, gen)

    result = bep.combine_desired(a, b, tie)

    agree_mask = brute.fast.eq(a, b).bool()   # (1, D) bool
    # Positions where a == b: result must match a (== b)
    assert torch.equal(result.bool()[agree_mask], a.bool()[agree_mask]), (
        "Where a and b agree, combine_desired must keep that value"
    )
    # Positions where a != b: result must match tie
    disagree_mask = ~agree_mask
    if disagree_mask.any():
        assert torch.equal(result.bool()[disagree_mask], tie.bool()[disagree_mask]), (
            "Where a and b disagree, combine_desired must fall back to tie"
        )


def test_combine_desired_none_b_returns_a(gen):
    """combine_desired with b=None returns a unchanged."""
    D = 64
    a = _rand(3, D, gen)
    tie = _rand(3, D, gen)
    result = bep.combine_desired(a, None, tie)
    assert torch.equal(result.bool(), a.bool()), (
        "combine_desired(a, None, tie) must return a unchanged"
    )


def test_mux_per_coordinate_select(gen):
    """mux(sel, t, f): coord-wise True→t, False→f."""
    D = 128
    sel = _rand(3, D, gen)
    t_bit = _rand(3, D, gen)
    f_bit = _rand(3, D, gen)

    result = bep.mux(sel, t_bit, f_bit)

    s = sel.bool()
    t = t_bit.bool()
    f = f_bit.bool()
    expected = torch.where(s, t, f)
    assert torch.equal(result.bool(), expected), (
        "mux must select t where sel is True and f where sel is False"
    )


# ── pm1_int ───────────────────────────────────────────────────────────────────────

def test_pm1_int_true_plus1_false_minus1(gen):
    """pm1_int maps True → +1, False → −1 as int8 (default dtype)."""
    b = vsa.to_bit1(torch.tensor([True, False, True, False]))
    pm1 = bep.pm1_int(b)
    assert pm1.dtype == torch.int8
    assert pm1.tolist() == [1, -1, 1, -1], (
        "pm1_int must map True→+1, False→−1"
    )


def test_pm1_int_dtype_int32(gen):
    """pm1_int with dtype=torch.int32 returns int32."""
    b = _rand(4, 64, gen)
    pm1 = bep.pm1_int(b, torch.int32)
    assert pm1.dtype == torch.int32
    assert bool((pm1.abs() == 1).all()), "All values must be ±1"


# ── DiagBind ──────────────────────────────────────────────────────────────────────

def test_diag_bind_forward_is_xnor_with_mask(gen):
    """DiagBind.forward(c) == c ⊗ m (XNOR with the learned mask)."""
    M, D = 5, 64
    db = DiagBind(D, name="db", generator=gen)
    c = _rand(M, D, gen)
    out = db.forward(c)

    m = db.m.bit
    c_pm1 = c.unpack_pm1()
    m_pm1 = m.unpack_pm1()
    expected = vsa.to_bit1((c_pm1 * m_pm1.unsqueeze(0)) > 0)

    assert torch.equal(out.bool(), expected.bool()), (
        "DiagBind.forward must return c ⊗ m (XNOR bind)"
    )


def test_diag_bind_backward_returns_astar_bound_m(gen):
    """DiagBind.backward(a*) returns c* = a* ⊗ m (binding is its own inverse)."""
    M, D = 4, 64
    db = DiagBind(D, name="db", generator=gen)
    c = _rand(M, D, gen)
    db.forward(c)

    a_star = _rand(M, D, gen)
    c_star = db.backward(a_star)

    expected = layers.bind_mask(a_star, db.m.bit)
    assert torch.equal(c_star.bool(), expected.bool()), (
        "DiagBind.backward must return a* ⊗ m"
    )


def test_diag_bind_backward_updates_m_H(gen):
    """DiagBind.backward updates m.H (majority of c ⊗ a* over the batch)."""
    M, D = 5, 64
    db = DiagBind(D, name="db", generator=gen)
    c = _rand(M, D, gen)
    db.forward(c)

    a_star = _rand(M, D, gen)
    H_before = db.m.H.clone()
    db.backward(a_star)
    H_after = db.m.H.clone()

    assert not torch.equal(H_before, H_after), (
        "DiagBind.backward must update m.H"
    )


# ── BepParam ─────────────────────────────────────────────────────────────────────

def test_bep_param_bit_equals_sign_H(gen):
    """BepParam.bit == sign(H): H >= 0 → True (+1), H < 0 → False (-1)."""
    p = random_bit_param((8, 4), "test", generator=gen)
    expected_bit = (p.H >= 0)
    assert torch.equal(p.bit.bool(), expected_bit), (
        "BepParam.bit must equal sign(H): H>=0 → True"
    )


def test_bep_param_accumulate_changes_H(gen):
    """BepParam.accumulate(delta) adds delta to H (in-place)."""
    p = random_bit_param((8, 4), "test", generator=gen)
    H_before = p.H.clone()
    delta = torch.ones(8, 4, dtype=torch.int32)
    p.accumulate(delta)
    assert torch.equal(p.H, (H_before.to(torch.int32) + 1).clamp(-128, 127).to(torch.int8)), (
        "accumulate must add delta to H"
    )
    assert p._dirty, "accumulate must mark _dirty=True"


def test_bep_param_accumulate_saturates_int8(gen):
    """BepParam.accumulate saturates H at the int8 rails instead of wrapping."""
    p = random_bit_param((4,), "test", generator=gen)
    p.H.fill_(120)
    p.accumulate(torch.full((4,), 20, dtype=torch.int32))
    assert bool((p.H == 127).all()), "positive int8 updates must saturate at 127"
    p.H.fill_(-120)
    p.accumulate(torch.full((4,), -20, dtype=torch.int32))
    assert bool((p.H == -128).all()), "negative int8 updates must saturate at -128"


def test_bep_param_step_reports_sign_flip_count(gen):
    """BepParam.step returns the number of bits whose sign flipped."""
    p = random_bit_param((16,), "test", generator=gen)
    p.step()  # initialize _prev_sign
    # Force exactly one flip by changing sign of element 0
    flipped_positive = bool(p.H[0] >= 0)
    p.H[0] = -1 if flipped_positive else 1
    n_flip = p.step()
    assert n_flip == 1, (
        f"step should report 1 sign flip, got {n_flip}"
    )


def test_bep_param_state_dict_roundtrip(gen):
    """BepParam.state_dict / load_state_dict is an exact round-trip for H."""
    p = random_bit_param((8, 4), "test", generator=gen)
    sd = p.state_dict()
    assert set(sd.keys()) >= {"H", "shape"}

    p2 = random_bit_param((8, 4), "other", generator=gen)
    p2.load_state_dict(sd)

    assert torch.equal(p.H, p2.H), "H must survive a state_dict round-trip"
    assert p2.shape == p.shape
    assert torch.equal(p.bit.bool(), p2.bit.bool()), (
        "bit derived from H must match after load_state_dict"
    )


def test_bep_param_param_bytes_is_numel(gen):
    """BepParam.param_bytes() == H.nbytes == numel (int8 = 1 byte each)."""
    p = random_bit_param((8, 16), "test", generator=gen)
    assert p.param_bytes() == p.H.nbytes, (
        "param_bytes() must equal H.nbytes"
    )
    assert p.param_bytes() == p.H.numel(), (
        "int8 H: param_bytes must equal numel"
    )


# ── random_bit_param ─────────────────────────────────────────────────────────────

def test_random_bit_param_shape_and_dtype(gen):
    """random_bit_param produces a BepParam of the correct shape."""
    shape = (16, 32)
    p = random_bit_param(shape, "rp", generator=gen)
    assert p.H.shape == torch.Size(shape)
    assert p.H.dtype == torch.int8
    assert p.bit.dtype == brute.bit1
    assert p.bit.shape == torch.Size(shape)


def test_random_bit_param_roughly_balanced(gen):
    """random_bit_param produces a roughly balanced ±1 distribution (p≈0.5)."""
    p = random_bit_param((512,), "rp", generator=gen, p_true=0.5)
    n_pos = int((p.H > 0).sum().item())
    n_neg = int((p.H < 0).sum().item())
    total = p.H.numel()
    # With 512 elements at p=0.5, expect within ±10% of 256 each
    assert abs(n_pos - total // 2) < total // 10, (
        f"Expected ~{total//2} positive H values, got {n_pos}"
    )
    assert abs(n_neg - total // 2) < total // 10, (
        f"Expected ~{total//2} negative H values, got {n_neg}"
    )
