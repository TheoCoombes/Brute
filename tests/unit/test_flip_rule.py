"""Unit tests for the flip-rule optimizer (`brute.optim.FlipRule`)."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn import BruteLinear
from brute.optim import FlipRule, propagate_error


def test_register_param_and_state():
    lin = BruteLinear(in_features=64, out_features=32)
    trainer = FlipRule(t_init=0.5, total_steps=1000)
    trainer.add_param("lin", lin.weight)
    assert "lin" in trainer.layers
    ls = trainer.layers["lin"]
    assert ls.confidence.shape == (32,)
    assert ls.f == 0.0
    assert ls.mu == 1.0


def test_n_ref_default_finalized_on_first_step():
    lin1 = BruteLinear(in_features=64, out_features=32)
    lin2 = BruteLinear(in_features=64, out_features=64)
    trainer = FlipRule(total_steps=1000)
    trainer.add_param("a", lin1.weight)
    trainer.add_param("b", lin2.weight)
    x = brute.randint(0, 2, (8, 64), dtype=brute.bit1)
    err = brute.randint(0, 2, (8, 32), dtype=brute.bit1)
    trainer.flip_step("a", x, err)
    # After first step, default n_ref values should be set.
    assert trainer.layers["a"].n_ref > 0
    assert trainer.layers["b"].n_ref > 0


def test_first_batch_no_flips_when_random():
    """With random err and x, vote magnitude is near noise → confidence
    never crosses → no flips on first batch."""
    torch.manual_seed(0)
    lin = BruteLinear(in_features=64, out_features=32)
    trainer = FlipRule(total_steps=1000, decisive_threshold=0.5)
    trainer.add_param("lin", lin.weight, n_ref=512.0)
    x = brute.randint(0, 2, (16, 64), dtype=brute.bit1)
    err = brute.randint(0, 2, (16, 32), dtype=brute.bit1)
    _, flips = trainer.flip_step("lin", x, err)
    assert flips == 0


def test_structured_signal_flips_against_misalignment():
    """When err and x are constant across the batch and W is misaligned with
    the desired correlation, the rule should flip some weights toward the
    aligned direction."""
    torch.manual_seed(42)
    lin = BruteLinear(in_features=32, out_features=16)
    trainer = FlipRule(t_init=0.5, total_steps=1000, decisive_threshold=0.05)
    trainer.add_param("lin", lin.weight, n_ref=512.0)

    # Strong constant signal: err = +1 everywhere, x = +1 everywhere → vote
    # is uniformly +B everywhere → flip iff current W = -1.
    B, m, n = 256, 16, 32
    err = brute.as_tensor(torch.ones(B, m, dtype=torch.bool), dtype=brute.bit1)
    x = brute.as_tensor(torch.ones(B, n, dtype=torch.bool), dtype=brute.bit1)

    # Run a few steps to build confidence and let flips happen.
    n_neg_before = (~lin.weight.bool()).sum().item()
    for _ in range(20):
        trainer.flip_step("lin", x, err)
        trainer.step_global()
    n_neg_after = (~lin.weight.bool()).sum().item()
    assert n_neg_after < n_neg_before, (
        f"flips should reduce the number of -1 weights; before={n_neg_before}, after={n_neg_after}"
    )


def test_mu_shrinks_when_thrashing():
    """If observed flip rate exceeds 2 * target, mu should multiplicatively
    shrink toward mu_min."""
    torch.manual_seed(0)
    lin = BruteLinear(in_features=64, out_features=64)
    trainer = FlipRule(
        t_init=2.0,  # large temperature → high flip rate
        target_flip_rate=0.001,  # low target so we cross 2*target easily
        mu_adjust=0.1,
        total_steps=100,
        decisive_threshold=0.05,
    )
    trainer.add_param("lin", lin.weight, n_ref=64.0)
    # Constant signal so vote is decisive and many weights flip.
    err = brute.as_tensor(torch.ones(64, 64, dtype=torch.bool), dtype=brute.bit1)
    x = brute.as_tensor(torch.ones(64, 64, dtype=torch.bool), dtype=brute.bit1)
    mu_init = trainer.layers["lin"].mu
    for _ in range(10):
        trainer.flip_step("lin", x, err)
        trainer.step_global()
    mu_final = trainer.layers["lin"].mu
    assert mu_final < mu_init, f"mu should shrink; init={mu_init}, final={mu_final}"


def test_propagate_error_returns_bit1():
    lin = BruteLinear(in_features=64, out_features=32)
    err = brute.randint(0, 2, (8, 32), dtype=brute.bit1)
    err_in = propagate_error(err, lin.weight)
    assert err_in.shape == (8, 64)
    assert getattr(err_in, "_is_bit1", False)


def test_propagate_error_signed_int():
    """The signed-int path is used for the LM-head boundary."""
    lin = BruteLinear(in_features=64, out_features=32)
    err = torch.randint(-7, 8, (8, 32), dtype=torch.int8)
    err_in = propagate_error(err, lin.weight)
    assert err_in.shape == (8, 64)
    assert getattr(err_in, "_is_bit1", False)


def test_cosine_temperature_schedule():
    trainer = FlipRule(t_init=0.5, t_final=0.01, total_steps=100)
    t0 = trainer.base_temperature()
    trainer.step_count = 100
    t_end = trainer.base_temperature()
    assert abs(t0 - 0.5) < 1e-6
    assert abs(t_end - 0.01) < 1e-6


def _vote_via_full_step(trainer, err, x, gate=None):
    """Helper: run flip_step at T=0 (no actual flips) and read vote by
    invoking the private packed-buffer kernel directly."""
    err_t_packed = err.t()._packed_buf.contiguous()
    x_t_packed = x.t()._packed_buf.contiguous()
    gate_t_packed = None
    if gate is not None:
        gate_t_packed = gate.t()._packed_buf.contiguous()
    return trainer._vote_block_bit1_packed(
        err_t_packed, x_t_packed, 0, err.shape[-1],
        err.shape[0], gate_t_packed,
    )


def test_gate_zero_masks_saturated_positions_in_vote():
    """Spec convention: gate=1 = near-boundary (include), gate=0 = saturated
    (mask out). With gate all 0 everywhere, every position is saturated →
    vote must be zero."""
    torch.manual_seed(0)
    B, m, n = 32, 8, 32
    # m and n must align to pack width for the raw-kernel path.
    m, n = 64, 64
    err = brute.randint(0, 2, (B, m), dtype=brute.bit1)
    x = brute.randint(0, 2, (B, n), dtype=brute.bit1)
    gate = brute.zeros(B, m, dtype=brute.bit1)
    lin = BruteLinear(in_features=n, out_features=m)
    trainer = FlipRule(t_init=0.0, total_steps=10)
    trainer.add_param("lin", lin.weight, n_ref=128.0)
    ungated = _vote_via_full_step(trainer, err, x, gate=None)
    fully_masked = _vote_via_full_step(trainer, err, x, gate=gate)
    assert (fully_masked == 0).all(), (
        f"fully-saturated gate must produce zero vote; "
        f"got max abs = {fully_masked.abs().max().item()}"
    )
    assert ungated.abs().max().item() > 0


def test_gate_all_one_passes_through_to_ungated_vote():
    """With gate=1 everywhere (every position near boundary, include all),
    the gated vote equals the ungated vote."""
    torch.manual_seed(0)
    B, m, n = 32, 64, 64
    err = brute.randint(0, 2, (B, m), dtype=brute.bit1)
    x = brute.randint(0, 2, (B, n), dtype=brute.bit1)
    gate = brute.ones(B, m, dtype=brute.bit1)
    lin = BruteLinear(in_features=n, out_features=m)
    trainer = FlipRule(t_init=0.0, total_steps=10)
    trainer.add_param("lin", lin.weight, n_ref=128.0)
    ungated = _vote_via_full_step(trainer, err, x, gate=None)
    gated = _vote_via_full_step(trainer, err, x, gate=gate)
    assert torch.equal(ungated, gated)


def test_propagate_error_with_gate_zero_zeroes_score():
    """propagate_error with gate=0 everywhere (fully saturated) → all-zero
    pre-sign score."""
    torch.manual_seed(0)
    B, m, n = 8, 32, 16
    weight = brute.randint(0, 2, (m, n), dtype=brute.bit1)
    err = brute.randint(0, 2, (B, m), dtype=brute.bit1)
    gate = brute.zeros(B, m, dtype=brute.bit1)
    score = propagate_error(err, weight, gate=gate, use_bit_balance=False)
    assert (score == 0).all()
