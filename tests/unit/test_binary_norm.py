"""Unit tests for `brute.nn.binary_norm` (bit-balanced sign + gate flag)."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn.binary_norm import bit_balance, BitBalancedNorm


@pytest.mark.parametrize("d", [16, 32, 64, 65, 128])
def test_balanced_output_exactly_half_for_unique_values(d):
    """When all values are unique, the median split is exact ±1 balance."""
    pre = torch.randperm(d).reshape(1, d).to(torch.int32)
    out, _ = bit_balance(pre)
    bool_out = out.bool().to(torch.int32)
    n_pos = bool_out.sum().item()
    # For even d, lower median → n_pos = d/2 (and (d/2 - 1) values equal median).
    # torch.median returns lower of two middle values for even-length input.
    if d % 2 == 0:
        # Strict > median: n_pos = d/2 (only the strictly greater half is +1).
        assert n_pos == d // 2, f"expected {d//2} positives for unique values, got {n_pos}"
    else:
        assert n_pos == d // 2, f"odd d={d}: expected {d//2} positives, got {n_pos}"


def test_bitbalance_output_is_bit1():
    pre = torch.randn(4, 32) * 10
    out, _ = bit_balance(pre.to(torch.int32))
    assert getattr(out, "_is_bit1", False), "expected bit1 output"
    assert tuple(out.shape) == (4, 32)


def test_gate_flag_marks_near_boundary():
    """With small abs-magnitude values, gate flag should fire."""
    d = 32
    # Half elements near zero, half far.
    pre = torch.cat([
        torch.zeros(d // 2, dtype=torch.int32),  # near median
        torch.full((d // 2,), 100, dtype=torch.int32),  # far
    ]).reshape(1, d)
    _, gate = bit_balance(pre, nu=0.05)
    # gate = True where |centered| <= nu * d = 0.05 * 32 = 1
    gate_bool = gate.bool().to(torch.int32)
    # Near-zero elements have small magnitude → gate=1.
    assert gate_bool[0, : d // 2].sum().item() == d // 2, "all near-zero should gate True"
    # Far elements: magnitude 100 >> 1 → gate=0.
    assert gate_bool[0, d // 2 :].sum().item() == 0, "no far elements should gate"


def test_nu_zero_disables_gate():
    pre = torch.zeros(4, 8, dtype=torch.int32)
    out, gate = bit_balance(pre, nu=0.0)
    assert gate is None


def test_module_form_equivalence_to_functional():
    norm = BitBalancedNorm(nu=0.05)
    pre = torch.randint(-10, 10, (8, 64), dtype=torch.int32)
    out_m, gate_m = norm(pre, return_gate=True)
    out_f, gate_f = bit_balance(pre, nu=0.05)
    assert torch.equal(out_m.bool(), out_f.bool())
    assert torch.equal(gate_m.bool(), gate_f.bool())


def test_module_return_gate_false_skips_gate():
    norm = BitBalancedNorm(nu=0.05)
    pre = torch.randint(-10, 10, (8, 64), dtype=torch.int32)
    _, gate = norm(pre, return_gate=False)
    assert gate is None
