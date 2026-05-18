"""Unit tests for the BruteLinear / BinaryLinear modules."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn import BruteLinear, BinaryLinear


def test_brute_linear_returns_int32():
    lin = BruteLinear(in_features=64, out_features=16)
    x = brute.randint(0, 2, (4, 64), dtype=brute.bit1)
    out = lin(x)
    assert out.shape == (4, 16)
    assert out.dtype == torch.int32


def test_brute_linear_matches_pm1_reference():
    """The bit1 matmul kernel should match plain ±1 matrix multiplication."""
    torch.manual_seed(0)
    lin = BruteLinear(in_features=64, out_features=16)
    x = brute.randint(0, 2, (8, 64), dtype=brute.bit1)
    out = lin(x)
    # Reference: x_pm1 @ W_pm1.T
    x_pm1 = (x.bool().to(torch.int32) * 2) - 1
    w_pm1 = (lin.weight.bool().to(torch.int32) * 2) - 1
    ref = x_pm1 @ w_pm1.t()
    assert torch.equal(out, ref)


def test_brute_linear_bias_added():
    lin = BruteLinear(in_features=32, out_features=8, bias=True)
    lin.bias.data.fill_(7)
    x = brute.randint(0, 2, (2, 32), dtype=brute.bit1)
    out_no_bias = BruteLinear(in_features=32, out_features=8, bias=False)
    out_no_bias.weight = lin.weight
    expected = out_no_bias(x) + 7
    assert torch.equal(lin(x), expected)


def test_brute_linear_multidim_leading():
    """Leading dims (..., K) should be preserved."""
    lin = BruteLinear(in_features=32, out_features=8)
    x = brute.randint(0, 2, (3, 5, 32), dtype=brute.bit1)
    out = lin(x)
    assert out.shape == (3, 5, 8)


def test_binary_linear_returns_bit1():
    blin = BinaryLinear(in_features=64, out_features=16)
    x = brute.randint(0, 2, (4, 64), dtype=brute.bit1)
    out, gate, pre = blin(x, return_gate=True, return_pre=True)
    assert out.shape == (4, 16)
    assert getattr(out, "_is_bit1", False)
    assert getattr(gate, "_is_bit1", False)
    assert pre.shape == (4, 16)
    assert pre.dtype == torch.int32


def test_binary_linear_balanced_output():
    """BinaryLinear should produce ~half +1 and half -1 per row (bit-balanced)."""
    blin = BinaryLinear(in_features=128, out_features=64)
    x = brute.randint(0, 2, (16, 128), dtype=brute.bit1)
    out, _, _ = blin(x)
    n_pos = out.bool().to(torch.int32).sum(dim=-1)
    # Median split gives exactly d/2 strict-positives when values are unique.
    # With ties it can drift slightly, but should stay close to d/2 = 32.
    assert (n_pos >= 24).all() and (n_pos <= 40).all(), (
        f"output not balanced: positive counts = {n_pos.tolist()}"
    )


def test_binary_linear_gate_none_when_not_requested():
    blin = BinaryLinear(in_features=32, out_features=16)
    x = brute.randint(0, 2, (2, 32), dtype=brute.bit1)
    out, gate, pre = blin(x, return_gate=False, return_pre=False)
    assert gate is None
    assert pre is None
