"""Stress test: thousands of inplace mutations must remain consistent."""
from __future__ import annotations

import pytest
import torch

import brute

pytestmark = pytest.mark.stress


def test_repeated_inplace_logical_not():
    """An even number of logical_not_ on the same tensor must be identity."""
    x = brute.randint(0, 2, (256,), dtype=brute.bit1)
    snapshot = x.bool().clone()
    for _ in range(1000):
        x.as_subclass(torch.Tensor).logical_not_()
        x.as_subclass(torch.Tensor).logical_not_()
    assert torch.equal(x.bool().as_subclass(torch.Tensor),
                       snapshot.as_subclass(torch.Tensor))


def test_repeated_fill_alternation():
    """Alternating fill_(True)/fill_(False) leaves us at a deterministic state."""
    x = brute.zeros(128, dtype=brute.bit1)
    for i in range(2000):
        x.as_subclass(torch.Tensor).fill_(bool(i % 2 == 0))
    expected = torch.zeros((128,), dtype=torch.bool)
    assert torch.equal(x.bool().as_subclass(torch.Tensor), expected)


def test_repeated_setitem():
    x = brute.zeros(64, dtype=brute.bit1)
    for i in range(64):
        x[i] = True
    assert x.popcount().item() == 64


def test_repeated_clone_independence():
    base = brute.zeros(128, dtype=brute.bit1)
    clones = [base.clone() for _ in range(64)]
    for i, c in enumerate(clones):
        c[i] = True
    for i, c in enumerate(clones):
        assert int(c.popcount().item()) == 1
