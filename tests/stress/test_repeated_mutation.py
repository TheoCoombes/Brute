"""Stress test: thousands of inplace mutations must remain consistent."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.stress


def test_repeated_inplace_logical_not():
    """An even number of logical_not_ on the same tensor must be identity."""
    x = bit1(torch.randint(0, 2, (256,), dtype=torch.bool))
    snapshot = x.bool().clone()
    for _ in range(1000):
        x.as_subclass(torch.Tensor).logical_not_()
        x.as_subclass(torch.Tensor).logical_not_()
    assert torch.equal(x.bool().as_subclass(torch.Tensor),
                       snapshot.as_subclass(torch.Tensor))


def test_repeated_fill_alternation():
    """Alternating fill_(True)/fill_(False) leaves us at a deterministic state."""
    x = bit1(torch.zeros((128,), dtype=torch.bool))
    for i in range(2000):
        x.as_subclass(torch.Tensor).fill_(bool(i % 2 == 0))
    expected_value = True  # 2000 iterations: last is i=1999, odd → False; but loop runs 0..1999, last is False
    # 2000 iters, last i=1999, i%2==0 → False, so final is False.
    expected = torch.zeros((128,), dtype=torch.bool)
    assert torch.equal(x.bool().as_subclass(torch.Tensor), expected)


def test_repeated_setitem():
    x = bit1(torch.zeros((64,), dtype=torch.bool))
    for i in range(64):
        x[i] = True
    assert x.popcount().item() == 64


def test_repeated_clone_independence():
    base = bit1(torch.zeros((128,), dtype=torch.bool))
    clones = [base.clone() for _ in range(64)]
    for i, c in enumerate(clones):
        c[i] = True
    for i, c in enumerate(clones):
        assert int(c.popcount().item()) == 1
