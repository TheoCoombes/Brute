"""Long-running randomized lifecycle test."""
from __future__ import annotations

import random

import pytest
import torch

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.stress


OPS = [
    ("logical_not", lambda t: torch.logical_not(t)),
    ("clone", lambda t: t.clone()),
    ("contiguous", lambda t: t.contiguous()),
    ("transpose", lambda t: t.t() if t.dim() == 2 else t),
    ("slice", lambda t: t[::2]),
    ("flip", lambda t: torch.flip(t, dims=[0])),
]


def test_randomized_lifecycle():
    """Apply 1000 random ops in sequence; each must yield a valid bit1 or bool tensor."""
    rng = random.Random(0)
    t = bit1(torch.randint(0, 2, (64, 64), dtype=torch.bool))
    for _ in range(1000):
        if t.numel() == 0:
            break  # avoid dead-tensor ops
        name, op = rng.choice(OPS)
        t = op(t)
        # Invariant: still a brute.Tensor.
        assert isinstance(t, brute.Tensor)
        # The dtype must remain bit1 or bool (the two states a bool-derived chain produces).
        assert t.dtype in (brute.bit1, torch.bool)


def test_randomized_logical_not_pairs():
    """A randomized sequence with paired logical_not_ must return to start."""
    rng = random.Random(123)
    initial = torch.randint(0, 2, (128,), dtype=torch.bool)
    t = bit1(initial.clone())
    pairs = 200
    for _ in range(pairs):
        t.as_subclass(torch.Tensor).logical_not_()
        t.as_subclass(torch.Tensor).logical_not_()
    assert torch.equal(t.bool().as_subclass(torch.Tensor), initial)
