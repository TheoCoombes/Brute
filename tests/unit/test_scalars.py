"""0-dim / scalar tensor parity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


def test_scalar_bit1(device):
    x = bit1(torch.tensor(True, device=device))
    assert x.ndim == 0
    assert bool(x.item()) is True


def test_scalar_bit1_false(device):
    x = bit1(torch.tensor(False, device=device))
    assert bool(x.item()) is False


def test_scalar_logical_not(device):
    x = bit1(torch.tensor(True, device=device))
    out = torch.logical_not(x)
    assert bool(out.item()) is False


def test_scalar_clone(device):
    x = bit1(torch.tensor(True, device=device))
    c = x.clone()
    assert bool(c.item()) is True


def test_scalar_popcount(device):
    x = bit1(torch.tensor(True, device=device))
    assert int(x.popcount().item()) == 1


def test_unsqueeze_scalar(device):
    x = bit1(torch.tensor(True, device=device))
    y = x.unsqueeze(0)
    assert y.shape == (1,)
    assert bool(y[0].item()) is True


def test_0_dim_to_python(device):
    x = bit1(torch.tensor(True, device=device))
    # bool conversion of a 0-dim tensor
    assert bool(x.item()) is True


def test_scalar_to_python_int(device):
    x = brute.tensor(5, dtype=torch.int32, device=device)
    assert int(x.item()) == 5
