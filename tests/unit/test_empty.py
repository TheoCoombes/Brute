"""Empty-tensor parity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


EMPTY_SHAPES = [
    (0,),
    (0, 0),
    (4, 0),
    (0, 4),
    (3, 0, 5),
    (0, 0, 0),
]


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_zeros_empty(shape, device):
    t = brute.zeros(*shape, dtype=brute.bit1, device=device)
    assert t.shape == shape
    assert t.numel() == 0


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_empty_popcount(shape, device):
    t = brute.zeros(*shape, dtype=brute.bit1, device=device)
    assert t.popcount().item() == 0


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_empty_bool_roundtrip(shape, device):
    t = brute.zeros(*shape, dtype=brute.bit1, device=device)
    b = t.bool()
    assert b.shape == shape
    assert b.numel() == 0


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_empty_logical_not(shape, device):
    t = brute.zeros(*shape, dtype=brute.bit1, device=device)
    out = torch.logical_not(t)
    assert out.shape == shape


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_empty_clone(shape, device):
    t = brute.zeros(*shape, dtype=brute.bit1, device=device)
    c = t.clone()
    assert c.shape == shape


def test_empty_sum_returns_zero(device):
    t = brute.zeros(0, dtype=brute.bit1, device=device)
    assert int(t.sum().item()) == 0


def test_empty_all_returns_true(device):
    t = brute.zeros(0, dtype=brute.bit1, device=device)
    assert bool(t.all().item()) is True


def test_empty_any_returns_false(device):
    t = brute.zeros(0, dtype=brute.bit1, device=device)
    assert bool(t.any().item()) is False
