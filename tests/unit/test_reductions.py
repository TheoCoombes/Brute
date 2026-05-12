"""Reduction-op parity with torch.bool."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


SHAPES = [(8,), (3, 5), (2, 3, 4)]


@pytest.mark.parametrize("shape", SHAPES)
def test_sum_matches_bool(shape, device):
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.sum()
    ref = src.sum()
    assert out.item() == ref.item()


@pytest.mark.parametrize("shape", SHAPES)
def test_all_matches_bool(shape, device):
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.all()
    ref = src.all()
    assert bool(out.item()) == bool(ref.item())


@pytest.mark.parametrize("shape", SHAPES)
def test_any_matches_bool(shape, device):
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.any()
    ref = src.any()
    assert bool(out.item()) == bool(ref.item())


def test_sum_dim(device):
    src = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.sum(dim=0)
    ref = src.sum(dim=0)
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_sum_keepdim(device):
    src = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.sum(dim=1, keepdim=True)
    ref = src.sum(dim=1, keepdim=True)
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_count_nonzero(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    assert torch.count_nonzero(bit).item() == 3


def test_nonzero(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    out = torch.nonzero(bit)
    ref = torch.nonzero(src)
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_argmax_argmin(device):
    """argmax/argmin on bool inputs is unsupported in torch — verify parity of the error."""
    src = torch.tensor([False, True, False, True, False], device=device)
    bit = bit1(src)
    # torch.argmax on bool raises; bit1 must mirror.
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.argmax(src)
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.argmax(bit)


def test_argmax_argmin_on_int(device):
    """argmax/argmin works on int — the brute wrapper must preserve the result."""
    a = brute.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.int32, device=device)
    ref = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.int32, device=device)
    assert torch.argmax(a).item() == torch.argmax(ref).item()
    assert torch.argmin(a).item() == torch.argmin(ref).item()


def test_max_min(device):
    src = torch.tensor([False, True, False, True, False], device=device)
    bit = bit1(src)
    assert bool(torch.max(bit).item()) is True
    assert bool(torch.min(bit).item()) is False


def test_all_empty_returns_true(device):
    """all() over an empty tensor is True (per torch semantics)."""
    bit = bit1(torch.zeros((0,), dtype=torch.bool, device=device))
    assert bool(bit.all().item()) is True


def test_any_empty_returns_false(device):
    bit = bit1(torch.zeros((0,), dtype=torch.bool, device=device))
    assert bool(bit.any().item()) is False


def test_sum_along_each_dim(device):
    src = torch.randint(0, 2, (3, 4, 5), dtype=torch.bool, device=device)
    bit = bit1(src)
    for d in range(3):
        out = bit.sum(dim=d)
        ref = src.sum(dim=d)
        assert torch.equal(out.as_subclass(torch.Tensor), ref)
