"""Unary-op parity with torch.bool / torch.Tensor."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


@pytest.mark.parametrize("shape", [(4,), (3, 5), (2, 3, 4)])
def test_logical_not_matches_bool(shape, device):
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    bit = bit1(src)
    out = torch.logical_not(bit)
    ref = torch.logical_not(src)
    assert_bit1_matches_bool(out, ref)


def test_bitwise_not_matches_bool(device):
    src = torch.tensor([True, False, True, False], device=device)
    bit = bit1(src)
    out = ~bit
    ref = ~src
    assert_bit1_matches_bool(out, ref)


def test_invert_operator(device):
    src = torch.tensor([True, False], device=device)
    bit = bit1(src)
    out = bit.bitwise_not()
    ref = src.bitwise_not()
    assert_bit1_matches_bool(out, ref)


def test_clone_preserves_values(device):
    src = torch.tensor([True, False, True], device=device)
    bit = bit1(src)
    c = bit.clone()
    assert_bit1_matches_bool(c, src)


def test_contiguous_preserves_values(device):
    src = torch.randint(0, 2, (4, 5), dtype=torch.bool, device=device)
    bit = bit1(src).t()
    c = bit.contiguous()
    assert_bit1_matches_bool(c, src.t().contiguous())


def test_neg_on_int(device):
    a = brute.tensor([1, -2, 3], dtype=torch.int32, device=device)
    out = -a
    ref = -torch.tensor([1, -2, 3], dtype=torch.int32, device=device)
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_abs_on_int(device):
    a = brute.tensor([-1, -2, 3], dtype=torch.int32, device=device)
    out = torch.abs(a)
    ref = torch.abs(torch.tensor([-1, -2, 3], dtype=torch.int32, device=device))
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_sign(device):
    a = brute.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device=device)
    out = torch.sign(a)
    ref = torch.sign(torch.tensor([-1.0, 0.0, 1.0], device=device))
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_sqrt(device):
    a = brute.tensor([1.0, 4.0, 9.0], dtype=torch.float32, device=device)
    out = torch.sqrt(a)
    assert torch.allclose(out.as_subclass(torch.Tensor),
                          torch.tensor([1.0, 2.0, 3.0], device=device))


def test_to_float_preserves_value(device):
    src = torch.tensor([True, False, True], device=device)
    bit = bit1(src)
    f = bit.float()
    ref = src.float()
    assert torch.equal(f.as_subclass(torch.Tensor), ref)


def test_to_int_preserves_value(device):
    src = torch.tensor([True, False, True], device=device)
    bit = bit1(src)
    i = bit.int()
    ref = src.int()
    assert torch.equal(i.as_subclass(torch.Tensor), ref)


def test_to_long_preserves_value(device):
    src = torch.tensor([True, False, True], device=device)
    bit = bit1(src)
    l = bit.long()
    ref = src.long()
    assert torch.equal(l.as_subclass(torch.Tensor), ref)
