"""Binary-op parity with torch.bool."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


SHAPES = [(4,), (3, 5), (2, 3, 4)]


@pytest.mark.parametrize("shape", SHAPES)
def test_logical_and(shape, device):
    a = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    ref = a & b
    out = bit1(a) & bit1(b)
    assert_bit1_matches_bool(out, ref)


@pytest.mark.parametrize("shape", SHAPES)
def test_logical_or(shape, device):
    a = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    ref = a | b
    out = bit1(a) | bit1(b)
    assert_bit1_matches_bool(out, ref)


@pytest.mark.parametrize("shape", SHAPES)
def test_logical_xor(shape, device):
    a = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    ref = a ^ b
    out = bit1(a) ^ bit1(b)
    assert_bit1_matches_bool(out, ref)


def test_bitwise_and_function(device):
    a = torch.tensor([True, True, False, False], device=device)
    b = torch.tensor([True, False, True, False], device=device)
    out = torch.bitwise_and(bit1(a), bit1(b))
    ref = torch.bitwise_and(a, b)
    assert_bit1_matches_bool(out, ref)


def test_bitwise_or_function(device):
    a = torch.tensor([True, True, False, False], device=device)
    b = torch.tensor([True, False, True, False], device=device)
    out = torch.bitwise_or(bit1(a), bit1(b))
    ref = torch.bitwise_or(a, b)
    assert_bit1_matches_bool(out, ref)


def test_bitwise_xor_function(device):
    a = torch.tensor([True, True, False, False], device=device)
    b = torch.tensor([True, False, True, False], device=device)
    out = torch.bitwise_xor(bit1(a), bit1(b))
    ref = torch.bitwise_xor(a, b)
    assert_bit1_matches_bool(out, ref)


def test_eq(device):
    a = torch.tensor([True, False, True], device=device)
    b = torch.tensor([True, True, False], device=device)
    out = torch.eq(bit1(a), bit1(b))
    ref = torch.eq(a, b)
    # eq across bool inputs returns bool; with bit1 inputs result should match the bool result.
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_ne(device):
    a = torch.tensor([True, False, True], device=device)
    b = torch.tensor([True, True, False], device=device)
    out = torch.ne(bit1(a), bit1(b))
    ref = torch.ne(a, b)
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_lt_le_gt_ge_on_int(device):
    a = brute.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    b = brute.tensor([2, 2, 2, 2], dtype=torch.int32, device=device)
    ref_a = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)
    ref_b = torch.tensor([2, 2, 2, 2], dtype=torch.int32, device=device)
    for op in [torch.lt, torch.le, torch.gt, torch.ge]:
        out = op(a, b)
        ref = op(ref_a, ref_b)
        assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_add_on_int(device):
    a = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    b = brute.tensor([4, 5, 6], dtype=torch.int32, device=device)
    out = a + b
    ref = torch.tensor([5, 7, 9], dtype=torch.int32, device=device)
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_sub_mul_div_on_float(device):
    a = brute.tensor([2.0, 4.0, 6.0], dtype=torch.float32, device=device)
    b = brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    assert torch.allclose((a - b).as_subclass(torch.Tensor),
                          torch.tensor([1.0, 2.0, 3.0], device=device))
    assert torch.allclose((a * b).as_subclass(torch.Tensor),
                          torch.tensor([2.0, 8.0, 18.0], device=device))
    assert torch.allclose((a / b).as_subclass(torch.Tensor),
                          torch.tensor([2.0, 2.0, 2.0], device=device))


def test_pow(device):
    a = brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    out = a ** 2
    ref = torch.tensor([1.0, 4.0, 9.0], device=device)
    assert torch.allclose(out.as_subclass(torch.Tensor), ref)


def test_where(device):
    cond = bit1(torch.tensor([True, False, True], device=device))
    a = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    b = brute.tensor([10, 20, 30], dtype=torch.int32, device=device)
    out = torch.where(cond, a, b)
    ref = torch.where(
        torch.tensor([True, False, True], device=device),
        torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
        torch.tensor([10, 20, 30], dtype=torch.int32, device=device),
    )
    assert torch.equal(out.as_subclass(torch.Tensor), ref)
