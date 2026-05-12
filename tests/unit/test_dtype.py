"""Dtype identity, equality, conversions, and promotion."""
from __future__ import annotations

import pytest
import torch

import brute
from brute.dtype import _Bit1DType
from tests.conftest import ALL_DTYPES
from tests.helpers import assert_bit1_matches_bool, assert_dtype, bit1


def test_bit1_dtype_singleton():
    a = brute.bit1
    b = brute.bit1
    assert a is b or a == b
    assert isinstance(a, _Bit1DType)


def test_bit1_not_equal_to_torch_bool():
    """Critical invariant: bit1 != torch.bool in both directions."""
    assert brute.bit1 != torch.bool
    assert torch.bool != brute.bit1


def test_brute_bool_is_torch_bool():
    assert brute.bool is torch.bool


def test_bit1_hashable():
    d = {brute.bit1: "bit1"}
    assert d[brute.bit1] == "bit1"


def test_bit1_repr():
    assert "bit1" in repr(brute.bit1)


@pytest.mark.parametrize("dtype", [
    torch.float32, torch.float64, torch.float16, torch.bfloat16,
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.uint8, torch.uint16, torch.uint32, torch.uint64,
    torch.bool, torch.complex64, torch.complex128,
])
def test_brute_reexports_torch_dtype(dtype):
    """Every standard torch dtype must be re-exported from brute."""
    name = str(dtype).removeprefix("torch.")
    assert getattr(brute, name) is dtype


def test_bit1_to_bool_roundtrip(device):
    x = bit1(torch.tensor([True, False, True, False], device=device))
    as_bool = x.bool()
    assert as_bool.dtype == torch.bool
    back = bit1(as_bool)
    assert_bit1_matches_bool(back, as_bool.as_subclass(torch.Tensor))


def test_to_dtype_bit1_to_bool(device):
    x = bit1(torch.tensor([True, False], device=device))
    as_bool = x.to(torch.bool)
    assert as_bool.dtype == torch.bool


def test_to_dtype_bool_to_bit1(device):
    x = brute.tensor([True, False, True], dtype=torch.bool, device=device)
    out = x.to(brute.bit1)
    assert out.dtype == brute.bit1


def test_to_dtype_int_to_bit1(device):
    x = brute.tensor([0, 1, 0, 1, 1], dtype=torch.int32, device=device)
    out = x.to(brute.bit1)
    assert out.dtype == brute.bit1
    expected = torch.tensor([False, True, False, True, True], device=device)
    assert_bit1_matches_bool(out, expected)


def test_to_dtype_float_to_bit1(device):
    x = brute.tensor([-1.0, 0.0, 0.5, 1.0], dtype=torch.float32, device=device)
    out = x.to(brute.bit1)
    # >0 → True for floats (per brute._to_bool convention)
    expected = torch.tensor([False, False, True, True], device=device)
    assert_bit1_matches_bool(out, expected)


@pytest.mark.parametrize("lhs", [torch.bool, torch.int32, torch.float32])
@pytest.mark.parametrize("rhs", [torch.bool, torch.int32, torch.float32])
def test_dtype_promotion_arithmetic(lhs, rhs, device):
    """Promotion rules should match torch."""
    a = brute.tensor([[1, 0], [0, 1]], dtype=lhs, device=device)
    b = brute.tensor([[1, 1], [1, 0]], dtype=rhs, device=device)
    ref_a = a.as_subclass(torch.Tensor)
    ref_b = b.as_subclass(torch.Tensor)
    out = a + b
    ref = ref_a + ref_b
    assert out.dtype == ref.dtype
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_bit1_arithmetic_promotes(device):
    """bit1 + int32 should promote (matching torch.bool semantics)."""
    a = bit1(torch.tensor([True, False, True], device=device))
    b = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    out = a + b
    ref = torch.tensor([True, False, True], device=device).int() + torch.tensor([1, 2, 3], device=device, dtype=torch.int32)
    assert out.dtype == ref.dtype
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_dtype_attribute_stable_across_views(device):
    x = bit1(torch.zeros((4, 4), dtype=torch.bool, device=device))
    assert x.dtype == brute.bit1
    # Views should preserve dtype.
    v = x[:, ::2]
    assert v.dtype == brute.bit1
    t = x.t()
    assert t.dtype == brute.bit1
