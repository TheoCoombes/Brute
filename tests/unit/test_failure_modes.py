"""Negative tests — intentionally violate invariants and check error behavior."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.helpers.corruption import (
    clear_packed_buf,
    corrupt_pack_dtype,
    flip_is_bit1,
    truncate_packed_buf,
)


def test_invalid_pack_dtype():
    with pytest.raises(TypeError):
        brute.tensor([True, False], dtype=brute.bit1, pack_dtype=torch.float32)


def test_invalid_dtype_for_bit1_creation():
    """Passing a non-coercible value to dtype=bit1 should raise."""
    # Lists of dicts can't be coerced.
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        brute.tensor([{"a": 1}], dtype=brute.bit1)


def test_element_size_raises_on_bit1():
    bit = bit1(torch.tensor([True, False]))
    with pytest.raises(TypeError):
        bit.element_size()


def test_itemsize_raises_on_bit1():
    bit = bit1(torch.tensor([True, False]))
    with pytest.raises(TypeError):
        _ = bit.itemsize


def test_popcount_on_non_bool_int_raises():
    t = brute.tensor([1, 2, 3], dtype=torch.int32)
    with pytest.raises(TypeError):
        t.popcount()


def test_unpack_pm1_on_non_bit1_raises():
    t = brute.tensor([True, False], dtype=torch.bool)
    with pytest.raises(TypeError):
        t.unpack_pm1()


def test_dtype_mismatch_failed_op():
    """Operations between incompatible shapes must raise."""
    a = bit1(torch.zeros((3, 4), dtype=torch.bool))
    b = bit1(torch.zeros((5, 4), dtype=torch.bool))
    with pytest.raises((RuntimeError, ValueError)):
        _ = a & b


def test_pack_buffer_truncation_recoverable():
    """If a user manually truncates a packed buffer, the property should recompute."""
    x = bit1(torch.tensor([True, False, True, False, True, False, True, False]))
    truncate_packed_buf(x)
    # Accessing _packed_buf should recompute since the version changed when we set None,
    # or the truncated buffer should still match the bool storage we never mutated.
    pb = x._packed_buf
    assert pb is not None


def test_corrupt_pack_dtype_raises_on_op():
    """A bad pack_dtype should be caught when an op tries to use it."""
    x = bit1(torch.tensor([True, False]))
    corrupt_pack_dtype(x)
    with pytest.raises((TypeError, RuntimeError, KeyError)):
        x.unpack_pm1()


def test_clear_packed_buf_recovers():
    """Setting _packed_buf=None should trigger lazy recompute."""
    x = bit1(torch.tensor([True, False, True]))
    clear_packed_buf(x)
    pb = x._packed_buf
    assert pb is not None and pb.numel() > 0


def test_invalid_strides_raises():
    with pytest.raises((RuntimeError, ValueError)):
        bit = bit1(torch.zeros((4,), dtype=torch.bool))
        bit.as_strided((4,), (-1,))


def test_negative_size():
    with pytest.raises((RuntimeError, ValueError)):
        brute.zeros(-1, dtype=brute.bit1)


def test_invalid_shape_for_view():
    bit = bit1(torch.zeros((6,), dtype=torch.bool))
    with pytest.raises((RuntimeError, ValueError)):
        bit.view(3, 3)
