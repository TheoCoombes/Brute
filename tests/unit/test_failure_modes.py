"""Negative tests — intentionally violate invariants and check error behavior."""
from __future__ import annotations

import pytest
import torch

import brute
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
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        brute.tensor([{"a": 1}], dtype=brute.bit1)


def test_element_size_raises_on_bit1():
    bit = brute.tensor([True, False], dtype=brute.bit1)
    with pytest.raises(TypeError):
        bit.element_size()


def test_itemsize_raises_on_bit1():
    bit = brute.tensor([True, False], dtype=brute.bit1)
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
    a = brute.zeros((3, 4), dtype=brute.bit1)
    b = brute.zeros((5, 4), dtype=brute.bit1)
    with pytest.raises((RuntimeError, ValueError)):
        _ = a & b


def test_pack_buffer_truncation_recoverable():
    """If a user manually truncates a packed buffer, the property should recompute."""
    x = brute.tensor([True, False, True, False, True, False, True, False], dtype=brute.bit1)
    truncate_packed_buf(x)
    pb = x._packed_buf
    assert pb is not None


def test_corrupt_pack_dtype_raises_on_op():
    """A bad pack_dtype should be caught when an op tries to use it."""
    x = brute.tensor([True, False], dtype=brute.bit1)
    corrupt_pack_dtype(x)
    with pytest.raises((TypeError, RuntimeError, KeyError)):
        x.unpack_pm1()


def test_clear_packed_buf_recovers():
    """Setting _packed_buf=None should trigger lazy recompute."""
    x = brute.tensor([True, False, True], dtype=brute.bit1)
    clear_packed_buf(x)
    pb = x._packed_buf
    assert pb is not None and pb.numel() > 0


def test_invalid_strides_raises():
    with pytest.raises((RuntimeError, ValueError)):
        bit = brute.zeros(4, dtype=brute.bit1)
        bit.as_strided((4,), (-1,))


def test_negative_size():
    with pytest.raises((RuntimeError, ValueError)):
        brute.zeros(-1, dtype=brute.bit1)


def test_invalid_shape_for_view():
    bit = brute.zeros(6, dtype=brute.bit1)
    with pytest.raises((RuntimeError, ValueError)):
        bit.view(3, 3)
