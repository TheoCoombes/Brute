"""In-place ops and cache invalidation for bit1 tensors."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


def test_fill_inplace_bit1(device):
    x = bit1(torch.zeros((8,), dtype=torch.bool, device=device))
    x.fill_(True)
    expected = torch.ones((8,), dtype=torch.bool, device=device)
    assert_bit1_matches_bool(x, expected)


def test_zero_inplace(device):
    x = bit1(torch.ones((8,), dtype=torch.bool, device=device))
    x.zero_()
    expected = torch.zeros((8,), dtype=torch.bool, device=device)
    assert_bit1_matches_bool(x, expected)


def test_logical_not_inplace_bool(device):
    """logical_not_ on a plain bool tensor."""
    src = torch.tensor([True, False, True, False], device=device)
    bit = bit1(src.clone())
    base = bit.as_subclass(torch.Tensor)
    base.logical_not_()
    expected = torch.tensor([False, True, False, True], device=device)
    assert_bit1_matches_bool(bit, expected)


def test_inplace_invalidates_packed_buf(device):
    """After an inplace mutation the packed buffer must be recomputed."""
    x = bit1(torch.zeros((64,), dtype=torch.bool, device=device))
    pb_before = x._packed_buf.clone()

    x.as_subclass(torch.Tensor).fill_(True)

    pb_after = x._packed_buf
    # Packed buffer must differ after content change.
    assert not torch.equal(pb_before, pb_after)


def test_inplace_setitem_invalidates_packed_buf(device):
    x = bit1(torch.zeros((16,), dtype=torch.bool, device=device))
    pb_before = x._packed_buf.clone()
    x[0] = True
    pb_after = x._packed_buf
    assert not torch.equal(pb_before, pb_after)


def test_inplace_copy(device):
    a = bit1(torch.zeros((4,), dtype=torch.bool, device=device))
    src = torch.tensor([True, False, True, False], device=device)
    a.copy_(src)
    assert_bit1_matches_bool(a, src)


def test_inplace_add(device):
    """Inplace add on a non-bool dtype must work via the standard torch path."""
    a = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    a.add_(1)
    ref = torch.tensor([2, 3, 4], dtype=torch.int32, device=device)
    assert torch.equal(a.as_subclass(torch.Tensor), ref)


def test_inplace_does_not_change_dtype(device):
    x = bit1(torch.zeros((4,), dtype=torch.bool, device=device))
    x.fill_(True)
    assert x.dtype == brute.bit1


def test_inplace_version_increments(device):
    x = bit1(torch.zeros((4,), dtype=torch.bool, device=device))
    v0 = x._version
    x.fill_(True)
    v1 = x._version
    assert v1 > v0
