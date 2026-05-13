"""Strided/scalar indexing corner cases not covered by test_indexing.py.

Focuses on patterns that interact with the lazy-bool design and the packed
cache invalidation logic: `x[:, 0] = scalar`, `x[mask] = scalar`,
`x[range, range] = scalar`, etc.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1, assert_bit1_matches_bool


SHAPES = [(8, 16), (3, 17), (4, 5, 6)]


# x[:, 0] = 1 — strided column scalar setitem 

@pytest.mark.parametrize("shape", SHAPES)
def test_setitem_col_scalar(shape, device, pack_dtype):
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    a = bit1(src.clone(), pack_dtype=pack_dtype)
    ref = src.clone()
    a[..., 0] = 1
    ref[..., 0] = True
    assert_bit1_matches_bool(a, ref)
    # Popcount must reflect the partial overwrite (sum increases by # zeroes set to 1).
    assert int(a.popcount()) == int(ref.long().sum())


@pytest.mark.parametrize("shape", SHAPES)
def test_setitem_col_scalar_after_dirty(shape, device, pack_dtype):
    """Apply strided setitem to a tensor whose bool view is still dirty."""
    a_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    a = bit1(a_src, pack_dtype=pack_dtype)
    b = bit1(b_src, pack_dtype=pack_dtype)
    c = a & b   # dirty bool
    c[..., 0] = 0
    ref = (a_src & b_src).clone()
    ref[..., 0] = False
    assert_bit1_matches_bool(c, ref)


# x[0] = scalar — leading-axis row scalar setitem 

@pytest.mark.parametrize("shape", SHAPES)
def test_setitem_row_scalar(shape, device, pack_dtype):
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    a = bit1(src.clone(), pack_dtype=pack_dtype)
    ref = src.clone()
    a[0] = 1
    ref[0] = True
    assert_bit1_matches_bool(a, ref)


# x[mask] = scalar — mask-based scalar setitem 

def test_setitem_mask_scalar(device, pack_dtype):
    shape = (4, 5)
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    a = bit1(src.clone(), pack_dtype=pack_dtype)
    ref = src.clone()
    a[mask] = 1
    ref[mask] = True
    assert_bit1_matches_bool(a, ref)


# Reads must produce bit1 when input is bit1 

@pytest.mark.parametrize("shape", SHAPES)
def test_getitem_row_preserves_bit1(shape, device):
    a = bit1(torch.randint(0, 2, shape, dtype=torch.bool, device=device))
    r = a[0]
    assert getattr(r, "_is_bit1", False), "row index should preserve bit1 dtype"


@pytest.mark.parametrize("shape", SHAPES)
def test_getitem_col_preserves_bit1(shape, device):
    a = bit1(torch.randint(0, 2, shape, dtype=torch.bool, device=device))
    r = a[..., 0]
    assert getattr(r, "_is_bit1", False), "column index should preserve bit1 dtype"


# Edge: setitem to a different boolean scalar form 

def test_setitem_torch_scalar_tensor(device):
    a = bit1(torch.zeros((4, 8), dtype=torch.bool, device=device))
    a[..., 0] = torch.tensor(True, device=device)
    expected = torch.zeros((4, 8), dtype=torch.bool, device=device)
    expected[..., 0] = True
    assert_bit1_matches_bool(a, expected)
