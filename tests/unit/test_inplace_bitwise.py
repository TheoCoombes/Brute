"""In-place bitwise op tests (`&=`, `|=`, `^=`) for bit1 tensors.

These ops now update only the packed cache and mark the bool view dirty —
correctness must still hold for every consumer (sum, popcount, tolist,
indexing, repr, ...).
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1, assert_bit1_matches_bool


SHAPES = [(64,), (3, 17), (5, 4, 7)]


# &= (IAND)

@pytest.mark.parametrize("shape", SHAPES)
def test_iand_correctness(shape, device, pack_dtype):
    a_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    a = bit1(a_src.clone(), pack_dtype=pack_dtype)
    b = bit1(b_src, pack_dtype=pack_dtype)
    a &= b
    assert_bit1_matches_bool(a, a_src & b_src)


def test_iand_dirty_state(device):
    a = bit1(torch.randint(0, 2, (32,), dtype=torch.bool, device=device))
    b = bit1(torch.randint(0, 2, (32,), dtype=torch.bool, device=device))
    a &= b
    assert a.__dict__.get('_bool_dirty', False) is True
    # Popcount fast path stays correct via the packed cache.
    assert int(a.popcount()) == int((a.bool().as_subclass(torch.Tensor)).long().sum())


# |= (IOR)

@pytest.mark.parametrize("shape", SHAPES)
def test_ior_correctness(shape, device, pack_dtype):
    a_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    a = bit1(a_src.clone(), pack_dtype=pack_dtype)
    b = bit1(b_src, pack_dtype=pack_dtype)
    a |= b
    assert_bit1_matches_bool(a, a_src | b_src)


# ^= (IXOR)

@pytest.mark.parametrize("shape", SHAPES)
def test_ixor_correctness(shape, device, pack_dtype):
    a_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b_src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    a = bit1(a_src.clone(), pack_dtype=pack_dtype)
    b = bit1(b_src, pack_dtype=pack_dtype)
    a ^= b
    assert_bit1_matches_bool(a, a_src ^ b_src)


# In-place after lazy bool (chains) 

def test_inplace_after_packed_output(device):
    """`(a & b) &= c` exercises both lazy-bool input and packed-cache update."""
    a = bit1(torch.randint(0, 2, (128,), dtype=torch.bool, device=device))
    b = bit1(torch.randint(0, 2, (128,), dtype=torch.bool, device=device))
    c = bit1(torch.randint(0, 2, (128,), dtype=torch.bool, device=device))
    out = a & b
    out &= c
    ref = a.bool().as_subclass(torch.Tensor) \
          & b.bool().as_subclass(torch.Tensor) \
          & c.bool().as_subclass(torch.Tensor)
    assert_bit1_matches_bool(out, ref)


def test_inplace_returns_self(device):
    a = bit1(torch.randint(0, 2, (32,), dtype=torch.bool, device=device))
    b = bit1(torch.randint(0, 2, (32,), dtype=torch.bool, device=device))
    result = a.__iand__(b)
    assert result is a


# Scalar mixed: bit1 & bool — falls back to bool path 

def test_bit1_and_bool_tensor_fallback(device):
    """When 'other' is not bit1, the fast path must be skipped (no false promote)."""
    a = bit1(torch.tensor([True, True, False, False], device=device))
    b = torch.tensor([True, False, True, False], device=device)  # plain torch.bool
    out = a & b
    # Per the dispatch isolation rule, the result stays bool (mixed input).
    assert getattr(out, '_is_bit1', False) is False
    expected = torch.tensor([True, False, False, False], device=device)
    assert torch.equal(out.as_subclass(torch.Tensor), expected)
