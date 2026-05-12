"""Fuzz broadcasting parity."""
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.fuzz


@st.composite
def broadcastable_pair(draw, max_dim=8):
    """Generate two shapes that broadcast against each other."""
    ndim = draw(st.integers(min_value=1, max_value=3))
    common = [draw(st.integers(min_value=1, max_value=max_dim)) for _ in range(ndim)]
    a_shape = []
    b_shape = []
    for d in common:
        a_shape.append(draw(st.sampled_from([1, d])))
        b_shape.append(draw(st.sampled_from([1, d])))
    return tuple(a_shape), tuple(b_shape)


@settings(max_examples=200, deadline=None)
@given(shapes=broadcastable_pair())
def test_logical_and_broadcast(shapes):
    a_shape, b_shape = shapes
    a = torch.randint(0, 2, a_shape, dtype=torch.bool)
    b = torch.randint(0, 2, b_shape, dtype=torch.bool)
    out = (bit1(a) & bit1(b)).bool().as_subclass(torch.Tensor)
    ref = a & b
    assert torch.equal(out, ref)


@settings(max_examples=200, deadline=None)
@given(shapes=broadcastable_pair())
def test_logical_or_broadcast(shapes):
    a_shape, b_shape = shapes
    a = torch.randint(0, 2, a_shape, dtype=torch.bool)
    b = torch.randint(0, 2, b_shape, dtype=torch.bool)
    out = (bit1(a) | bit1(b)).bool().as_subclass(torch.Tensor)
    ref = a | b
    assert torch.equal(out, ref)


@settings(max_examples=200, deadline=None)
@given(shapes=broadcastable_pair())
def test_logical_xor_broadcast(shapes):
    a_shape, b_shape = shapes
    a = torch.randint(0, 2, a_shape, dtype=torch.bool)
    b = torch.randint(0, 2, b_shape, dtype=torch.bool)
    out = (bit1(a) ^ bit1(b)).bool().as_subclass(torch.Tensor)
    ref = a ^ b
    assert torch.equal(out, ref)
