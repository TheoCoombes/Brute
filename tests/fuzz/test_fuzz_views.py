"""Fuzz strided / view operations."""
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.fuzz


shape_strat = st.tuples(
    st.integers(min_value=1, max_value=16),
    st.integers(min_value=1, max_value=16),
)


@settings(max_examples=150, deadline=None)
@given(shape=shape_strat)
def test_transpose_value_preserved(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    assert torch.equal(bit.t().bool().as_subclass(torch.Tensor), src.t())


@settings(max_examples=150, deadline=None)
@given(shape=shape_strat, step=st.integers(min_value=1, max_value=4))
def test_strided_slice_value_preserved(shape, step):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    assert torch.equal(bit[::step].bool().as_subclass(torch.Tensor), src[::step])


@settings(max_examples=150, deadline=None)
@given(shape=shape_strat)
def test_view_reshape_matches(shape):
    n = shape[0] * shape[1]
    src = torch.randint(0, 2, (n,), dtype=torch.bool)
    bit = bit1(src)
    assert torch.equal(bit.reshape(shape).bool().as_subclass(torch.Tensor),
                       src.reshape(shape))


@settings(max_examples=100, deadline=None)
@given(
    shape=st.tuples(
        st.integers(1, 8),
        st.integers(1, 8),
        st.integers(1, 8),
    )
)
def test_permute_invariance(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    # Composition of permute and its inverse is identity.
    out = bit.permute(2, 0, 1).permute(1, 2, 0)
    assert torch.equal(out.bool().as_subclass(torch.Tensor), src)
