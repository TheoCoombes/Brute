"""Property-based fuzz: pack→unpack round-trip and shape preservation."""
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.fuzz


# Bounded shapes — keep generation small so the suite stays fast.
nonempty_shape = st.lists(
    st.integers(min_value=1, max_value=16),
    min_size=1,
    max_size=4,
).map(tuple)

any_shape = st.lists(
    st.integers(min_value=0, max_value=16),
    min_size=0,
    max_size=4,
).map(tuple)


@settings(max_examples=200, deadline=None)
@given(shape=nonempty_shape)
def test_pack_unpack_roundtrip(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    assert torch.equal(bit.bool().as_subclass(torch.Tensor), src)


@settings(max_examples=200, deadline=None)
@given(shape=nonempty_shape)
def test_popcount_matches_sum(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    expected = int(src.long().sum().item())
    assert int(bit.popcount().item()) == expected


@settings(max_examples=100, deadline=None)
@given(shape=any_shape)
def test_empty_handled(shape):
    src = torch.zeros(shape, dtype=torch.bool)
    bit = bit1(src)
    assert bit.shape == shape


@settings(max_examples=200, deadline=None)
@given(shape=nonempty_shape)
def test_logical_not_inverts(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    assert torch.equal(torch.logical_not(bit).bool().as_subclass(torch.Tensor),
                       torch.logical_not(src))
