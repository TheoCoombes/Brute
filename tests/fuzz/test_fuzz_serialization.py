"""Fuzz pickle / torch.save round-trips."""
from __future__ import annotations

import io
import pickle

import pytest
import torch
from hypothesis import given, settings, strategies as st

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.fuzz


nonempty_shape = st.lists(
    st.integers(min_value=1, max_value=12),
    min_size=1,
    max_size=3,
).map(tuple)


@settings(max_examples=100, deadline=None)
@given(shape=nonempty_shape)
def test_pickle_roundtrip_preserves_values(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    y = pickle.loads(pickle.dumps(bit))
    assert y.dtype == brute.bit1
    assert torch.equal(y.bool().as_subclass(torch.Tensor), src)


@settings(max_examples=100, deadline=None)
@given(shape=nonempty_shape)
def test_torch_save_roundtrip(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    buf = io.BytesIO()
    torch.save(bit, buf)
    buf.seek(0)
    y = torch.load(buf, weights_only=False)
    assert y.dtype == brute.bit1
    assert torch.equal(y.bool().as_subclass(torch.Tensor), src)
