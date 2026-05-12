"""Verify the lazy `_packed_buf` cache rebuilds correctly under mutation pressure."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.stress


def test_packed_buf_reflects_mutation():
    x = bit1(torch.zeros((64,), dtype=torch.bool))
    pb0 = x._packed_buf.clone()
    x.fill_(True)
    pb1 = x._packed_buf
    assert not torch.equal(pb0, pb1)
    x.fill_(False)
    pb2 = x._packed_buf
    assert torch.equal(pb2, pb0)  # back to all zeros


def test_packed_buf_after_long_mutation_chain():
    x = bit1(torch.zeros((64,), dtype=torch.bool))
    for i in range(500):
        x.as_subclass(torch.Tensor)[i % 64] = True
        x.as_subclass(torch.Tensor)[i % 64] = False
    # After matched flips we should be back to all zeros.
    assert x.popcount().item() == 0


def test_repeated_buf_access_uses_cache():
    x = bit1(torch.randint(0, 2, (64,), dtype=torch.bool))
    refs = [x._packed_buf for _ in range(50)]
    # Repeated reads with no mutation should return the SAME cached tensor.
    for r in refs[1:]:
        assert r is refs[0]


def test_buf_invalidated_after_setitem():
    x = bit1(torch.zeros((64,), dtype=torch.bool))
    first = x._packed_buf
    x[0] = True
    second = x._packed_buf
    assert first is not second
