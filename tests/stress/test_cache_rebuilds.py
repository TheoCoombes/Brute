"""Verify the lazy `_packed_buf` cache rebuilds correctly under mutation pressure."""
from __future__ import annotations

import pytest
import torch

import brute

pytestmark = pytest.mark.stress


def test_packed_buf_reflects_mutation():
    x = brute.zeros(64, dtype=brute.bit1)
    pb0 = x._packed_buf.clone()
    x.fill_(True)
    pb1 = x._packed_buf
    assert not torch.equal(pb0, pb1)
    x.fill_(False)
    pb2 = x._packed_buf
    assert torch.equal(pb2, pb0)


def test_packed_buf_after_long_mutation_chain():
    x = brute.zeros(64, dtype=brute.bit1)
    for i in range(500):
        x.as_subclass(torch.Tensor)[i % 64] = True
        x.as_subclass(torch.Tensor)[i % 64] = False
    assert x.popcount().item() == 0


def test_repeated_buf_access_uses_cache():
    x = brute.randint(0, 2, (64,), dtype=brute.bit1)
    refs = [x._packed_buf for _ in range(50)]
    for r in refs[1:]:
        assert r is refs[0]


def test_buf_invalidated_after_setitem():
    x = brute.zeros(64, dtype=brute.bit1)
    first = x._packed_buf
    x[0] = True
    second = x._packed_buf
    assert first is not second
