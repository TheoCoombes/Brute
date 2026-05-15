"""Aliasing / storage-sharing parity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import (
    assert_bit1_matches_bool,
    assert_different_storage,
    assert_same_storage,
)


def test_view_shares_storage(device):
    x = brute.zeros((8, 8), dtype=brute.bit1, device=device)
    v = x[:, ::2]
    assert_same_storage(x, v)


def test_transpose_shares_storage(device):
    x = brute.zeros((4, 5), dtype=brute.bit1, device=device)
    t = x.t()
    assert_same_storage(x, t)


def test_clone_distinct_storage(device):
    x = brute.zeros(4, dtype=brute.bit1, device=device)
    c = x.clone()
    assert_different_storage(x, c)


def test_contiguous_distinct_storage_after_transpose(device):
    x = brute.zeros((4, 5), dtype=brute.bit1, device=device)
    c = x.t().contiguous()
    assert_different_storage(x, c)


def test_view_alias_propagation(device):
    """Mutation through a view must propagate back to the base."""
    base = brute.zeros((8, 8), dtype=brute.bit1, device=device)
    view = base[:, ::2]

    view.fill_(True)

    expected = torch.zeros((8, 8), dtype=torch.bool, device=device)
    expected[:, ::2] = True
    assert_bit1_matches_bool(base, expected)


def test_overlapping_view_inplace(device):
    x = brute.zeros(16, dtype=brute.bit1, device=device)
    a = x[1:]
    a.fill_(True)
    expected = torch.tensor([False] + [True] * 15, device=device)
    assert_bit1_matches_bool(x, expected)


def test_mutation_through_setitem_propagates(device):
    base = brute.zeros((4, 4), dtype=brute.bit1, device=device)
    base[0, :] = True
    expected = torch.zeros((4, 4), dtype=torch.bool, device=device)
    expected[0, :] = True
    assert_bit1_matches_bool(base, expected)


def test_clone_does_not_alias_packed_buf(device):
    x = brute.zeros(16, dtype=brute.bit1, device=device)
    c = x.clone()
    c.as_subclass(torch.Tensor).fill_(True)
    assert torch.equal(
        x.bool().as_subclass(torch.Tensor),
        torch.zeros((16,), dtype=torch.bool, device=device),
    )


def test_view_dtype_preserved(device):
    x = brute.zeros((4, 4), dtype=brute.bit1, device=device)
    v = x[:, ::2]
    assert v.dtype == brute.bit1
