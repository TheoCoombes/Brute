"""Aliasing / storage-sharing parity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import (
    assert_bit1_matches_bool,
    assert_different_storage,
    assert_same_storage,
    bit1,
)


def test_view_shares_storage(device):
    x = bit1(torch.zeros((8, 8), dtype=torch.bool, device=device))
    v = x[:, ::2]
    assert_same_storage(x, v)


def test_transpose_shares_storage(device):
    x = bit1(torch.zeros((4, 5), dtype=torch.bool, device=device))
    t = x.t()
    assert_same_storage(x, t)


def test_clone_distinct_storage(device):
    x = bit1(torch.zeros((4,), dtype=torch.bool, device=device))
    c = x.clone()
    assert_different_storage(x, c)


def test_contiguous_distinct_storage_after_transpose(device):
    x = bit1(torch.zeros((4, 5), dtype=torch.bool, device=device))
    c = x.t().contiguous()
    assert_different_storage(x, c)


def test_view_alias_propagation(device):
    """Mutation through a view must propagate back to the base."""
    base = bit1(torch.zeros((8, 8), dtype=torch.bool, device=device))
    view = base[:, ::2]

    view.fill_(True)

    expected = torch.zeros((8, 8), dtype=torch.bool, device=device)
    expected[:, ::2] = True
    assert_bit1_matches_bool(base, expected)


def test_overlapping_view_inplace(device):
    x = bit1(torch.zeros((16,), dtype=torch.bool, device=device))
    a = x[1:]
    a.fill_(True)
    expected = torch.tensor([False] + [True] * 15, device=device)
    assert_bit1_matches_bool(x, expected)


def test_mutation_through_setitem_propagates(device):
    base = bit1(torch.zeros((4, 4), dtype=torch.bool, device=device))
    base[0, :] = True
    expected = torch.zeros((4, 4), dtype=torch.bool, device=device)
    expected[0, :] = True
    assert_bit1_matches_bool(base, expected)


def test_clone_does_not_alias_packed_buf(device):
    x = bit1(torch.zeros((16,), dtype=torch.bool, device=device))
    c = x.clone()
    # Mutate the clone — original must remain untouched.
    c.as_subclass(torch.Tensor).fill_(True)
    assert torch.equal(
        x.bool().as_subclass(torch.Tensor),
        torch.zeros((16,), dtype=torch.bool, device=device),
    )


def test_view_dtype_preserved(device):
    x = bit1(torch.zeros((4, 4), dtype=torch.bool, device=device))
    v = x[:, ::2]
    assert v.dtype == brute.bit1
