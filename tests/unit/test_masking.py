"""Masking and where / masked_select / masked_fill parity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


def test_masked_fill_bit1_into_self(device):
    src = torch.zeros((4,), dtype=torch.bool, device=device)
    bit = bit1(src.clone())
    mask = torch.tensor([True, False, True, False], device=device)
    bit.masked_fill_(mask, True)
    ref = torch.zeros((4,), dtype=torch.bool, device=device).masked_fill_(mask, True)
    assert_bit1_matches_bool(bit, ref)


def test_masked_fill_returns_new(device):
    """masked_fill on a bit1 with a plain bool mask returns bool (dtype isolation)."""
    base = torch.zeros((4,), dtype=torch.bool, device=device)
    bit = bit1(base)
    mask = torch.tensor([True, False, True, False], device=device)
    out = bit.masked_fill(mask, True)
    ref = base.masked_fill(mask, True)
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_masked_select(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    mask = torch.tensor([True, False, True, True, False], device=device)
    out = torch.masked_select(bit, mask)
    ref = torch.masked_select(src, mask)
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_where_with_bit1_condition(device):
    cond_bool = torch.tensor([True, False, True], device=device)
    cond = bit1(cond_bool)
    a = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    b = brute.tensor([10, 20, 30], dtype=torch.int32, device=device)
    out = torch.where(cond, a, b)
    ref = torch.where(cond_bool,
                      torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
                      torch.tensor([10, 20, 30], dtype=torch.int32, device=device))
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_bool_mask_indexing(device):
    src = torch.randint(0, 2, (4, 5), dtype=torch.bool, device=device)
    bit = bit1(src)
    mask = torch.tensor([True, False, True, False], device=device)
    out = bit[mask]
    ref = src[mask]
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_masked_scatter(device):
    base = torch.zeros((4,), dtype=torch.bool, device=device)
    bit = bit1(base.clone())
    mask = torch.tensor([True, False, True, False], device=device)
    src = torch.tensor([True, True], device=device)
    bit.masked_scatter_(mask, src)
    ref = base.clone().masked_scatter_(mask, src)
    assert_bit1_matches_bool(bit, ref)
