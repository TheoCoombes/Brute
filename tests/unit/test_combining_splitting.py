"""Combining (cat/stack) and splitting parity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


def test_cat(device):
    a = bit1(torch.tensor([True, False], device=device))
    b = bit1(torch.tensor([True, True], device=device))
    out = torch.cat([a, b])
    ref = torch.cat([torch.tensor([True, False], device=device),
                     torch.tensor([True, True], device=device)])
    assert_bit1_matches_bool(out, ref)


def test_cat_dim(device):
    a = bit1(torch.tensor([[True, False], [False, True]], device=device))
    b = bit1(torch.tensor([[True, True], [False, False]], device=device))
    out = torch.cat([a, b], dim=0)
    ref = torch.cat([torch.tensor([[True, False], [False, True]], device=device),
                     torch.tensor([[True, True], [False, False]], device=device)],
                    dim=0)
    assert_bit1_matches_bool(out, ref)


def test_stack(device):
    a = bit1(torch.tensor([True, False], device=device))
    b = bit1(torch.tensor([True, True], device=device))
    out = torch.stack([a, b])
    ref = torch.stack([torch.tensor([True, False], device=device),
                       torch.tensor([True, True], device=device)])
    assert_bit1_matches_bool(out, ref)


def test_split(device):
    src = torch.randint(0, 2, (6,), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = torch.split(bit, 2)
    ref = torch.split(src, 2)
    assert len(out) == len(ref)
    for o, r in zip(out, ref):
        assert_bit1_matches_bool(o, r)


def test_chunk(device):
    src = torch.randint(0, 2, (6,), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = torch.chunk(bit, 3)
    ref = torch.chunk(src, 3)
    assert len(out) == len(ref)
    for o, r in zip(out, ref):
        assert_bit1_matches_bool(o, r)


def test_unbind(device):
    src = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = torch.unbind(bit, dim=0)
    ref = torch.unbind(src, dim=0)
    assert len(out) == 3
    for o, r in zip(out, ref):
        assert_bit1_matches_bool(o, r)


def test_hsplit_vsplit(device):
    src = torch.randint(0, 2, (4, 6), dtype=torch.bool, device=device)
    bit = bit1(src)
    for out, ref in zip(torch.hsplit(bit, 3), torch.hsplit(src, 3)):
        assert_bit1_matches_bool(out, ref)
    for out, ref in zip(torch.vsplit(bit, 2), torch.vsplit(src, 2)):
        assert_bit1_matches_bool(out, ref)
