"""Rearrangement / replication ops: roll, flip, tile, repeat_interleave, rot90."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


def test_roll(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    out = torch.roll(bit, 2)
    ref = torch.roll(src, 2)
    assert_bit1_matches_bool(out, ref)


def test_flip(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    out = torch.flip(bit, dims=[0])
    ref = torch.flip(src, dims=[0])
    assert_bit1_matches_bool(out, ref)


def test_fliplr_flipud(device):
    src = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(torch.fliplr(bit), torch.fliplr(src))
    assert_bit1_matches_bool(torch.flipud(bit), torch.flipud(src))


def test_rot90(device):
    src = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = torch.rot90(bit, k=1, dims=(0, 1))
    ref = torch.rot90(src, k=1, dims=(0, 1))
    assert_bit1_matches_bool(out, ref)


def test_tile(device):
    src = torch.tensor([True, False], device=device)
    bit = bit1(src)
    out = torch.tile(bit, (3,))
    ref = torch.tile(src, (3,))
    assert_bit1_matches_bool(out, ref)


def test_repeat_interleave(device):
    src = torch.tensor([True, False, True], device=device)
    bit = bit1(src)
    out = torch.repeat_interleave(bit, 2)
    ref = torch.repeat_interleave(src, 2)
    assert_bit1_matches_bool(out, ref)
