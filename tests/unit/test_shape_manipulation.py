"""Shape manipulation ops: reshape, flatten, permute, squeeze, etc."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


def test_reshape(device):
    src = torch.randint(0, 2, (6,), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.reshape(2, 3)
    ref = src.reshape(2, 3)
    assert_bit1_matches_bool(out, ref)


def test_flatten(device):
    src = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.flatten()
    ref = src.flatten()
    assert_bit1_matches_bool(out, ref)


def test_squeeze(device):
    src = torch.randint(0, 2, (1, 4, 1, 3), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit.squeeze(), src.squeeze())
    assert_bit1_matches_bool(bit.squeeze(0), src.squeeze(0))


def test_unsqueeze(device):
    src = torch.randint(0, 2, (4, 3), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit.unsqueeze(0), src.unsqueeze(0))
    assert_bit1_matches_bool(bit.unsqueeze(-1), src.unsqueeze(-1))


def test_permute(device):
    src = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit.permute(2, 0, 1), src.permute(2, 0, 1))


def test_transpose(device):
    src = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit.transpose(0, 1), src.transpose(0, 1))


def test_t(device):
    src = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit.t(), src.t())


def test_movedim(device):
    src = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(torch.movedim(bit, 0, 2), torch.movedim(src, 0, 2))


def test_swapaxes(device):
    src = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(torch.swapaxes(bit, 0, 2), torch.swapaxes(src, 0, 2))


def test_atleast_1d_2d_3d(device):
    s = torch.tensor(True, device=device)
    bit = bit1(s)
    assert torch.atleast_1d(bit).ndim == 1
    assert torch.atleast_2d(bit).ndim == 2
    assert torch.atleast_3d(bit).ndim == 3


def test_narrow(device):
    src = torch.randint(0, 2, (8,), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(torch.narrow(bit, 0, 2, 4), torch.narrow(src, 0, 2, 4))


def test_select(device):
    src = torch.randint(0, 2, (4, 5), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(torch.select(bit, 0, 1), torch.select(src, 0, 1))
