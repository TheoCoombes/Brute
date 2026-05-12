"""Strides, contiguity, and stride-bearing views."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


@pytest.mark.parametrize("index", [
    pytest.param(lambda x: x[::2], id="step2"),
    pytest.param(lambda x: x[:, ::3], id="cols_step3"),
    pytest.param(lambda x: x.transpose(0, 1), id="transpose_01"),
    pytest.param(lambda x: x.permute(2, 0, 1), id="permute_201"),
])
def test_strided_views_match_bool(index, device):
    base = torch.randint(0, 2, (8, 8, 8), dtype=torch.bool, device=device)
    ref = index(base)
    bit_view = index(bit1(base))
    assert_bit1_matches_bool(bit_view, ref)


def test_stride_returned_correctly(device):
    x = bit1(torch.zeros((4, 8), dtype=torch.bool, device=device))
    assert x.stride() == (8, 1)


def test_is_contiguous_default(device):
    x = bit1(torch.zeros((4, 8), dtype=torch.bool, device=device))
    assert x.is_contiguous()


def test_transposed_not_contiguous(device):
    x = bit1(torch.zeros((4, 8), dtype=torch.bool, device=device))
    y = x.t()
    assert not y.is_contiguous()


def test_contiguous_returns_contiguous(device):
    x = bit1(torch.zeros((4, 8), dtype=torch.bool, device=device))
    y = x.t().contiguous()
    assert y.is_contiguous()


def test_non_contiguous_value_correctness(device):
    src = torch.randint(0, 2, (8, 8), dtype=torch.bool, device=device)
    bit = bit1(src)

    transposed = bit.t()
    assert_bit1_matches_bool(transposed, src.t())

    sliced = bit[::2, ::3]
    assert_bit1_matches_bool(sliced, src[::2, ::3])


def test_negative_stride_via_flip(device):
    src = torch.randint(0, 2, (8,), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit.flip(0), src.flip(0))


def test_unfold(device):
    src = torch.randint(0, 2, (8,), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.unfold(0, 3, 1)
    ref = src.unfold(0, 3, 1)
    assert_bit1_matches_bool(out, ref)


def test_as_strided(device):
    src = torch.randint(0, 2, (12,), dtype=torch.bool, device=device)
    bit = bit1(src)
    out = bit.as_strided((3, 4), (4, 1))
    ref = src.as_strided((3, 4), (4, 1))
    assert_bit1_matches_bool(out, ref)
