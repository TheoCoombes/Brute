"""Views and view-like operations: must mirror torch.bool exactly."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, assert_dtype, bit1
from tests.helpers.generators import VIEW_TRANSFORMS_2D, VIEW_TRANSFORMS_3D


@pytest.mark.parametrize("name,transform", VIEW_TRANSFORMS_2D)
def test_view_2d_matches_bool(name, transform, device):
    src = torch.randint(0, 2, (8, 8), dtype=torch.bool, device=device)
    bit = bit1(src)

    ref_view = transform(src)
    bit_view = transform(bit)

    assert_dtype(bit_view, brute.bit1)
    assert_bit1_matches_bool(bit_view, ref_view, msg=f"view={name}")


@pytest.mark.parametrize("name,transform", VIEW_TRANSFORMS_3D)
def test_view_3d_matches_bool(name, transform, device):
    src = torch.randint(0, 2, (4, 6, 8), dtype=torch.bool, device=device)
    bit = bit1(src)

    ref_view = transform(src)
    bit_view = transform(bit)
    assert_bit1_matches_bool(bit_view, ref_view, msg=f"view={name}")


def test_reshape_preserves_dtype(device):
    x = bit1(torch.zeros((6,), dtype=torch.bool, device=device))
    y = x.reshape(2, 3)
    assert y.shape == (2, 3)
    assert y.dtype == brute.bit1


def test_view_method(device):
    x = bit1(torch.zeros((6,), dtype=torch.bool, device=device))
    y = x.view(2, 3)
    assert y.shape == (2, 3)
    assert y.dtype == brute.bit1


def test_transpose_preserves_dtype(device):
    x = bit1(torch.zeros((4, 5), dtype=torch.bool, device=device))
    y = x.t()
    assert y.shape == (5, 4)
    assert y.dtype == brute.bit1


def test_permute_preserves_dtype(device):
    x = bit1(torch.zeros((2, 3, 4), dtype=torch.bool, device=device))
    y = x.permute(2, 0, 1)
    assert y.shape == (4, 2, 3)
    assert y.dtype == brute.bit1


def test_squeeze_unsqueeze_preserve_dtype(device):
    x = bit1(torch.zeros((1, 4, 1, 3), dtype=torch.bool, device=device))
    y = x.squeeze()
    assert y.shape == (4, 3)
    assert y.dtype == brute.bit1
    z = y.unsqueeze(0)
    assert z.shape == (1, 4, 3)
    assert z.dtype == brute.bit1


def test_flatten(device):
    x = bit1(torch.tensor([[True, False], [False, True]], device=device))
    y = x.flatten()
    assert y.shape == (4,)
    assert y.dtype == brute.bit1


def test_chained_views(device):
    src = torch.randint(0, 2, (8, 8), dtype=torch.bool, device=device)
    bit = bit1(src)
    ref = src.t()[::2].permute(1, 0)
    out = bit.t()[::2].permute(1, 0)
    assert_bit1_matches_bool(out, ref)


def test_contiguous_after_transpose(device):
    src = torch.randint(0, 2, (4, 5), dtype=torch.bool, device=device)
    bit = bit1(src)
    transposed = bit.t()
    contig = transposed.contiguous()
    assert contig.is_contiguous()
    assert_bit1_matches_bool(contig, src.t().contiguous())


def test_expand(device):
    x = bit1(torch.tensor([[True], [False]], device=device))
    expanded = x.expand(2, 4)
    assert expanded.shape == (2, 4)
    assert expanded.dtype == brute.bit1
    ref = torch.tensor([[True], [False]], device=device).expand(2, 4)
    assert_bit1_matches_bool(expanded, ref)


def test_repeat(device):
    x = bit1(torch.tensor([True, False], device=device))
    y = x.repeat(3)
    ref = torch.tensor([True, False], device=device).repeat(3)
    assert_bit1_matches_bool(y, ref)
