"""Indexing parity with torch.bool."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


def test_int_indexing(device):
    src = torch.tensor([True, False, True, False], device=device)
    bit = bit1(src)
    assert bool(bit[0].item()) is True
    assert bool(bit[1].item()) is False


def test_slice_indexing(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit[1:4], src[1:4])


def test_negative_index(device):
    src = torch.tensor([True, False, True], device=device)
    bit = bit1(src)
    assert bool(bit[-1].item()) is True


def test_2d_indexing(device):
    src = torch.tensor([[True, False], [False, True]], device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit[0], src[0])
    assert_bit1_matches_bool(bit[:, 0], src[:, 0])


def test_ellipsis_index(device):
    src = torch.randint(0, 2, (3, 4, 5), dtype=torch.bool, device=device)
    bit = bit1(src)
    assert_bit1_matches_bool(bit[..., 0], src[..., 0])
    assert_bit1_matches_bool(bit[0, ...], src[0, ...])


def test_bool_mask_indexing(device):
    """bit1[bool_mask] indexes correctly. Result dtype follows the dtype
    isolation invariant (mixed bit1 + bool → bool)."""
    src = torch.tensor([True, False, True, False, True], device=device)
    mask = torch.tensor([True, False, True, True, False], device=device)
    bit = bit1(src)
    out = bit[mask]
    ref = src[mask]
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_bit1_mask_indexing(device):
    """bit1 indexed by another bit1 mask preserves bit1 (all-bit1 inputs)."""
    src = torch.tensor([True, False, True, False, True], device=device)
    mask = bit1(torch.tensor([True, False, True, True, False], device=device))
    bit = bit1(src)
    out = bit[mask]
    ref = src[torch.tensor([True, False, True, True, False], device=device)]
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_int_tensor_indexing(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    idx = torch.tensor([0, 2, 4], device=device)
    bit = bit1(src)
    out = bit[idx]
    ref = src[idx]
    assert_bit1_matches_bool(out, ref)


def test_advanced_indexing(device):
    src = torch.randint(0, 2, (4, 5), dtype=torch.bool, device=device)
    bit = bit1(src)
    rows = torch.tensor([0, 2], device=device)
    cols = torch.tensor([1, 3], device=device)
    out = bit[rows, cols]
    ref = src[rows, cols]
    assert_bit1_matches_bool(out, ref)


def test_setitem_scalar(device):
    src = torch.tensor([True, False, True, False], device=device)
    bit = bit1(src.clone())
    bit[1] = True
    expected = torch.tensor([True, True, True, False], device=device)
    assert_bit1_matches_bool(bit, expected)


def test_setitem_slice(device):
    src = torch.tensor([True, False, True, False], device=device)
    bit = bit1(src.clone())
    bit[1:3] = False
    expected = torch.tensor([True, False, False, False], device=device)
    assert_bit1_matches_bool(bit, expected)


def test_setitem_mask(device):
    src = torch.tensor([True, False, True, False], device=device)
    bit = bit1(src.clone())
    mask = torch.tensor([True, True, False, False], device=device)
    bit[mask] = False
    expected = torch.tensor([False, False, True, False], device=device)
    assert_bit1_matches_bool(bit, expected)


def test_index_put(device):
    src = torch.zeros((4, 4), dtype=torch.bool, device=device)
    bit = bit1(src.clone())
    idx = (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    bit.index_put_(idx, torch.tensor([True, True, True], device=device))
    ref = src.clone()
    ref.index_put_(idx, torch.tensor([True, True, True], device=device))
    assert_bit1_matches_bool(bit, ref)


def test_gather(device):
    src = torch.tensor([[True, False, True], [False, True, False]], device=device)
    bit = bit1(src)
    idx = torch.tensor([[0, 2], [1, 0]], device=device)
    out = torch.gather(bit, 1, idx)
    ref = torch.gather(src, 1, idx)
    assert_bit1_matches_bool(out, ref)


def test_scatter(device):
    """scatter into bool / bit1."""
    base = torch.zeros((3, 4), dtype=torch.bool, device=device)
    bit = bit1(base.clone())
    idx = torch.tensor([[0, 1, 2, 0]], device=device)
    val = torch.tensor([[True, True, True, True]], device=device)
    out = bit.scatter(0, idx, val)
    ref = base.scatter(0, idx, val)
    assert torch.equal(
        out.bool().as_subclass(torch.Tensor),
        ref,
    )


def test_item_on_scalar(device):
    bit = bit1(torch.tensor(True, device=device))
    assert bool(bit.item()) is True


def test_index_select(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    out = torch.index_select(bit, 0, torch.tensor([0, 2, 4], device=device))
    ref = torch.index_select(src, 0, torch.tensor([0, 2, 4], device=device))
    assert_bit1_matches_bool(out, ref)


def test_masked_select(device):
    src = torch.tensor([True, False, True, False, True], device=device)
    bit = bit1(src)
    mask = torch.tensor([True, False, True, True, False], device=device)
    out = torch.masked_select(bit, mask)
    ref = torch.masked_select(src, mask)
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)


def test_take(device):
    if device == "mps":
        pytest.skip("torch.take on bool is not implemented on MPS")
    src = torch.tensor([[True, False], [True, True]], device=device)
    bit = bit1(src)
    out = torch.take(bit, torch.tensor([0, 3], device=device))
    ref = torch.take(src, torch.tensor([0, 3], device=device))
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)
