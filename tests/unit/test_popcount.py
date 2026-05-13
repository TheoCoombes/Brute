"""Popcount validation."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.helpers.reference_ops import ref_packed_popcount


@pytest.mark.parametrize("shape", [(0,), (1,), (8,), (16, 32), (3, 4, 5)])
def test_popcount_matches_sum_of_bool(shape, device):
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    bit = bit1(src)
    if src.numel() == 0:
        assert bit.popcount().item() == 0
        return
    expected = int(src.long().sum().item())
    assert int(bit.popcount().item()) == expected


def test_popcount_all_zero(device):
    bit = bit1(torch.zeros((128,), dtype=torch.bool, device=device))
    assert bit.popcount().item() == 0


def test_popcount_all_one(device):
    bit = bit1(torch.ones((128,), dtype=torch.bool, device=device))
    assert bit.popcount().item() == 128


def test_popcount_non_byte_aligned(device):
    """A length that isn't a multiple of 8/32/64 still counts correctly."""
    src = torch.tensor([True] * 5 + [False] * 6, device=device)  # 5 ones, 11 elements
    bit = bit1(src)
    assert bit.popcount().item() == 5


def test_popcount_scalar(device):
    bit = bit1(torch.tensor(True, device=device))
    assert bit.popcount().item() == 1


def test_popcount_on_torch_bool_input(device):
    """popcount() on a plain bool brute.Tensor should also work."""
    t = brute.tensor([True, False, True, True], dtype=torch.bool, device=device)
    assert t.popcount().item() == 3


def test_popcount_raises_on_non_bool_non_bit1(device):
    t = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    with pytest.raises(TypeError):
        t.popcount()


@pytest.mark.parametrize("pack_dtype", [torch.uint8, torch.uint32, torch.uint64])
def test_popcount_pack_dtype_invariant(pack_dtype, device):
    src = torch.randint(0, 2, (64,), dtype=torch.bool, device=device)
    bit = bit1(src, pack_dtype=pack_dtype)
    expected = int(src.long().sum().item())
    assert int(bit.popcount().item()) == expected


def test_packed_popcount_op(device):
    """popcount() returns the scalar total number of set bits."""
    src = torch.tensor([True, False, True, True, False, True], device=device)
    bit = bit1(src)
    assert int(bit.popcount().item()) == 4


def test_per_element_popcount_op(device):
    """word_popcount() returns per-packed-word popcount."""
    src = torch.tensor([True, False, True, True, False, True, True, False], device=device)
    bit = bit1(src, pack_dtype=torch.uint8)
    out = bit.word_popcount()
    # One uint8 word with 5 set bits: 1+0+1+1+0+1+1+0 = 5
    assert int(out.flatten()[0].item()) == 5


def test_python_reference_matches_kernel(device):
    src = torch.randint(0, 2, (128,), dtype=torch.bool, device=device)
    bit = bit1(src)
    fast = bit.popcount().cpu()
    slow = ref_packed_popcount(bit._packed_buf.cpu())
    assert int(fast.item()) == int(slow.item())
