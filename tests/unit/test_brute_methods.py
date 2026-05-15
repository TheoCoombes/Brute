"""bit1-specific method tests: hamming, word_popcount, randomize_, popcount,
unpack_pm1. Each method has a packed-buffer fast path; we verify correctness
against a bool oracle.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1, assert_bit1_matches_bool


SHAPES = [(8,), (64,), (3, 17), (4, 64)]


# hamming

@pytest.mark.parametrize("shape", SHAPES)
def test_hamming_correctness(shape, device):
    # Keep torch.randint — we need the same bool data for both the bit1 and ref.
    a = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    b = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    ref = int((a ^ b).long().sum())
    got = int(bit1(a).hamming(bit1(b)))
    assert got == ref


def test_hamming_zero_when_equal(device):
    a = brute.randint(0, 2, (128,), dtype=brute.bit1, device=device)
    assert int(a.hamming(a)) == 0


def test_hamming_shape_mismatch_raises(device):
    a = brute.zeros(8, dtype=brute.bit1, device=device)
    b = brute.zeros(16, dtype=brute.bit1, device=device)
    with pytest.raises(ValueError):
        a.hamming(b)


def test_hamming_only_bit1(device):
    a = brute.zeros(8, dtype=brute.bit1, device=device)
    b = torch.zeros(8, dtype=torch.bool, device=device)
    with pytest.raises(TypeError):
        a.hamming(b)


# word_popcount

@pytest.mark.parametrize("shape", SHAPES)
def test_word_popcount_matches_total(shape, device):
    a = brute.randint(0, 2, shape, dtype=brute.bit1, device=device)
    wp = a.word_popcount()
    total = int(wp.as_subclass(torch.Tensor).long().sum())
    expected = int(a.popcount())
    assert total == expected


def test_word_popcount_only_bit1(device):
    b = torch.zeros(8, dtype=torch.bool, device=device).as_subclass(brute.Tensor)
    with pytest.raises(TypeError):
        b.word_popcount()


# randomize_

def test_randomize_changes_values(device):
    a = brute.zeros(1024, dtype=brute.bit1, device=device)
    a.randomize_()
    pc = int(a.popcount())
    assert 0 < pc < 1024


def test_randomize_pad_bits_remain_zero(device):
    """After randomize_, packed_popcount must still equal the live-bit count."""
    n = 1003
    a = brute.zeros(n, dtype=brute.bit1, device=device)
    a.randomize_()
    materialised = a.bool().as_subclass(torch.Tensor)
    expected = int(materialised.long().sum())
    assert int(a.popcount()) == expected


def test_randomize_only_bit1(device):
    b = torch.zeros(8, dtype=torch.bool, device=device).as_subclass(brute.Tensor)
    with pytest.raises(TypeError):
        b.randomize_()


# unpack_pm1

def test_unpack_pm1_values(device):
    a = brute.tensor([True, False, True, True, False], dtype=brute.bit1, device=device)
    pm = a.unpack_pm1()
    ref = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0], device=device)
    assert torch.allclose(pm.cpu(), ref.cpu())


# invert correctness (pad-safe)

@pytest.mark.parametrize("n", [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1003, 4096])
def test_invert_pad_safe(n, device, pack_dtype):
    """`~bit1` must zero pad bits so popcount(packed) == n - popcount(input)."""
    # Keep torch.randint — src is the bool oracle for assert_bit1_matches_bool.
    src = torch.randint(0, 2, (n,), dtype=torch.bool, device=device)
    a = bit1(src, pack_dtype=pack_dtype)
    inv = ~a
    assert int(inv.popcount()) == n - int(a.popcount())
    assert_bit1_matches_bool(inv, ~src)


@pytest.mark.parametrize("n", [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1003])
def test_invert_then_sum(n, device, pack_dtype):
    # Keep torch.randint — src is the bool oracle for comparison.
    src = torch.randint(0, 2, (n,), dtype=torch.bool, device=device)
    a = bit1(src, pack_dtype=pack_dtype)
    inv = ~a
    assert int(inv.sum()) == int((~src).long().sum())
