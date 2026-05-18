"""Unit tests for the multi-scale binary positional embedding."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn import BinaryPositionEmbedding, make_binary_positions


def test_shape_and_dtype():
    p = make_binary_positions(c=32, d=64)
    assert p.shape == (32, 64)
    assert p.dtype == brute.bit1


def test_position_0_all_true():
    """Convention: (0 >> any) & 1 == 0 → True for every channel at i=0."""
    p = make_binary_positions(c=8, d=64)
    assert p[0].bool().all().item()


def test_locality_adjacent_positions_mostly_agree():
    """Adjacent positions should differ in only the lowest-period bit group."""
    c, d = 128, 64
    p = make_binary_positions(c, d)
    pm = p.bool().to(torch.int32)
    # log2(C)=7, chans_per_group = 64 // 7 = 9 → adjacent positions differ
    # exactly in those ~9 channels keyed off bit 0.
    h01 = (pm[0] ^ pm[1]).sum().item()
    assert 1 <= h01 <= 18, f"adjacent hamming should be small; got {h01}/{d}"


def test_module_buffer_registered():
    pos = BinaryPositionEmbedding(max_context=64, dim=32)
    assert hasattr(pos, "positions")
    assert pos.positions.dtype == brute.bit1
    assert pos.positions.shape == (64, 32)


def test_module_slices_seq_len():
    pos = BinaryPositionEmbedding(max_context=64, dim=32)
    out = pos(16)
    assert out.shape == (16, 32)
    assert getattr(out, "_is_bit1", False)


def test_module_rejects_oversized_seq():
    pos = BinaryPositionEmbedding(max_context=16, dim=32)
    with pytest.raises(ValueError, match="exceeds max_context"):
        pos(17)


def test_distinct_positions_are_distinct():
    pos = make_binary_positions(c=16, d=64)
    pm = pos.bool().to(torch.int32)
    # No two positions should be byte-for-byte identical.
    for i in range(16):
        for j in range(i + 1, 16):
            ham = (pm[i] ^ pm[j]).sum().item()
            assert ham > 0, f"positions {i} and {j} are identical"
