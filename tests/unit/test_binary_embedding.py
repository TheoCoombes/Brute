"""Unit tests for `brute.nn.BinaryEmbedding`."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn import BinaryEmbedding


def test_construction_and_shape():
    emb = BinaryEmbedding(num_embeddings=64, dim=32)
    assert emb.weight.shape == (64, 32)
    assert emb.weight.dtype == brute.bit1


def test_lookup_returns_bit1():
    emb = BinaryEmbedding(num_embeddings=64, dim=32)
    ids = torch.tensor([0, 1, 2, 5, 32])
    out = emb(ids)
    assert out.shape == (5, 32)
    assert getattr(out, "_is_bit1", False)


def test_lookup_returns_correct_rows():
    emb = BinaryEmbedding(num_embeddings=64, dim=128)
    weight_bool = emb.weight.bool().clone()
    ids = torch.tensor([3, 17, 0, 63])
    out = emb(ids)
    out_bool = out.bool()
    for k, i in enumerate(ids.tolist()):
        assert torch.equal(out_bool[k], weight_bool[i]), f"row {i} mismatch"


def test_lookup_preserves_leading_shape():
    emb = BinaryEmbedding(num_embeddings=16, dim=64)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = emb(ids)
    assert out.shape == (2, 3, 64)
    assert getattr(out, "_is_bit1", False)


def test_lookup_multidim_with_repeats():
    """Index gather should support repeats and multi-dim outputs."""
    emb = BinaryEmbedding(num_embeddings=8, dim=16)
    ids = torch.tensor([[0, 0, 1], [2, 1, 0]])
    out = emb(ids)
    assert out.shape == (2, 3, 16)
    weight_bool = emb.weight.bool()
    assert torch.equal(out.bool()[0, 0], weight_bool[0])
    assert torch.equal(out.bool()[0, 1], weight_bool[0])
    assert torch.equal(out.bool()[1, 0], weight_bool[2])
