"""Lazy bool backing invariant tests.

After a packed-bitwise op, the bool view is left uninitialised (dirty) until
some operation actually needs it. Every code path that consumes bool data
MUST first call `_ensure_bool_valid`. These tests exercise the contract.
"""
from __future__ import annotations

import pickle

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool


@pytest.fixture(autouse=True)
def _deterministic_seed(seed):
    pass


# Construction invariants

def test_bitwise_output_is_dirty(device):
    a = brute.randint(0, 2, (64,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (64,), dtype=brute.bit1, device=device)
    c = a & b
    assert c.__dict__.get('_bool_dirty', False) is True, \
        "fresh packed-bitwise output should leave the bool view dirty"
    pb = c._packed_buf
    assert pb is not None
    assert pb.numel() > 0


def test_invert_is_dirty(device):
    a = brute.randint(0, 2, (100,), dtype=brute.bit1, device=device)
    c = ~a
    assert c.__dict__.get('_bool_dirty', False) is True


def test_in_place_and_marks_dirty(device):
    a = brute.randint(0, 2, (32,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (32,), dtype=brute.bit1, device=device)
    a &= b
    assert a.__dict__.get('_bool_dirty', False) is True


# Lazy materialisation triggers

def test_sum_does_not_dirty_clear(device):
    """sum() takes the packed fast path; bool stays dirty."""
    a = brute.randint(0, 2, (256,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (256,), dtype=brute.bit1, device=device)
    c = a & b
    s = int(c.sum())
    assert isinstance(s, int)
    assert c.__dict__.get('_bool_dirty', False) is True


def test_tolist_clears_dirty(device):
    a = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    c = a & b
    _ = c.tolist()
    assert c.__dict__.get('_bool_dirty', False) is False


def test_repr_clears_dirty(device):
    a = brute.randint(0, 2, (8,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (8,), dtype=brute.bit1, device=device)
    c = a & b
    _ = repr(c)
    assert c.__dict__.get('_bool_dirty', False) is False


def test_bool_method_clears_dirty(device):
    a = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    c = a & b
    _ = c.bool()
    assert c.__dict__.get('_bool_dirty', False) is False


def test_to_device_clears_dirty(device):
    a = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    c = a & b
    _ = c.to(device=device)
    assert c.__dict__.get('_bool_dirty', False) is False


def test_pickle_clears_dirty(device):
    a = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (16,), dtype=brute.bit1, device=device)
    c = a & b
    blob = pickle.dumps(c)
    assert c.__dict__.get('_bool_dirty', False) is False
    restored = pickle.loads(blob)
    assert torch.equal(
        restored.bool().as_subclass(torch.Tensor),
        c.bool().as_subclass(torch.Tensor),
    )


# Correctness through dirty path

def test_dirty_chain_correctness(device):
    a = brute.randint(0, 2, (1024,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (1024,), dtype=brute.bit1, device=device)
    c = brute.randint(0, 2, (1024,), dtype=brute.bit1, device=device)
    chain = (a & b) ^ c | ~a
    ref = (a.bool().as_subclass(torch.Tensor)
           & b.bool().as_subclass(torch.Tensor)) ^ c.bool().as_subclass(torch.Tensor) \
          | ~a.bool().as_subclass(torch.Tensor)
    assert_bit1_matches_bool(chain, ref)


def test_dirty_sum_correctness(device):
    a = brute.randint(0, 2, (4096,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (4096,), dtype=brute.bit1, device=device)
    c = a & b
    expected = int((a.bool().as_subclass(torch.Tensor)
                    & b.bool().as_subclass(torch.Tensor)).long().sum())
    assert int(c.sum()) == expected


def test_dirty_then_index_correctness(device):
    a = brute.randint(0, 2, (8, 16), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (8, 16), dtype=brute.bit1, device=device)
    c = a & b
    row = c[0]
    ref = (a.bool().as_subclass(torch.Tensor)[0]
           & b.bool().as_subclass(torch.Tensor)[0])
    assert_bit1_matches_bool(row, ref)


def test_dirty_then_inplace_fill_correctness(device):
    a = brute.randint(0, 2, (64,), dtype=brute.bit1, device=device)
    b = brute.randint(0, 2, (64,), dtype=brute.bit1, device=device)
    c = a & b
    c.fill_(False)
    ref = torch.zeros(64, dtype=torch.bool, device=device)
    assert_bit1_matches_bool(c, ref)
    assert int(c.sum()) == 0


# randomize_ leaves a valid lazy state

def test_randomize_yields_lazy_dirty(device):
    a = brute.zeros(1024, dtype=brute.bit1, device=device)
    a.randomize_()
    assert a.__dict__.get('_bool_dirty', False) is True
    pc = int(a.popcount())
    assert 0 <= pc <= 1024
    materialised = a.bool().as_subclass(torch.Tensor)
    assert int(materialised.long().sum()) == pc
