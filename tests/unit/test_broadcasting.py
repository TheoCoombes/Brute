"""Broadcasting behavior parity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


BROADCAST_SHAPES = [
    ((1, 4, 1), (3, 1, 5)),
    ((4, 1), (1, 6)),
    ((1,), (8,)),
    ((), (8,)),
    ((), ()),
    ((1, 1, 1), (2, 3, 4)),
    ((2, 1, 4), (1, 3, 4)),
]


@pytest.mark.parametrize("a_shape,b_shape", BROADCAST_SHAPES)
def test_broadcast_and(a_shape, b_shape, device):
    a = torch.randint(0, 2, a_shape, dtype=torch.bool, device=device)
    b = torch.randint(0, 2, b_shape, dtype=torch.bool, device=device)
    ref = a & b
    out = bit1(a) & bit1(b)
    assert_bit1_matches_bool(out, ref)


@pytest.mark.parametrize("a_shape,b_shape", BROADCAST_SHAPES)
def test_broadcast_or(a_shape, b_shape, device):
    a = torch.randint(0, 2, a_shape, dtype=torch.bool, device=device)
    b = torch.randint(0, 2, b_shape, dtype=torch.bool, device=device)
    ref = a | b
    out = bit1(a) | bit1(b)
    assert_bit1_matches_bool(out, ref)


@pytest.mark.parametrize("a_shape,b_shape", BROADCAST_SHAPES)
def test_broadcast_xor(a_shape, b_shape, device):
    a = torch.randint(0, 2, a_shape, dtype=torch.bool, device=device)
    b = torch.randint(0, 2, b_shape, dtype=torch.bool, device=device)
    ref = a ^ b
    out = bit1(a) ^ bit1(b)
    assert_bit1_matches_bool(out, ref)


def test_scalar_broadcast(device):
    a = torch.tensor([True, False, True], device=device)
    bit = bit1(a)
    out = bit & True
    ref = a & True
    assert_bit1_matches_bool(out, ref)


def test_failed_broadcast_raises(device):
    a = bit1(torch.zeros((3, 4), dtype=torch.bool, device=device))
    b = bit1(torch.zeros((5, 4), dtype=torch.bool, device=device))
    with pytest.raises((RuntimeError, ValueError)):
        _ = a & b


def test_broadcast_tensors(device):
    a = torch.tensor([[True], [False]], device=device)
    b = torch.tensor([True, False, True], device=device)
    a_bit, b_bit = brute.broadcast_tensors(bit1(a), bit1(b))
    a_ref, b_ref = torch.broadcast_tensors(a, b)
    assert_bit1_matches_bool(a_bit, a_ref)
    assert_bit1_matches_bool(b_bit, b_ref)


def test_broadcast_to(device):
    x = bit1(torch.tensor([[True], [False]], device=device))
    out = torch.broadcast_to(x, (2, 4))
    ref = torch.broadcast_to(torch.tensor([[True], [False]], device=device), (2, 4))
    assert_bit1_matches_bool(out, ref)


def test_broadcast_with_torch_bool(device):
    """bit1 broadcast with a plain torch.bool input — must NOT silently coerce
    the bool side to bit1 (dtype isolation invariant)."""
    a = bit1(torch.tensor([True, False], device=device))
    b = torch.tensor([True, True], device=device)
    out = a & b
    # The result should still match the boolean computation.
    ref = torch.tensor([True, False], device=device) & torch.tensor([True, True], device=device)
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref)
