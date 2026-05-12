"""__torch_function__ dispatch behavior."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


def test_subclass_preserved_through_torch_op(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    b = bit1(torch.tensor([True, True, False], device=device))
    out = torch.logical_and(a, b)
    assert isinstance(out, brute.Tensor)


def test_subclass_preserved_through_clone(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    c = a.clone()
    assert isinstance(c, brute.Tensor)
    assert c.dtype == brute.bit1


def test_bit1_op_bit1_returns_bit1(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    b = bit1(torch.tensor([False, True, False], device=device))
    out = a & b
    assert out.dtype == brute.bit1


def test_bit1_op_bool_does_not_coerce_to_bit1(device):
    """Mixed bit1+bool inputs must NOT silently coerce the bool side to bit1."""
    a = bit1(torch.tensor([True, False, True], device=device))
    b = torch.tensor([True, True, False], device=device)
    out = a & b
    assert out.dtype == torch.bool, (
        f"Expected torch.bool to avoid silent bit1 coercion; got {out.dtype}"
    )


def test_torch_function_returns_brute_subclass_for_non_bool(device):
    """Plain non-bool ops should still return brute.Tensor (subclass preserved)."""
    a = brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    out = a + 1
    assert isinstance(out, brute.Tensor)


def test_torch_function_handles_reductions(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    s = torch.sum(a)
    assert isinstance(s, brute.Tensor)


def test_torch_function_handles_kwargs(device):
    a = bit1(torch.tensor([[True, False], [False, True]], device=device))
    s = torch.sum(a, dim=0)
    assert isinstance(s, brute.Tensor)


def test_torch_function_handles_tuple_returns(device):
    a = bit1(torch.tensor([True, False, True, False], device=device))
    out = torch.unique(a, return_counts=True)
    # Should be a tuple of two tensors.
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_dispatch_does_not_recurse(device):
    """Ensure repeated dispatch does not blow the recursion stack."""
    a = bit1(torch.tensor([True, False], device=device))
    for _ in range(50):
        a = a.clone()
    assert isinstance(a, brute.Tensor)


def test_clone_preserves_bit1_pack_dtype(device):
    a = bit1(torch.tensor([True, False, True], device=device), pack_dtype=torch.uint32)
    c = a.clone()
    if c.dtype == brute.bit1:
        # If clone preserves dtype, it should preserve pack_dtype too.
        assert c.pack_dtype == torch.uint32


@pytest.mark.parametrize("op", [
    torch.logical_and, torch.logical_or, torch.logical_xor,
])
def test_logical_ops_propagate_bit1(op, device):
    a = bit1(torch.tensor([True, False, True], device=device))
    b = bit1(torch.tensor([False, True, True], device=device))
    out = op(a, b)
    assert out.dtype == brute.bit1


def test_logical_not_propagates_bit1(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    out = torch.logical_not(a)
    assert out.dtype == brute.bit1


def test_dispatch_for_unsupported_op_falls_back_or_raises(device):
    """An op that returns a non-bool tensor on a bool input should not be promoted to bit1."""
    a = bit1(torch.tensor([True, False, True], device=device))
    out = a.int()
    assert out.dtype == torch.int32 or out.dtype == torch.int64


def test_dispatch_dtype_isolation_invariant(device):
    """All-bit1 inputs → bit1 output. Mixed bit1+bool → bool output."""
    a = bit1(torch.tensor([True, False, True], device=device))
    b_bit = bit1(torch.tensor([False, True, True], device=device))
    b_bool = torch.tensor([False, True, True], device=device)

    out_pure = a & b_bit
    out_mixed = a & b_bool
    assert out_pure.dtype == brute.bit1
    assert out_mixed.dtype == torch.bool
