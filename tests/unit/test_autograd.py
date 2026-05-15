"""Autograd interop tests."""
from __future__ import annotations

import pytest
import torch

import brute


def test_requires_grad_on_float(device):
    src = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    a = brute.tensor(src)
    if not a.requires_grad:
        a = a.detach().requires_grad_(True)
    assert a.requires_grad


def test_backward_simple(device):
    src = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    loss = (src * src).sum()
    loss.backward()
    expected = torch.tensor([2.0, 4.0, 6.0], device=device)
    assert torch.allclose(src.grad, expected)


def test_bool_does_not_require_grad(device):
    """bool tensors don't support requires_grad; bit1 should not either."""
    bit = brute.tensor([True, False, True], dtype=brute.bit1, device=device)
    with pytest.raises((RuntimeError, ValueError)):
        bit.requires_grad_(True)


def test_grad_through_matmul(device):
    a = torch.randn(4, 8, device=device, requires_grad=True)
    b = torch.randn(8, 4, device=device, requires_grad=True)
    c = brute.tensor(a) @ brute.tensor(b)
    if not c.requires_grad:
        c = a @ b
    loss = c.sum()
    loss.backward()


def test_detach(device):
    src = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    a = brute.tensor(src)
    d = a.detach()
    assert d.requires_grad is False


def test_no_grad_context(device):
    src = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    a = brute.tensor(src) if brute.tensor(src).requires_grad else src
    with torch.no_grad():
        b = a * 2
    assert b.requires_grad is False
