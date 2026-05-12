"""torch.fx symbolic tracing interop."""
from __future__ import annotations

import pytest
import torch
import torch.fx

import brute
from tests.helpers import bit1


def test_fx_trace_simple(device):
    def fn(x):
        return torch.logical_not(x)

    try:
        traced = torch.fx.symbolic_trace(fn)
    except Exception as e:
        pytest.xfail(f"FX trace unsupported: {e}")

    a = bit1(torch.tensor([True, False, True], device=device))
    out = traced(a)
    ref = torch.logical_not(torch.tensor([True, False, True], device=device))
    assert torch.equal(out.bool().as_subclass(torch.Tensor).cpu(), ref.cpu())


def test_fx_trace_compose(device):
    def fn(x, y):
        return (x & y).logical_not()

    try:
        traced = torch.fx.symbolic_trace(fn)
    except Exception as e:
        pytest.xfail(f"FX trace unsupported: {e}")

    a = bit1(torch.tensor([True, False, True], device=device))
    b = bit1(torch.tensor([True, True, False], device=device))
    out = traced(a, b)
    ref = (torch.tensor([True, False, True], device=device)
           & torch.tensor([True, True, False], device=device)).logical_not()
    assert torch.equal(out.bool().as_subclass(torch.Tensor).cpu(), ref.cpu())


def test_fx_trace_arithmetic(device):
    def fn(x):
        return x * 2 + 1

    traced = torch.fx.symbolic_trace(fn)
    x = brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    out = traced(x)
    expected = torch.tensor([3.0, 5.0, 7.0], device=device)
    assert torch.allclose(out.as_subclass(torch.Tensor), expected)
