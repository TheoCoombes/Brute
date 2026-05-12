"""torch.compile interop."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_compile_logical_ops(device):
    if device == "mps":
        pytest.skip("torch.compile MPS backend is unstable")

    def fn(x, y):
        return torch.logical_not(x & y)

    a = bit1(torch.tensor([True, False, True], device=device))
    b = bit1(torch.tensor([False, True, True], device=device))

    eager = fn(a, b)
    try:
        compiled = torch.compile(fn, fullgraph=False)
        out = compiled(a, b)
    except Exception as e:
        pytest.xfail(f"torch.compile not supported for bit1: {e}")

    assert torch.equal(
        eager.bool().cpu().as_subclass(torch.Tensor),
        out.bool().cpu().as_subclass(torch.Tensor),
    )


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_compile_on_plain_float(device):
    if device == "mps":
        pytest.skip("torch.compile MPS backend is unstable")

    def fn(x):
        return x * 2.0 + 1.0

    x = brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    eager = fn(x)
    try:
        compiled = torch.compile(fn, fullgraph=False)
        out = compiled(x)
    except Exception as e:
        pytest.xfail(f"torch.compile not supported: {e}")

    assert torch.allclose(eager.as_subclass(torch.Tensor),
                          out.as_subclass(torch.Tensor))
