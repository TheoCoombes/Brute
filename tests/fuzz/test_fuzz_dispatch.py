"""Dispatch fuzz: a curated list of torch ops must not crash on bit1 input."""
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.fuzz


# Curated set of torch namespace ops that should accept a 2-D bool tensor and
# return some kind of tensor result. None of these should crash on bit1 inputs.
BOOL_DISPATCH_OPS = [
    ("clone", lambda t: torch.clone(t)),
    ("contiguous", lambda t: t.contiguous()),
    ("t", lambda t: t.t()),
    ("flatten", lambda t: torch.flatten(t)),
    ("logical_not", lambda t: torch.logical_not(t)),
    ("bitwise_not", lambda t: torch.bitwise_not(t)),
    ("sum", lambda t: torch.sum(t)),
    ("all", lambda t: torch.all(t)),
    ("any", lambda t: torch.any(t)),
    ("count_nonzero", lambda t: torch.count_nonzero(t)),
    # argmax/argmin are unsupported on bool in torch — excluded here.
    ("max", lambda t: torch.max(t)),
    ("min", lambda t: torch.min(t)),
    ("transpose", lambda t: t.transpose(0, 1)),
    ("squeeze", lambda t: t.squeeze()),
    ("unsqueeze", lambda t: t.unsqueeze(0)),
    ("flip", lambda t: torch.flip(t, dims=[0])),
    ("nonzero", lambda t: torch.nonzero(t)),
    ("argwhere", lambda t: torch.argwhere(t)),
]


@pytest.mark.parametrize("name,op", BOOL_DISPATCH_OPS)
@settings(max_examples=20, deadline=None)
@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
)
def test_dispatch_does_not_crash(name, op, shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    try:
        result = op(bit)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"op {name} crashed on bit1 input shape={shape}: {e!r}")
    # Result must remain a tensor (subclass or not).
    assert result is None or isinstance(result, torch.Tensor)


BINARY_BOOL_OPS = [
    ("logical_and", lambda a, b: a & b),
    ("logical_or", lambda a, b: a | b),
    ("logical_xor", lambda a, b: a ^ b),
    ("eq", torch.eq),
    ("ne", torch.ne),
]


@pytest.mark.parametrize("name,op", BINARY_BOOL_OPS)
@settings(max_examples=20, deadline=None)
@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
)
def test_binary_dispatch_does_not_crash(name, op, shape):
    a = torch.randint(0, 2, shape, dtype=torch.bool)
    b = torch.randint(0, 2, shape, dtype=torch.bool)
    out = op(bit1(a), bit1(b))
    ref = op(a, b)
    assert torch.equal(out.bool().as_subclass(torch.Tensor), ref.bool()), (
        f"op {name} mismatch on shape={shape}"
    )
