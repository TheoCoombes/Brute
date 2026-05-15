"""Fuzz XNOR-popcount matmul."""
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

import brute
from tests.helpers import bit1
from tests.helpers.reference_ops import ref_matmul_pm1

pytestmark = pytest.mark.fuzz


@settings(max_examples=120, deadline=None)
@given(
    M=st.integers(min_value=1, max_value=32),
    K=st.integers(min_value=1, max_value=64),
    N=st.integers(min_value=1, max_value=32),
)
def test_fuzz_xnor_matmul(M, K, N):
    a_bool = torch.randint(0, 2, (M, K), dtype=torch.bool)
    b_bool_T = torch.randint(0, 2, (N, K), dtype=torch.bool)

    a_bit = bit1(a_bool)
    b_bit = bit1(b_bool_T)
    out = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, b_bit._packed_buf, K
    ).to(torch.float32)

    ref = ref_matmul_pm1(a_bool, b_bool_T.t())
    assert torch.allclose(out, ref), (
        f"mismatch for M={M}, K={K}, N={N}"
    )
