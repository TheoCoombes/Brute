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
    pack_width=st.sampled_from([8, 32, 64]),
)
def test_fuzz_xnor_matmul(M, K, N, pack_width):
    pack_dtype = {8: torch.uint8, 32: torch.uint32, 64: torch.uint64}[pack_width]
    a_bool = torch.randint(0, 2, (M, K), dtype=torch.bool)
    b_bool_T = torch.randint(0, 2, (N, K), dtype=torch.bool)

    a_bit = bit1(a_bool, pack_dtype=pack_dtype)
    b_bit = bit1(b_bool_T, pack_dtype=pack_dtype)
    out = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, b_bit._packed_buf, K, pack_width
    ).to(torch.float32)

    ref = ref_matmul_pm1(a_bool, b_bool_T.t())
    assert torch.allclose(out, ref), (
        f"mismatch for M={M}, K={K}, N={N}, pw={pack_width}"
    )
