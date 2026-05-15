"""Stress tests on large tensors."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1

pytestmark = pytest.mark.stress


@pytest.mark.parametrize("shape", [(1 << 16,), (4096, 256), (256, 4096)])
def test_large_pack_unpack_roundtrip(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    assert torch.equal(bit.bool().as_subclass(torch.Tensor), src)


@pytest.mark.parametrize("shape", [(1 << 16,), (1024, 64)])
def test_large_popcount_matches_sum(shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    bit = bit1(src)
    assert int(bit.popcount().item()) == int(src.long().sum().item())


@pytest.mark.parametrize("M,K,N", [(64, 1024, 64), (128, 512, 128)])
def test_large_xnor_matmul(M, K, N):
    a_bool = torch.randint(0, 2, (M, K), dtype=torch.bool)
    b_bool_T = torch.randint(0, 2, (N, K), dtype=torch.bool)
    a_bit = bit1(a_bool)
    b_bit = bit1(b_bool_T)
    out = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, b_bit._packed_buf, K
    ).to(torch.float32)
    a_pm1 = a_bool.float() * 2 - 1
    b_pm1 = b_bool_T.t().float() * 2 - 1
    ref = a_pm1 @ b_pm1
    assert torch.allclose(out, ref)


def test_very_large_clone_does_not_share_storage():
    src = torch.randint(0, 2, (4096,), dtype=torch.bool)
    bit = bit1(src)
    c = bit.clone()
    assert (bit.as_subclass(torch.Tensor).untyped_storage().data_ptr()
            != c.as_subclass(torch.Tensor).untyped_storage().data_ptr())
