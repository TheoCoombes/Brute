"""Matmul tests.

bit1 × bit1 matmul uses XNOR-popcount on the ±1 encoding (BNN convention).
That is: True → +1, False → −1, then `A @ B = 2 * popcount(XNOR(A,B)) − K`.

Non-bit1 matmul should behave like torch.matmul.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.helpers.reference_ops import (
    ref_matmul_pm1,
    ref_xnor_popcount_matmul,
)


XNOR_MATMUL_SHAPES = [
    ((1, 1), (1, 1)),
    ((2, 8), (8, 4)),
    ((8, 64), (64, 8)),
    ((31, 63), (63, 7)),
    ((16, 8), (8, 16)),
]


@pytest.mark.parametrize("pack_width", [8, 32, 64])
@pytest.mark.parametrize("M,K,N", [(4, 8, 4), (1, 16, 1), (3, 5, 7), (8, 64, 8)])
def test_xnor_popcount_matmul_op(M, K, N, pack_width, device):
    """Validate the low-level torch.ops.brute.xnor_popcount_matmul against reference."""
    if pack_width == 8:
        pack_dtype = torch.uint8
    elif pack_width == 32:
        pack_dtype = torch.uint32
    else:
        pack_dtype = torch.uint64

    a_bool = torch.randint(0, 2, (M, K), dtype=torch.bool, device=device)
    b_bool_T = torch.randint(0, 2, (N, K), dtype=torch.bool, device=device)

    a_bit = bit1(a_bool, pack_dtype=pack_dtype)
    b_bit = bit1(b_bool_T, pack_dtype=pack_dtype)

    out = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, b_bit._packed_buf, K, pack_width
    ).cpu().to(torch.float32)

    # Reference: A as ±1, B.T as ±1 → A @ B.T (run on CPU).
    ref = ref_matmul_pm1(a_bool.cpu(), b_bool_T.cpu().t().contiguous())
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)


@pytest.mark.parametrize("M,K,N", [(4, 8, 4), (3, 5, 7)])
def test_xnor_popcount_pure_reference_match(M, K, N, device):
    """The Python reference implementation must match the C++ kernel."""
    a_bool = torch.randint(0, 2, (M, K), dtype=torch.bool, device=device)
    b_bool_T = torch.randint(0, 2, (N, K), dtype=torch.bool, device=device)
    a_bit = bit1(a_bool, pack_dtype=torch.uint8)
    b_bit = bit1(b_bool_T, pack_dtype=torch.uint8)

    fast = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, b_bit._packed_buf, K, 8
    ).cpu()
    slow = ref_xnor_popcount_matmul(
        a_bit._packed_buf.cpu(), b_bit._packed_buf.cpu(), K, 8
    )
    assert torch.equal(fast, slow)


def test_xnor_matmul_padding_correction(device):
    """K not a multiple of pack_width must still produce correct values."""
    M, K, N = 3, 5, 2  # K=5 needs padding to 8
    a_bool = torch.tensor([[True, False, True, False, True],
                           [False, True, False, True, False],
                           [True, True, False, False, True]], device=device)
    b_bool_T = torch.tensor([[True, False, True, False, True],
                             [False, True, True, False, False]], device=device)

    a_bit = bit1(a_bool)
    b_bit = bit1(b_bool_T)
    out = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, b_bit._packed_buf, K, 8
    ).cpu().to(torch.float32)
    ref = ref_matmul_pm1(a_bool.cpu(), b_bool_T.t().cpu())
    assert torch.allclose(out, ref)


def test_matmul_on_non_bit1_dtype(device):
    """Plain (non-bit1) matmul must mirror torch.matmul exactly."""
    a = brute.tensor(torch.randn(4, 8, device=device))
    b = brute.tensor(torch.randn(8, 4, device=device))
    out = a @ b
    ref = a.as_subclass(torch.Tensor) @ b.as_subclass(torch.Tensor)
    assert torch.allclose(out.as_subclass(torch.Tensor), ref)


def test_matmul_int_tensors(device):
    a = brute.tensor([[1, 2, 3]], dtype=torch.int32, device=device)
    b = brute.tensor([[4], [5], [6]], dtype=torch.int32, device=device)
    out = a @ b
    ref = torch.tensor([[32]], dtype=torch.int32, device=device)
    assert torch.equal(out.as_subclass(torch.Tensor), ref)


def test_matmul_vec_vec(device):
    a = brute.tensor([1.0, 2.0, 3.0], device=device)
    b = brute.tensor([4.0, 5.0, 6.0], device=device)
    out = torch.dot(a, b)
    assert float(out.item()) == 32.0


def test_xnor_matmul_zero_result_on_orthogonal_pattern():
    """+1 / -1 alternating row dot with random row → result in [-K, K]."""
    K = 16
    a_bool = torch.zeros((2, K), dtype=torch.bool)
    a_bool[0, ::2] = True
    a_bool[1, 1::2] = True

    a_bit = bit1(a_bool)
    out = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, a_bit._packed_buf, K, 8
    )
    # Diagonal must equal K (each row matches itself).
    diag = torch.diag(out)
    assert torch.equal(diag, torch.full((2,), K, dtype=torch.int32))


@pytest.mark.parametrize("pack_width,pack_dtype", [
    (8, torch.uint8), (32, torch.uint32), (64, torch.uint64),
])
def test_xnor_matmul_pack_width_consistency(pack_width, pack_dtype, device):
    """The result must be invariant to the pack width chosen."""
    M, K, N = 4, 32, 4
    torch.manual_seed(0)
    a_bool = torch.randint(0, 2, (M, K), dtype=torch.bool, device=device)
    b_bool_T = torch.randint(0, 2, (N, K), dtype=torch.bool, device=device)

    a_bit = bit1(a_bool, pack_dtype=pack_dtype)
    b_bit = bit1(b_bool_T, pack_dtype=pack_dtype)
    out = torch.ops.brute.xnor_popcount_matmul(
        a_bit._packed_buf, b_bit._packed_buf, K, pack_width
    ).cpu().to(torch.float32)

    ref = ref_matmul_pm1(a_bool.cpu(), b_bool_T.cpu().t().contiguous())
    assert torch.allclose(out, ref), f"mismatch for pack_width={pack_width}"
