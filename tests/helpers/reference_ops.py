"""Pure-Python reference implementations of brute extensions.

These are intentionally slow and obviously correct — they exist to compare
against the optimized C++/Metal/CUDA kernels.
"""
from __future__ import annotations

import torch


def ref_popcount(packed: torch.Tensor) -> torch.Tensor:
    """Reference per-element popcount on a packed uint tensor."""
    if packed.dtype == torch.uint8:
        bits = 8
        mask = 0xFF
    elif packed.dtype in (torch.uint32, torch.int32):
        bits = 32
        mask = 0xFFFFFFFF
    elif packed.dtype in (torch.uint64, torch.int64):
        bits = 64
        mask = 0xFFFFFFFFFFFFFFFF
    else:
        raise TypeError(f"unsupported packed dtype: {packed.dtype}")

    flat = packed.contiguous().view(-1)
    out = torch.empty_like(flat, dtype=torch.int32)
    for i, v in enumerate(flat.tolist()):
        out[i] = bin(int(v) & mask).count("1")
    return out.view(packed.shape)


def ref_packed_popcount(packed: torch.Tensor) -> torch.Tensor:
    """Reference: total popcount across an entire packed buffer → int64 scalar."""
    return ref_popcount(packed).long().sum()


def ref_xnor_popcount_matmul(
    A_packed: torch.Tensor,
    B_packed: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Reference: K_logical − 2·popcount(A_row XOR B_row).

    Both A and B are (rows, Kp) int64-packed (pack_width = 64).
    """
    assert A_packed.dim() == 2 and B_packed.dim() == 2
    Kp = A_packed.size(1)
    assert Kp == B_packed.size(1)
    M = A_packed.size(0)
    N = B_packed.size(0)
    mask = 0xFFFFFFFFFFFFFFFF
    A = A_packed.cpu().tolist()
    B = B_packed.cpu().tolist()
    out = torch.zeros((M, N), dtype=torch.int32)
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(Kp):
                xor = (int(A[m][k]) ^ int(B[n][k])) & mask
                acc += bin(xor).count("1")
            out[m, n] = K - 2 * acc
    return out


def ref_matmul_pm1(a_bool: torch.Tensor, b_bool: torch.Tensor) -> torch.Tensor:
    """Reference float matmul of bool→{+1,-1} encoding."""
    a_pm1 = a_bool.float() * 2 - 1
    b_pm1 = b_bool.float() * 2 - 1
    return a_pm1 @ b_pm1


def ref_matmul_bool(a_bool: torch.Tensor, b_bool: torch.Tensor) -> torch.Tensor:
    """Reference int matmul of two boolean tensors (count of AND matches)."""
    return a_bool.int() @ b_bool.int()
