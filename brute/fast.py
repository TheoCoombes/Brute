"""Direct-call entry points for hot-loop bit1 ops — skip ``__torch_function__``.

Even with all dispatch optimisations, ``__torch_function__`` adds ~5–10 µs
per call (Python entry, pytree walk, fast-path check). For tight inference
loops where you want the absolute minimum overhead, call these helpers
instead — they bypass the override entirely and call ``torch.ops.brute.*``
straight on the packed buffers.

Usage::

    from brute.fast import bitwise_xor, matmul, popcount

    out = bitwise_xor(a, b)        # ~1 µs Python overhead
    c   = matmul(x, w_packed_T)    # bit1 × bit1 → int32
    n   = popcount(a)              # int64 scalar — total set bits

All helpers assume **valid** bit1 inputs with matching shape (where shape
compatibility matters). They do NOT validate; they rely on the underlying
C++ kernels for correctness checks. Use the regular :class:`brute.Tensor`
API when you need broadcasting, type promotion, or mixed-dtype semantics.
"""

from __future__ import annotations

from typing import Any
import torch

from brute.dtype import _PACK_WIDTH
from brute.tensor import Tensor


def _is_bit1(x) -> bool:
    return isinstance(x, Tensor) and getattr(x, '_is_bit1', False)


# ── Bitwise binary ──────────────────────────────────────────────────────

def bitwise_xor(a: Tensor, b: Tensor) -> Tensor:
    """``a ^ b`` on two same-shape bit1 tensors. Returns lazy bit1."""
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.bitwise_xor(a._packed_buf, b._packed_buf),
        list(a.shape),
    )


def bitwise_and(a: Tensor, b: Tensor) -> Tensor:
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.bitwise_and(a._packed_buf, b._packed_buf),
        list(a.shape),
    )


def bitwise_or(a: Tensor, b: Tensor) -> Tensor:
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.bitwise_or(a._packed_buf, b._packed_buf),
        list(a.shape),
    )


def bitwise_not(a: Tensor) -> Tensor:
    """Pad-safe ``~a`` — the trailing word's pad bits stay 0."""
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.bit1_not_packed(
            a._packed_buf, int(a.shape[-1]), _PACK_WIDTH,
        ),
        list(a.shape),
    )


# ── Comparison ──────────────────────────────────────────────────────────

def eq(a: Tensor, b: Tensor) -> Tensor:
    """``a == b`` — pad-safe XNOR."""
    xored = torch.ops.brute.bitwise_xor(a._packed_buf, b._packed_buf)
    inverted = torch.ops.brute.bit1_not_packed(
        xored, int(a.shape[-1]), _PACK_WIDTH,
    )
    return Tensor._make_bit1_from_packed(inverted, list(a.shape))


def ne(a: Tensor, b: Tensor) -> Tensor:
    """``a != b`` — pure XOR."""
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.bitwise_xor(a._packed_buf, b._packed_buf),
        list(a.shape),
    )


# ── Reductions ──────────────────────────────────────────────────────────

def popcount(a: Tensor) -> Tensor:
    """Total set-bit count of a bit1 tensor as an int64 scalar."""
    return Tensor._make_plain(torch.ops.brute.packed_popcount(a._packed_buf))


def hamming(a: Tensor, b: Tensor) -> Tensor:
    """Total Hamming distance between two bit1 tensors as an int64 scalar."""
    return Tensor._make_plain(
        torch.ops.brute.bit1_hamming_total(a._packed_buf, b._packed_buf)
    )


# ── Matmul ──────────────────────────────────────────────────────────────

def matmul(a: Tensor, b_t: Tensor) -> torch.Tensor:
    """A (M×K) @ B^T-as-(N×K). Returns int32 ``K - 2H`` matrix.

    Caller is responsible for storing B in (N, K) form (which is the natural
    layout when B's rows are the output features). For Linear layers this is
    natural; for arbitrary matmul, transpose B first.
    """
    K = int(a.shape[-1])
    return torch.ops.brute.xnor_popcount_matmul(
        a._packed_buf, b_t._packed_buf, K, _PACK_WIDTH,
    )


# ── Packed-buffer arithmetic for the truly performance-paranoid ─────────

def xor_packed(a_buf: torch.Tensor, b_buf: torch.Tensor) -> torch.Tensor:
    """Raw packed XOR — operates on `_packed_buf` directly. No bit1 wrapping."""
    return torch.ops.brute.bitwise_xor(a_buf, b_buf)


def and_packed(a_buf: torch.Tensor, b_buf: torch.Tensor) -> torch.Tensor:
    return torch.ops.brute.bitwise_and(a_buf, b_buf)


def or_packed(a_buf: torch.Tensor, b_buf: torch.Tensor) -> torch.Tensor:
    return torch.ops.brute.bitwise_or(a_buf, b_buf)


def matmul_packed(a_buf: torch.Tensor, b_buf: torch.Tensor,
                   K: int, pack_width: int) -> torch.Tensor:
    """Raw packed matmul — operates on `_packed_buf` directly.

    Use this in CUDA Graph captures where you've extracted the packed buffers
    once and want to launch the kernel with absolute minimum Python overhead.
    """
    return torch.ops.brute.xnor_popcount_matmul(a_buf, b_buf, K, pack_width)


__all__ = [
    'bitwise_xor', 'bitwise_and', 'bitwise_or', 'bitwise_not',
    'eq', 'ne',
    'popcount', 'hamming',
    'matmul',
    'xor_packed', 'and_packed', 'or_packed', 'matmul_packed',
]
