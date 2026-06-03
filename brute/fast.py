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

from brute.tensor import Tensor


def _is_bit1(x) -> bool:
    return isinstance(x, Tensor) and getattr(x, '_is_bit1', False)


# ── Bitwise binary ──────────────────────────────────────────────────────

def bitwise_xor(a: Tensor, b: Tensor) -> Tensor:
    """``a ^ b`` on two same-shape bit1 tensors. Returns lazy bit1."""
    return Tensor._make_bit1_from_packed(
        torch.bitwise_xor(a._packed_buf, b._packed_buf),
        list(a.shape),
    )


def bitwise_and(a: Tensor, b: Tensor) -> Tensor:
    return Tensor._make_bit1_from_packed(
        torch.bitwise_and(a._packed_buf, b._packed_buf),
        list(a.shape),
    )


def bitwise_or(a: Tensor, b: Tensor) -> Tensor:
    return Tensor._make_bit1_from_packed(
        torch.bitwise_or(a._packed_buf, b._packed_buf),
        list(a.shape),
    )


def bitwise_not(a: Tensor) -> Tensor:
    """Pad-safe ``~a`` — the trailing word's pad bits stay 0."""
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.bit1_not_packed(
            a._packed_buf, int(a.shape[-1])),
        list(a.shape),
    )


# ── Comparison ──────────────────────────────────────────────────────────

def eq(a: Tensor, b: Tensor) -> Tensor:
    """``a == b`` — pad-safe XNOR."""
    xored = torch.bitwise_xor(a._packed_buf, b._packed_buf)
    inverted = torch.ops.brute.bit1_not_packed(
        xored, int(a.shape[-1]))
    return Tensor._make_bit1_from_packed(inverted, list(a.shape))


def ne(a: Tensor, b: Tensor) -> Tensor:
    """``a != b`` — pure XOR."""
    return Tensor._make_bit1_from_packed(
        torch.bitwise_xor(a._packed_buf, b._packed_buf),
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
        a._packed_buf, b_t._packed_buf, K)


def sign(a: torch.Tensor) -> Tensor:
    """Pack ``a >= 0`` directly into bit1 storage without a bool temporary."""
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.pack_sign(a),
        list(a.shape),
    )


def matmul_sign(a: Tensor, b_t: Tensor, K: int) -> Tensor:
    """A (M×K) @ B^T-as-(N×K) → packed bit1 (M, ceil(N/64)).

    Fuses the XNOR-popcount matmul and the sign threshold into one pass,
    avoiding the int32 intermediate.  C[m,n] = 1 iff K − 2H(A[m],B[n]) ≥ 0.

    Use instead of ``matmul`` + ``sign`` in BooleanLinear when ``boundary_nu``
    is None (the common case) — the int32 ``z`` is never materialised.
    """
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.xnor_popcount_matmul_sign(
            a._packed_buf, b_t._packed_buf, K),
        list(a.shape[:-1]) + [int(b_t.shape[0])],
    )


def majority(rows: Tensor, k: int, D: int) -> Tensor:
    """Bit-sliced majority vote over ``k`` packed rows.

    ``rows`` is shape ``(batch, k, D)`` bit1.  Returns ``(batch, D)`` bit1
    where bit d is 1 iff more than k//2 of the k input rows have bit d set.
    For odd k there are no ties.
    """
    return Tensor._make_bit1_from_packed(
        torch.ops.brute.packed_majority(rows._packed_buf, k, D),
        [int(rows.shape[0]), D],
    )


def episodic_causal_search(
    qc: Tensor,
    kc_buf: torch.Tensor,
    qp: Tensor,
    pos_buf: torch.Tensor,
    payload: torch.Tensor,
    cnt: torch.Tensor,
    D: int,
) -> tuple[Tensor, torch.Tensor, torch.Tensor]:
    """Fused windowed Hamming search + top-1 + payload gather (inference path).

    Args:
        qc:      (B, D) bit1 — content query
        kc_buf:  (B, N, Kp) int64 packed — content key ring buffer
        qp:      (B, D) bit1 — position query
        pos_buf: (B, N, Kp) int64 packed — position ring buffer
        payload: (B, N, Kp) int64 packed — payload ring buffer
        cnt:     (B,) int32 — valid slot count per batch element
        D:       logical dimension

    Returns:
        read:  (B, D) bit1 — gathered payload at argmax slot
        idx:   (B,) int32 — argmax slot index (-1 if no valid slot)
        score: (B,) int32 — best total Hamming score
    """
    Kp = int(kc_buf.shape[-1])
    B  = int(qc.shape[0])
    read_packed, idx, score = torch.ops.brute.episodic_causal_search(
        qc._packed_buf, kc_buf, qp._packed_buf, pos_buf, payload, cnt, D)
    return (
        Tensor._make_bit1_from_packed(read_packed, [B, D]),
        idx,
        score,
    )


def bsr_scan(q: Tensor, assoc: Tensor, decay_shifts: torch.Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fused packed BSR forward scan.

    Returns ``(read, state, gate)`` as bit1 tensors. ``state`` is the recurrent
    sign before each write; ``gate`` is true where the delta write fires.
    """
    D = int(q.shape[-1])
    read, state, gate = torch.ops.brute.bsr_scan(
        q._packed_buf, assoc._packed_buf, decay_shifts, D)
    shape = list(q.shape)
    return (
        Tensor._make_bit1_from_packed(read, shape),
        Tensor._make_bit1_from_packed(state, shape),
        Tensor._make_bit1_from_packed(gate, shape),
    )


# ── Packed-buffer arithmetic for the truly performance-paranoid ─────────

def xor_packed(a_buf: torch.Tensor, b_buf: torch.Tensor) -> torch.Tensor:
    """Raw packed XOR — operates on `_packed_buf` directly. No bit1 wrapping."""
    return torch.bitwise_xor(a_buf, b_buf)


def and_packed(a_buf: torch.Tensor, b_buf: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_and(a_buf, b_buf)


def or_packed(a_buf: torch.Tensor, b_buf: torch.Tensor) -> torch.Tensor:
    return torch.bitwise_or(a_buf, b_buf)


def matmul_packed(a_buf: torch.Tensor, b_buf: torch.Tensor, K: int) -> torch.Tensor:
    """Raw packed matmul — operates on `_packed_buf` directly.

    Use this in CUDA Graph captures where you've extracted the packed buffers
    once and want to launch the kernel with absolute minimum Python overhead.
    """
    return torch.ops.brute.xnor_popcount_matmul(a_buf, b_buf, K)


__all__ = [
    'bitwise_xor', 'bitwise_and', 'bitwise_or', 'bitwise_not',
    'eq', 'ne',
    'popcount', 'hamming',
    'matmul', 'sign', 'matmul_sign', 'majority', 'episodic_causal_search',
    'bsr_scan',
    'xor_packed', 'and_packed', 'or_packed', 'matmul_packed',
]
