"""Vector-Symbolic Algebra (VSA) primitives for HÆMMR, on packed ``brute.bit1``.

Everything here operates on the *bipolar hypercube* H_D = {+1, -1}^D, encoded
as packed 1-bit tensors (``brute.bit1``).  The encoding convention used
throughout the whole model is::

        +1  <->  bit 1  (True)
        -1  <->  bit 0  (False)

Under this map (see the HÆMMR whitepaper §2 and BOLD Def. 3.1):

    * XNOR is bipolar multiplication:        e(xnor(a,b)) = e(a)·e(b)
    * signed dot product is popcount-derived:  <u,v> = D - 2·Hamming(u,v)

The three classical VSA operators are therefore *entirely bitwise*:

    * binding   a ⊗ b   = XNOR(a, b)                 (associative, own inverse)
    * bundling  ⊕{x_i}  = coordinate-wise majority   (superposition)
    * permute   ρ^k(x)  = fixed cyclic shift          (position / order)

We deliberately keep the hot path on the *packed* buffers and never unpack a
bit1 tensor to a byte bool tensor:  ``bind`` is a single packed XNOR,
``bundle3`` is three packed AND/OR ops, ``permute`` rolls whole rows.  The only
places integers appear are where the architecture itself is integer-valued
(the signed matmul pre-activations and the BSR accumulator), exactly as the
paper prescribes ("the only non-bitwise element is the transient integer
accumulator").
"""

from __future__ import annotations

from typing import Sequence

import torch

import brute


# ── bit1 <-> ±1 conversions ───────────────────────────────────────────────────

def to_bit1(x: torch.Tensor) -> brute.Tensor:
    """Encode a ±1 / bool / >0 tensor as a packed ``brute.bit1`` tensor.

    ``+1 -> True``, ``-1 -> False``.  Accepts bool directly; any other dtype is
    thresholded at ``> 0``.
    """
    if isinstance(x, brute.Tensor) and getattr(x, "_is_bit1", False):
        return x
    if x.dtype == torch.bool:
        return brute.as_tensor(x, dtype=brute.bit1)
    return brute.as_tensor(x > 0, dtype=brute.bit1)


def to_pm1(x: brute.Tensor) -> torch.Tensor:
    """Decode a ``brute.bit1`` tensor to a float32 ±1 tensor (+1 / -1)."""
    return x.unpack_pm1()


def sign_to_bit1(z: torch.Tensor) -> brute.Tensor:
    """Threshold an integer/real tensor to bit1 with the ``sign(0) = +1`` rule.

    This is the forward Boolean activation ``y = T iff z >= 0`` (BOLD §3.1,
    Def. 3.3 adapted so that the binary state is never 0).
    """
    return brute.as_tensor(z >= 0, dtype=brute.bit1)


# ── Binding (⊗) — XNOR ─────────────────────────────────────────────────────────

def bind(a: brute.Tensor, b: brute.Tensor) -> brute.Tensor:
    """``a ⊗ b`` — elementwise XNOR of two same-shape bit1 tensors (packed)."""
    return brute.fast.eq(a, b)


def bind_pm1(a_pm1: torch.Tensor, b_pm1: torch.Tensor) -> torch.Tensor:
    """Binding in the ±1 domain — plain elementwise product."""
    return a_pm1 * b_pm1


# ── Bundling (⊕) — coordinate-wise majority ────────────────────────────────────

def bundle3(a: brute.Tensor, b: brute.Tensor, c: brute.Tensor) -> brute.Tensor:
    """3-way majority vote ``maj(a,b,c)`` — fully packed (AND/OR only).

    ``maj = (a&b) | (b&c) | (a&c)``.  With an odd number of inputs there are no
    ties, so this is exact and needs no integer accumulator.  Used for the
    binary residual merge (skip ⊕ transform ⊕ learned-tiebreaker).
    """
    ab = brute.fast.bitwise_and(a, b)
    bc = brute.fast.bitwise_and(b, c)
    ac = brute.fast.bitwise_and(a, c)
    return brute.fast.bitwise_or(brute.fast.bitwise_or(ab, bc), ac)


def bundle_sum_pm1(stacked_pm1: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Majority bundle of many ±1 vectors via their integer coordinate sum.

    ``stacked_pm1`` holds the constituents along ``dim``; returns the signed
    integer tally (the BSR/Hopfield "vote accumulator").  Caller applies
    :func:`sign_to_bit1` to obtain the bundled bit-vector.
    """
    return stacked_pm1.sum(dim=dim)


# ── Permutation (ρ) — fixed cyclic shift ───────────────────────────────────────

def position_codes(base_pm1: torch.Tensor, n: int) -> brute.Tensor:
    """Pre-compute ``[ρ^0(POS), ρ^1(POS), ..., ρ^{n-1}(POS)]`` as a bit1 tensor.

    ``ρ`` is a cyclic shift by one coordinate, so ``ρ^i(POS)`` is ``POS`` rolled
    by ``i``.  Returns shape ``(n, D)`` bit1.  Computed once per forward (cheap):
    a single ``unfold``-style roll on the ±1 base then one pack.
    """
    D = base_pm1.shape[-1]
    dev = base_pm1.device
    idx = (torch.arange(D, device=dev).unsqueeze(0)
           - torch.arange(n, device=dev).unsqueeze(1)) % D                   # (n, D)
    rolled = base_pm1[idx]                                                   # (n, D) ±1
    return to_bit1(rolled)


def random_hypervectors(num: int, D: int, *, generator: torch.Generator | None = None,
                        device=None) -> brute.Tensor:
    """``num`` independent uniform-random ±1 hypervectors as a bit1 tensor."""
    bits = torch.randint(0, 2, (num, D), generator=generator).bool()  # CPU generator
    hv = brute.as_tensor(bits, dtype=brute.bit1)
    return hv.to(device) if device is not None else hv


def hamming_similarity(q_bit: brute.Tensor, keys_bit: brute.Tensor) -> torch.Tensor:
    """Signed similarity ``<q, key> = D - 2·Hamming`` for every key (XNOR+popcount).

    ``q_bit`` is ``(M, D)`` (a batch of queries), ``keys_bit`` is ``(N, D)``.
    Returns an int32 ``(M, N)`` similarity matrix via the fused bit1 matmul.
    """
    return brute.fast.matmul(q_bit, keys_bit)
