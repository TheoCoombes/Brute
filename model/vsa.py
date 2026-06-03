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
from brute.tensor import Tensor as _BT


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
    if z.dim() >= 1 and z.dtype in (torch.int32, torch.float32):
        return brute.fast.sign(z)
    mask = z >= 0
    packed = torch.ops.brute.pack_bool(mask)
    return _BT._make_bit1_from_packed(packed, list(mask.shape))


# ── Binding (⊗) — XNOR ─────────────────────────────────────────────────────────

def bind(a: brute.Tensor, b: brute.Tensor) -> brute.Tensor:
    """``a ⊗ b`` — elementwise XNOR of two same-shape bit1 tensors (packed)."""
    return brute.fast.eq(a, b)


def unbind(a: brute.Tensor, b: brute.Tensor) -> brute.Tensor:
    """``a ⊘ b`` — the inverse of binding.  XNOR is its own inverse, so this is
    literally :func:`bind`; the alias exists so call-sites that *recover* a
    position-free value from a bound key read clearly."""
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


def hierarchical_position_codes(chunk_base_pm1: torch.Tensor, offset_base_pm1: torch.Tensor,
                                n: int, *, chunk: int = 256) -> brute.Tensor:
    """Hierarchical positional binding for the episodic address lane.

    Factor ``i = b·C + o`` (chunk ``b``, offset ``o``) over two *orthogonal*
    bases and bind::

        POS_i = ρ^b(P_chunk) ⊗ ρ^o(P_offset)

    A single cyclic shift ``ρ^i`` develops harmonics/wrap-around over long
    contexts (so ``POS_10000`` can spuriously correlate with ``POS_10``); the
    two-level factorisation gives non-overlapping positional codes over ``C·D``
    tokens while living entirely in the address lane (never decoded).  Returns
    ``(n, D)`` bit1.
    """
    D = chunk_base_pm1.shape[-1]
    dev = chunk_base_pm1.device
    i = torch.arange(n, device=dev)
    b = (i // chunk).unsqueeze(1)                                           # (n,1)
    o = (i % chunk).unsqueeze(1)                                            # (n,1)
    didx = torch.arange(D, device=dev).unsqueeze(0)                         # (1,D)
    chunk_rolled = chunk_base_pm1[(didx - b) % D]                           # (n,D) ρ^b
    offset_rolled = offset_base_pm1[(didx - o) % D]                         # (n,D) ρ^o
    return to_bit1(chunk_rolled * offset_rolled)                            # bind in ±1


def random_hypervectors(num: int, D: int, *, generator: torch.Generator | None = None,
                        device=None) -> brute.Tensor:
    """``num`` independent uniform-random ±1 hypervectors as a bit1 tensor."""
    bits = torch.randint(0, 2, (num, D), generator=generator).bool()  # CPU generator
    hv = brute.as_tensor(bits, dtype=brute.bit1)
    return hv.to(device) if device is not None else hv


def lsh_buckets(kc_packed: torch.Tensor, S: int) -> torch.Tensor:
    """LSH bucket index for each token using low bits of the first packed word.

    ``kc_packed`` is ``(B, n, Kp)`` int64.  Returns ``(B, n)`` int64 bucket
    indices in ``[0, S)``.  Content keys are balanced binary codes, so their
    low bits are a free locality hash — no extra projection needed.
    """
    n_bits = max(1, (S - 1).bit_length())
    mask   = (1 << n_bits) - 1
    return (kc_packed[:, :, 0] & mask).long() % S


def packed_majority_vote(rows: brute.Tensor, k: int, D: int) -> brute.Tensor:
    """Majority vote over ``k`` packed binary rows → packed bit1 output.

    Thin wrapper around ``brute.fast.majority`` for use in the model layers.
    ``rows`` is ``(batch, k, D)`` bit1; returns ``(batch, D)`` bit1.
    """
    return brute.fast.majority(rows, k, D)


def hamming_similarity(q_bit: brute.Tensor, keys_bit: brute.Tensor) -> torch.Tensor:
    """Signed similarity ``<q, key> = D - 2·Hamming`` for every key (XNOR+popcount).

    ``q_bit`` is ``(M, D)`` (a batch of queries), ``keys_bit`` is ``(N, D)``.
    Returns an int32 ``(M, N)`` similarity matrix via the fused bit1 matmul.
    """
    return brute.fast.matmul(q_bit, keys_bit)


# ── Binary Equiangular Frame (structured codebook) ─────────────────────────────

def binary_equiangular_frame(C: int, D: int, *, alpha: float = 1.0, n_sweeps: int = 30,
                             generator: torch.Generator | None = None,
                             tol: float = 1e-6, max_cells: int = 80_000_000,
                             max_optimized_codes: int = 512,
                             verbose: bool = False) -> torch.Tensor:
    """Build ``C`` maximally- and uniformly-separated ±1 codes in ``H_D`` (BEP App. C).

    Minimises ``J = Σ_{i<j}⟨ρ_i,ρ_j⟩ + α·Var_{i<j}(⟨ρ_i,ρ_j⟩)`` (BEP Eq. 13) by
    greedy coordinate flips: the first term pushes every pair toward maximal
    Hamming separation (negative inner product), the second makes the spacing
    *uniform* (equiangular).  Replacing HÆMMR's random token codebook with a BEF
    removes the anomalously-close pairs that drive decode collisions.

    Returns a ``(C, D)`` float ±1 tensor.  This inline implementation is meant
    for tests and small local models; production-scale vocabularies should use
    an offline precompute.  It falls back to a random balanced frame when the
    greedy optimiser would be too expensive for interactive startup.
    """
    if C < 2 or C * D > max_cells or C > max_optimized_codes:
        bits = torch.randint(0, 2, (C, D), generator=generator)
        return bits.float() * 2 - 1

    rho = (torch.randint(0, 2, (C, D), generator=generator).float() * 2 - 1)    # (C,D)
    G = rho @ rho.t()                                                           # (C,C)
    P = C * (C - 1) / 2.0
    off = ~torch.eye(C, dtype=torch.bool)
    Ssum = G[off].sum() / 2.0                                                   # Σ_{i<j} G_ij
    Q = (G[off] ** 2).sum() / 2.0                                              # Σ_{i<j} G_ij²

    def cost(Ssum, Q):
        mu = Ssum / P
        var = Q / P - mu * mu
        return Ssum + alpha * var

    order_gen = generator
    for sweep in range(n_sweeps):
        flips = 0
        perm = torch.randperm(C, generator=order_gen)
        for i in perm.tolist():
            s = rho[i]                                                          # (D,) ±1
            mask = off[i]                                                       # (C,) j≠i
            rho_rest = rho[mask]                                                # (C-1,D)
            w = G[i][mask]                                                      # (C-1,)
            colsum = rho_rest.sum(dim=0)                                        # (D,)  Σ_{j≠i} ρ_j[k]
            wrho = w @ rho_rest                                                 # (D,)  Σ_{j≠i} G_ij ρ_j[k]
            dSsum = -2.0 * s * colsum                                           # (D,)
            dQ = -4.0 * s * wrho + 4.0 * (C - 1)                               # (D,)
            new_Ssum = Ssum + dSsum
            new_Q = Q + dQ
            mu = new_Ssum / P
            new_cost = new_Ssum + alpha * (new_Q / P - mu * mu)
            base = cost(Ssum, Q)
            gain = new_cost - base                                             # (D,)
            k = int(torch.argmin(gain).item())
            if gain[k] < -tol:
                # apply flip of coordinate k of code i
                sk = s[k].item()
                delta = (-2.0 * sk) * rho[:, k]                                 # (C,) change to G[i,:]
                delta[i] = 0.0
                G[i] += delta
                G[:, i] += delta
                rho[i, k] = -sk
                Ssum = Ssum + dSsum[k]
                Q = Q + dQ[k]
                flips += 1
        if verbose:
            mu = (Ssum / P).item()
            var = (Q / P - mu * mu)
            print(f"  BEF sweep {sweep}: flips={flips} mean_sim={mu:.2f} std_sim={var**0.5:.2f}")
        if flips == 0:
            break
    return rho
