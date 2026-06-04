"""Tests for :mod:`vsa` — Vector-Symbolic Algebra primitives on packed brute.bit1.

Every test verifies an exact algebraic property of the VSA operators, not just
a smoke-test.  The encoding convention is: +1 ↔ True (bit 1), -1 ↔ False (bit 0).

Key identities tested:

* XNOR = bipolar multiply:  e(xnor(a,b)) = e(a)·e(b)
* bind is its own inverse:   bind(bind(a,b), b) == a
* bundle3 = coordinate majority of three vectors
* hamming_similarity = D − 2·Hamming for every pair
* sign_to_bit1: z >= 0 → True (sign(0) = +1)
* to_bit1 / to_pm1 round-trips
* balanced_hash_frame_bool: deterministic, balanced, per-id regenerable
* binary_equiangular_frame: shape, dtype, well-separated rows
* random_hypervectors: shape, dtype, reproducible under a fixed generator
* position_codes / hierarchical_position_codes: shape and distinctness
"""

from __future__ import annotations

import torch
import pytest
import brute
import vsa


def _rand(rows: int, D: int, gen: torch.Generator) -> brute.Tensor:
    """Convenience wrapper for random_hypervectors."""
    return vsa.random_hypervectors(rows, D, generator=gen)


# ── bind is XNOR = bipolar multiply ──────────────────────────────────────────────

def test_bind_is_xnor_equals_bipolar_product(gen):
    """bind(a,b) in ±1 domain equals elementwise product: e(bind(a,b)) = e(a)·e(b)."""
    D = 128
    a = _rand(8, D, gen)
    b = _rand(8, D, gen)
    bound_pm1 = vsa.bind(a, b).unpack_pm1()
    expected_pm1 = a.unpack_pm1() * b.unpack_pm1()
    assert torch.equal(bound_pm1, expected_pm1), (
        "bind(a,b) in ±1 domain must equal coordinate-wise product e(a)·e(b)"
    )


def test_bind_is_own_inverse(gen):
    """bind is its own inverse: bind(bind(a,b), b) == a (XNOR is self-inverse)."""
    D = 64
    a = _rand(6, D, gen)
    b = _rand(6, D, gen)
    roundtrip = vsa.bind(vsa.bind(a, b), b)
    assert torch.equal(roundtrip.unpack_pm1(), a.unpack_pm1()), (
        "bind(bind(a,b), b) must recover a exactly"
    )


def test_bind_commutative(gen):
    """bind is commutative: bind(a,b) == bind(b,a)."""
    D = 64
    a = _rand(4, D, gen)
    b = _rand(4, D, gen)
    assert torch.equal(vsa.bind(a, b).unpack_pm1(), vsa.bind(b, a).unpack_pm1()), (
        "bind must be commutative"
    )


# ── bundle3 = coordinate-wise majority ───────────────────────────────────────────

def test_bundle3_equals_majority(gen):
    """bundle3(a,b,c) equals coordinate-wise majority of the three ±1 vectors.

    With 3 odd-count inputs there are never ties, so majority = sign(sum).
    """
    D = 128
    a = _rand(1, D, gen)
    b = _rand(1, D, gen)
    c = _rand(1, D, gen)

    result = vsa.bundle3(a, b, c)

    pa, pb, pc = a.unpack_pm1(), b.unpack_pm1(), c.unpack_pm1()
    majority_bool = (pa + pb + pc) >= 0          # sign(0) = +1 consistent with convention
    expected = vsa.to_bit1(majority_bool)

    assert torch.equal(result.bool(), expected.bool()), (
        "bundle3 must equal the coordinate-wise majority sign of the three inputs"
    )


def test_bundle3_majority_no_ties(gen):
    """With 3 vectors, every coordinate has exactly 2-1 or 3-0 votes — no ties."""
    D = 128
    a = _rand(1, D, gen)
    b = _rand(1, D, gen)
    c = _rand(1, D, gen)

    pa, pb, pc = a.unpack_pm1(), b.unpack_pm1(), c.unpack_pm1()
    sums = (pa + pb + pc).abs()
    # With three ±1 values the absolute sum is always 1 or 3 (never 0 / tie)
    assert bool((sums == 1).logical_or(sums == 3).all()), (
        "Three ±1 vectors cannot produce a zero coordinate sum"
    )


# ── hamming_similarity = D − 2·Hamming ───────────────────────────────────────────

def test_hamming_similarity_matches_popcount(gen):
    """hamming_similarity(q, keys)[i,j] == D − 2·Hamming(q_i, keys_j)."""
    D = 64
    M, N = 5, 7
    q = _rand(M, D, gen)
    keys = _rand(N, D, gen)

    sim = vsa.hamming_similarity(q, keys)
    assert sim.shape == (M, N)

    for i in range(M):
        for j in range(N):
            ham = int((q[i] ^ keys[j]).popcount().item())
            expected = D - 2 * ham
            assert int(sim[i, j].item()) == expected, (
                f"sim[{i},{j}] should be {expected} (D-2·Ham) but got {sim[i,j].item()}"
            )


def test_hamming_similarity_self_is_D(gen):
    """hamming_similarity(q, q)[i,i] == D (every bit agrees with itself)."""
    D = 128
    q = _rand(6, D, gen)
    sim = vsa.hamming_similarity(q, q)
    diagonal = sim.diagonal()
    assert torch.equal(diagonal, torch.full((6,), D, dtype=diagonal.dtype)), (
        "Self-similarity must be D (zero Hamming distance)"
    )


# ── sign_to_bit1 ─────────────────────────────────────────────────────────────────

def test_sign_to_bit1_nonnegative_is_true(gen):
    """sign_to_bit1: z >= 0 → True (+1), z < 0 → False (-1); sign(0) = +1."""
    z = torch.tensor([-3, -1, 0, 1, 5], dtype=torch.int32)
    result = vsa.sign_to_bit1(z)
    expected_bool = torch.tensor([False, False, True, True, True])
    assert torch.equal(result.bool(), expected_bool), (
        "sign_to_bit1 must map z>=0 to True (including 0), negative to False"
    )


def test_sign_to_bit1_float_input(gen):
    """sign_to_bit1 works on float tensors with the same threshold convention."""
    z = torch.tensor([-1.0, -0.001, 0.0, 0.001, 2.0])
    result = vsa.sign_to_bit1(z)
    expected_bool = torch.tensor([False, False, True, True, True])
    assert torch.equal(result.bool(), expected_bool)


# ── to_bit1 / to_pm1 round-trips ─────────────────────────────────────────────────

def test_to_bit1_to_pm1_roundtrip_pm1_float():
    """to_pm1(to_bit1(x)) == x for a ±1 float tensor."""
    pm1 = torch.tensor([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    bit = vsa.to_bit1(pm1)
    back = vsa.to_pm1(bit)
    assert torch.equal(pm1, back), "±1 float → bit1 → ±1 float must be identity"


def test_to_bit1_from_bool():
    """to_bit1 accepts a bool tensor: True → bit True (+1), False → bit False (-1)."""
    b = torch.tensor([True, False, True, False, True])
    bit = vsa.to_bit1(b)
    assert torch.equal(bit.bool(), b), "bool input: True must map to True in bit1"


def test_to_bit1_from_positive_threshold():
    """to_bit1 thresholds at > 0: exactly-zero becomes False, any positive → True."""
    vals = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    bit = vsa.to_bit1(vals)
    expected = torch.tensor([False, False, False, True, True])  # 0.0 is NOT > 0
    assert torch.equal(bit.bool(), expected), (
        "to_bit1 must threshold at >0 (zero → False)"
    )


def test_to_pm1_true_is_plus1():
    """to_pm1: True (bit 1) → +1.0, False (bit 0) → -1.0."""
    b = torch.tensor([True, False, True])
    bit = vsa.to_bit1(b)
    pm1 = vsa.to_pm1(bit)
    assert pm1.tolist() == [1.0, -1.0, 1.0], (
        "True must decode to +1.0, False to -1.0"
    )


# ── balanced_hash_frame_bool ──────────────────────────────────────────────────────

def test_balanced_hash_frame_deterministic():
    """Same ids + same seed → identical bit matrix."""
    D = 128
    ids = torch.arange(20)
    a = vsa.balanced_hash_frame_bool(ids, D, seed=42)
    b = vsa.balanced_hash_frame_bool(ids, D, seed=42)
    assert torch.equal(a, b), "balanced_hash_frame_bool must be deterministic"


def test_balanced_hash_frame_different_seeds_differ():
    """Different seeds produce different bit matrices (with overwhelming probability)."""
    D = 128
    ids = torch.arange(10)
    a = vsa.balanced_hash_frame_bool(ids, D, seed=1)
    b = vsa.balanced_hash_frame_bool(ids, D, seed=2)
    assert not torch.equal(a, b), "Different seeds must produce different frames"


def test_balanced_hash_frame_even_D_half_ones():
    """For even D every row has exactly D//2 True bits."""
    D = 64
    ids = torch.arange(16)
    bits = vsa.balanced_hash_frame_bool(ids, D, seed=0)
    row_counts = bits.sum(dim=1)
    expected = torch.full((16,), D // 2, dtype=row_counts.dtype)
    assert torch.equal(row_counts, expected), (
        f"All rows must have exactly D//2={D//2} True bits for even D"
    )


def test_balanced_hash_frame_odd_D_half_ones():
    """For odd D every row has D//2 or D//2 + 1 True bits."""
    D = 65
    ids = torch.arange(10)
    bits = vsa.balanced_hash_frame_bool(ids, D, seed=3)
    counts = bits.sum(dim=1)
    lo, hi = D // 2, D // 2 + 1
    assert bool(((counts == lo) | (counts == hi)).all()), (
        f"Odd D={D}: every row must have {lo} or {hi} True bits"
    )


def test_balanced_hash_frame_per_id_regenerable():
    """Calling with a subset of ids gives the same rows as the full matrix at those ids."""
    D = 64
    full_ids = torch.arange(20)
    bits_full = vsa.balanced_hash_frame_bool(full_ids, D, seed=7)

    subset_ids = torch.tensor([3, 8, 15])
    bits_sub = vsa.balanced_hash_frame_bool(subset_ids, D, seed=7)

    for k, id_ in enumerate(subset_ids.tolist()):
        assert torch.equal(bits_sub[k], bits_full[id_]), (
            f"Subset row for id={id_} must match full-frame row"
        )


# ── binary_equiangular_frame ──────────────────────────────────────────────────────

def test_binary_equiangular_frame_shape_and_dtype(gen):
    """binary_equiangular_frame returns (C, D) float ±1 tensor."""
    C, D = 16, 64
    rho = vsa.binary_equiangular_frame(C, D, generator=gen)
    assert rho.shape == (C, D)
    assert rho.dtype == torch.float32
    assert torch.equal(rho.abs(), torch.ones(C, D)), (
        "All entries must be exactly ±1"
    )


def test_binary_equiangular_frame_well_separated(gen):
    """Off-diagonal pairwise inner products are well below D (well-separated rows)."""
    C, D = 16, 64
    rho = vsa.binary_equiangular_frame(C, D, n_sweeps=30, generator=gen)
    G = rho @ rho.T
    off_diag = G[~torch.eye(C, dtype=torch.bool)]
    max_abs_sim = off_diag.abs().max().item()
    # Well-separated: max |<ρ_i, ρ_j>| should be substantially less than D
    # A random ±1 frame would have E[|sim|] ≈ sqrt(D)~8; greedy should stay below D/2
    assert max_abs_sim < D // 2, (
        f"BEF rows should be well-separated; max |sim|={max_abs_sim} >= D//2={D//2}"
    )


# ── random_hypervectors ───────────────────────────────────────────────────────────

def test_random_hypervectors_shape_and_dtype(gen):
    """random_hypervectors returns a brute.bit1 tensor of the requested shape."""
    num, D = 10, 128
    hvs = vsa.random_hypervectors(num, D, generator=gen)
    assert hvs.dtype == brute.bit1
    assert list(hvs.shape) == [num, D]


def test_random_hypervectors_reproducible():
    """Same generator seed → identical bit1 tensor."""
    g1 = torch.Generator(device="cpu").manual_seed(999)
    g2 = torch.Generator(device="cpu").manual_seed(999)
    a = vsa.random_hypervectors(5, 64, generator=g1)
    b = vsa.random_hypervectors(5, 64, generator=g2)
    assert torch.equal(a.bool(), b.bool()), (
        "random_hypervectors must be reproducible under a fixed seed"
    )


def test_random_hypervectors_different_seeds_differ():
    """Different seeds produce different hypervectors (with high probability)."""
    g1 = torch.Generator(device="cpu").manual_seed(1)
    g2 = torch.Generator(device="cpu").manual_seed(2)
    a = vsa.random_hypervectors(5, 128, generator=g1)
    b = vsa.random_hypervectors(5, 128, generator=g2)
    assert not torch.equal(a.bool(), b.bool()), (
        "Different seeds should produce different hypervectors"
    )


# ── position_codes / hierarchical_position_codes ─────────────────────────────────

def test_position_codes_shape_and_dtype(gen):
    """position_codes returns bit1 of shape (n, D)."""
    D = 64
    n = 8
    base_pm1 = torch.randn(D, generator=gen).sign().float()
    codes = vsa.position_codes(base_pm1, n)
    assert codes.dtype == brute.bit1
    assert list(codes.shape) == [n, D]


def test_position_codes_distinct_across_positions(gen):
    """No two position codes are identical (shift gives distinct vectors)."""
    D = 64
    n = 16
    base_pm1 = torch.randn(D, generator=gen).sign().float()
    codes = vsa.position_codes(base_pm1, n)
    for i in range(n):
        for j in range(i + 1, n):
            ham = int((codes[i] ^ codes[j]).popcount().item())
            assert ham > 0, (
                f"position_codes[{i}] and position_codes[{j}] must differ"
            )


def test_hierarchical_position_codes_shape_and_dtype(gen):
    """hierarchical_position_codes returns bit1 of shape (n, D)."""
    D = 128
    n = 32
    chunk_base = torch.randn(D, generator=gen).sign().float()
    offset_base = torch.randn(D, generator=gen).sign().float()
    codes = vsa.hierarchical_position_codes(chunk_base, offset_base, n, chunk=8)
    assert codes.dtype == brute.bit1
    assert list(codes.shape) == [n, D]


def test_hierarchical_position_codes_distinct(gen):
    """Hierarchical position codes are distinct across positions."""
    D = 128
    n = 16
    chunk_base = torch.randn(D, generator=gen).sign().float()
    offset_base = torch.randn(D, generator=gen).sign().float()
    codes = vsa.hierarchical_position_codes(chunk_base, offset_base, n, chunk=4)
    for i in range(n):
        for j in range(i + 1, n):
            ham = int((codes[i] ^ codes[j]).popcount().item())
            assert ham > 0, (
                f"hierarchical_position_codes[{i}] and [{j}] must differ"
            )
