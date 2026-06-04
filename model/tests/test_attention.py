"""Exact-mechanism tests for :class:`attention.BinaryMultiHeadAttention`.

These verify the attention *mechanism* is algebraically exact — the load-bearing
"attention works fully" evidence — independent of any training run:

* scores are the signed dot ``⟨q,k⟩ = d_h − 2·Hamming``;
* the causal mask (strict / non-strict) and integer ALiBi behave correctly,
  including recency selection;
* hardmax **exactly** copies the arg-max value row;
* the soft int8 vote bundle obeys the dominance theorem (a key whose weight
  exceeds the rest copies its value);
* the BEP value-path backward routes the desired output to the selected source;
* the Hamming-margin score-lane objective provably converges;
* head split / merge is an exact packed round-trip and every output is bit1.
"""

from __future__ import annotations

import torch

import brute
import attention as A
from attention import BinaryMultiHeadAttention, signed_bundle, alibi_slopes, _split_head, _merge_heads
import bep
from bep import BepConfig, BepOptimizer
import vsa


def _rand_bits(rows, D, g):
    return vsa.random_hypervectors(rows, D, generator=g)


# ── scores == d_h − 2·Hamming ───────────────────────────────────────────────────

def test_scores_are_signed_dot(gen):
    D = 128
    q = _rand_bits(10, D, gen)
    k = _rand_bits(7, D, gen)
    scores = brute.fast.matmul(q, k)                      # (10,7) int32
    assert scores.shape == (10, 7)
    # reference signed dot: D − 2·Hamming(q_i, k_j)
    for i in range(10):
        for j in range(7):
            ham = int((q[i] ^ k[j]).popcount().item())
            assert int(scores[i, j].item()) == D - 2 * ham


def test_self_score_is_dimension(gen):
    # ⟨q,q⟩ = d_h (every bit agrees).
    D = 64
    q = _rand_bits(5, D, gen)
    s = brute.fast.matmul(q, q)
    assert torch.equal(s.diagonal(), torch.full((5,), D, dtype=s.dtype))


# ── causal masking ──────────────────────────────────────────────────────────────

def test_causal_nonstrict_keeps_self(gen):
    mha = BinaryMultiHeadAttention(64, 1, name="m", attn_mode="hardmax",
                                   alibi=False, causal=True, causal_strict=False, generator=gen)
    _, keep = mha._geometry(6, torch.device("cpu"))
    expect = torch.arange(6).unsqueeze(1) >= torch.arange(6).unsqueeze(0)   # j<=i
    assert torch.equal(keep, expect)


def test_causal_strict_excludes_self(gen):
    mha = BinaryMultiHeadAttention(64, 1, name="m", attn_mode="hardmax",
                                   alibi=False, causal=True, causal_strict=True, generator=gen)
    _, keep = mha._geometry(6, torch.device("cpu"))
    expect = torch.arange(6).unsqueeze(1) > torch.arange(6).unsqueeze(0)    # j<i
    assert torch.equal(keep, expect)


def test_hardmax_never_attends_future(gen):
    B, n, D = 2, 8, 64
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    mha = BinaryMultiHeadAttention(D, 1, name="m", attn_mode="hardmax",
                                   alibi=True, causal=True, generator=gen)
    mha.forward(x)
    idx = mha._cache["idx_heads"][0]                       # (B,n) argmax source
    pos = torch.arange(n).unsqueeze(0)
    assert bool((idx <= pos).all()), "hardmax selected a future key"


# ── ALiBi: monotonic recency, and recency-dominant selection ────────────────────

def test_alibi_slopes_scale_with_head_dim():
    s = alibi_slopes(4, 64)
    assert s[0].item() == 1                                # content head
    assert s[-1].item() >= 2 * 64                          # recency head dominates content
    assert bool((s[1:] >= s[:-1]).all())                   # non-decreasing


def test_alibi_bias_is_monotone_in_distance(gen):
    mha = BinaryMultiHeadAttention(64, 1, name="m", attn_mode="hardmax",
                                   alibi=True, causal=True, generator=gen)
    relpos, keep = mha._geometry(8, torch.device("cpu"))
    q = _rand_bits(8, 64, gen)
    k = _rand_bits(8, 64, gen)
    ell = mha._scores_head(q, k, relpos, keep, slope=10)
    # For a fixed query row, the ALiBi penalty grows with distance i-j: compare
    # the bias-only contribution by subtracting the raw content score.
    raw = brute.fast.matmul(q, k)
    for i in range(8):
        biases = [int(ell[i, j] - raw[i, j]) for j in range(i + 1)]   # = -10*(i-j)
        # as j increases 0→i the distance shrinks, so the bias rises monotonically
        assert biases == sorted(biases)
        if i > 0:
            assert biases[0] < biases[-1]                              # distant < near


def test_alibi_recency_selects_previous_token(gen):
    # A recency-dominant slope (> 2 d_h) makes strict-causal hardmax attend to i-1.
    B, n, D = 2, 8, 64
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    mha = BinaryMultiHeadAttention(D, 1, name="m", attn_mode="hardmax", alibi=True,
                                   causal=True, causal_strict=True,
                                   alibi_slopes_override=(2 * D + 1,), generator=gen)
    mha.forward(x)
    idx = mha._cache["idx_heads"][0]
    want = torch.arange(n).clamp_min(0)
    want = (torch.arange(n) - 1).clamp_min(0)              # i-1 (row0 → 0, masked)
    assert torch.equal(idx[:, 1:], want[1:].unsqueeze(0).expand(B, -1))


# ── hardmax exactly copies the arg-max value row ────────────────────────────────

def test_hardmax_copies_argmax_value(gen):
    B, n, D = 3, 7, 64
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    # value_proj=False, single head ⇒ output is the gathered raw concept = x[argmax].
    mha = BinaryMultiHeadAttention(D, 1, name="m", attn_mode="hardmax", alibi=True,
                                   causal=True, value_proj=False, generator=gen)
    a = mha.forward(x)
    idx = mha._cache["idx_heads"][0]                       # (B,n)
    for b in range(B):
        gathered = x[b].index_select(0, idx[b])            # (n,D) = x[b, argmax]
        assert torch.equal(a[b].unpack_pm1(), gathered.unpack_pm1())


# ── soft int8 vote bundle: dominance theorem ────────────────────────────────────

def test_signed_bundle_dominant_weight_copies_value(gen):
    # out[m] = sign(Σ_n W[m,n]·pm1(V[n])); if one weight exceeds the rest, copy it.
    B, M, N, D = 1, 1, 5, 128
    V = _rand_bits(N, D, gen).reshape(B, N, D)
    W = torch.ones(B, M, N, dtype=torch.int8)             # all +1 …
    W[0, 0, 2] = 10                                        # … except a dominant key
    out = signed_bundle(W, V, prefer_kernel=False)
    assert torch.equal(out[0, 0].unpack_pm1(), V[0, 2].unpack_pm1())


def test_signed_bundle_majority_when_balanced(gen):
    # Equal weights ⇒ coordinate majority of the selected values.
    B, M, N, D = 1, 1, 3, 96
    V = _rand_bits(N, D, gen).reshape(B, N, D)
    W = torch.ones(B, M, N, dtype=torch.int8)
    out = signed_bundle(W, V, prefer_kernel=False)
    vpm = V[0].unpack_pm1()                                # (3,D)
    majority = (vpm.sum(0) >= 0)                           # sign(Σ), sign(0)=+1
    assert torch.equal(out[0, 0].bool().reshape(-1), majority.reshape(-1))


def test_signed_bundle_zero_row_is_positive(gen):
    # All-zero weights ⇒ accumulator 0 ⇒ sign(0)=+1 everywhere.
    V = _rand_bits(4, 64, gen).reshape(1, 4, 64)
    W = torch.zeros(1, 1, 4, dtype=torch.int8)
    out = signed_bundle(W, V, prefer_kernel=False)
    assert bool(out[0, 0].bool().all())


def test_soft_band1_matches_hardmax_on_unique_argmax(gen):
    # With band=1 and a unique arg-max per row, soft attention copies the arg-max
    # value — the same row hardmax would gather (the dominance corollary).
    B, n, D = 2, 6, 64
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    common = dict(alibi=True, causal=True, causal_strict=True,
                  alibi_slopes_override=(2 * D + 1,), value_proj=False)
    hard = BinaryMultiHeadAttention(D, 1, name="h", attn_mode="hardmax", generator=gen, **common)
    soft = BinaryMultiHeadAttention(D, 1, name="s", attn_mode="soft", attn_band=1, generator=gen, **common)
    # share weights so q/k/v match exactly
    soft.Wq, soft.Wk = hard.Wq, hard.Wk
    ah = hard.forward(x)
    as_ = soft.forward(x)
    assert torch.equal(ah[:, 1:].unpack_pm1(), as_[:, 1:].unpack_pm1())


# ── BEP value-path backward routes desired to the selected source ───────────────

def test_value_backward_routes_to_source_hardmax(gen):
    # Recency head: query i reads source i-1, so the desired output o*_i must be
    # routed into v*_{i-1}. With value_proj=False, x* = v*, so x*_{i-1} == o*_i.
    B, n, D = 2, 6, 64
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    mha = BinaryMultiHeadAttention(D, 1, name="m", attn_mode="hardmax", alibi=True,
                                   causal=True, causal_strict=True, value_proj=False,
                                   alibi_slopes_override=(2 * D + 1,), generator=gen)
    mha.forward(x)
    a_star = _rand_bits(B * n, D, gen).reshape(B, n, D)
    x_star = mha.backward(a_star)
    # source j=i-1 (j>=1) is read only by query i (recency is a bijection); source
    # 0 also catches the masked row-0 hardmax arg-max, so check sources j>=1.
    for b in range(B):
        for i in range(2, n):
            assert torch.equal(x_star[b, i - 1].unpack_pm1(), a_star[b, i].unpack_pm1())


def test_value_backward_soft_uses_transpose_weights(gen):
    # Soft backward is signed_bundle(wᵀ, o*): with a single dominant source per
    # query, v*_j matches the desired of the query that selected it.
    B, n, D = 2, 5, 64
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    mha = BinaryMultiHeadAttention(D, 1, name="m", attn_mode="soft", attn_band=1,
                                   alibi=True, causal=True, causal_strict=True,
                                   value_proj=False, alibi_slopes_override=(2 * D + 1,),
                                   generator=gen)
    mha.forward(x)
    a_star = _rand_bits(B * n, D, gen).reshape(B, n, D)
    x_star = mha.backward(a_star)
    # recency ⇒ w_ij = band·1[j=i-1]; wᵀ routes o*_i to source i-1.
    for b in range(B):
        for i in range(1, n):
            assert torch.equal(x_star[b, i - 1].unpack_pm1(), a_star[b, i].unpack_pm1())


# ── Hamming-margin score-lane objective converges ───────────────────────────────

def test_margin_objective_converges(gen):
    # With a fixed input and a fixed supervised key, the margin objective must
    # drive ⟨q_i, k_match⟩ above θ⁺·d_h and the loss toward zero.
    B, n, D = 4, 6, 128
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    mha = BinaryMultiHeadAttention(D, 1, name="m", attn_mode="hardmax", alibi=False,
                                   causal=True, value_proj=True, generator=gen)
    opt = BepOptimizer(mha.params(), BepConfig())
    matched = torch.zeros(B, n, dtype=torch.long)         # every query attends to key 0
    matched[:, 0] = -1                                     # position 0 has no earlier key
    active = torch.ones(B * n, dtype=torch.bool)

    def mean_match_score():
        mha.forward(x)
        qh = mha._cache["q_head_bits"][0]; kh = mha._cache["k_head_bits"][0]
        tot = 0.0
        for b in range(B):
            s = brute.fast.matmul(qh[b], kh[b])
            tot += float(s[1:, 0].float().mean())
        return tot / B

    mha.forward(x)                                         # populate the cache
    first = mha.margin_loss(matched, active_rows=active)
    s0 = mean_match_score()
    for _ in range(60):
        mha.forward(x)
        mha.margin_loss(matched, active_rows=active)
        opt.step()
    last = mha.margin_loss(matched, active_rows=active)
    s1 = mean_match_score()
    assert last < 0.25 * first, f"margin loss did not fall: {first:.1f} -> {last:.1f}"
    # ⟨q_i, k_match⟩ climbs from ~0 (random) toward θ⁺·d_h (=0.5·D), a large rise.
    assert s1 > 0.45 * D, f"matched score did not approach theta_pos*d_h: {s1:.1f}"
    assert s1 > s0 + 25


def test_margin_returns_zero_without_valid_matches(gen):
    B, n, D = 2, 4, 128
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    mha = BinaryMultiHeadAttention(D, 2, name="m", attn_mode="soft", generator=gen)
    mha.forward(x)
    matched = torch.full((B, n), -1, dtype=torch.long)    # no supervision anywhere
    assert mha.margin_loss(matched) == 0.0


# ── packed head split / merge + dtype invariants ────────────────────────────────

def test_head_split_merge_roundtrip(gen):
    B, n, D, H = 2, 4, 256, 4
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    heads = [_split_head(x, h, H) for h in range(H)]
    assert all(list(hd.shape) == [B, n, D // H] for hd in heads)
    merged = _merge_heads(heads)
    assert list(merged.shape) == [B, n, D]
    assert torch.equal(merged.unpack_pm1(), x.unpack_pm1())


def test_forward_outputs_are_bit1(gen):
    B, n, D = 2, 5, 128
    x = _rand_bits(B * n, D, gen).reshape(B, n, D)
    for vp in (True, False):
        for mode in ("soft", "hardmax"):
            mha = BinaryMultiHeadAttention(D, 2, name="m", attn_mode=mode,
                                           value_proj=vp, generator=gen)
            a = mha.forward(x)
            assert a.dtype == brute.bit1
            assert list(a.shape) == [B, n, D]
            xs = mha.backward(a)
            assert xs.dtype == brute.bit1
            assert list(xs.shape) == [B, n, D]


def test_requires_packed_head_dim():
    import pytest
    with pytest.raises(ValueError):
        BinaryMultiHeadAttention(96, 3, name="m")          # d_h=32 not a multiple of 64
