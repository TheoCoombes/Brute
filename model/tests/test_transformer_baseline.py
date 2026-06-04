"""Structural and comparison tests for the binary transformer vs the float baseline.

Covers:
* Binary attention score == real signed dot product of binarised ±1 vectors.
* ALiBi correspondence: binary integer bias and float baseline share the same
  recency ordering (argmax of nearest valid key for the recency-dominant head).
* Baseline learns prev-token copy to > 0.9 (slow: sanity for well-posedness).
* Binary-vs-baseline parity on prev-token copy (slow): both exceed 0.9.
* Fast structural IO comparison: binary never attends future (idx ≤ pos); float
  softmax weights are ~0 above the causal diagonal.
"""

from __future__ import annotations

import pytest
import torch

import brute
import vsa
from attention import BinaryMultiHeadAttention
from _helpers import (
    TransformerBaseline,
    retrieval_config,
    train_baseline,
    train_prev_token,
    prev_token_batch,
)
from model import BinaryTransformerLM


# ── binary attention score == real dot product ────────────────────────────────────

class TestBinaryScoreEqualsRealDot:
    """The binary score ⟨q,k⟩ = d_h − 2·Hamming is algebraically identical
    to the signed ±1 dot product of the binarised vectors."""

    def test_matmul_equals_pm1_dot_small(self, gen):
        D = 128
        q = vsa.random_hypervectors(8, D, generator=gen)   # (8, D) bit1
        k = vsa.random_hypervectors(6, D, generator=gen)   # (6, D) bit1
        scores = brute.fast.matmul(q, k)                   # (8, 6) int32
        assert scores.shape == (8, 6)
        for i in range(8):
            for j in range(6):
                ref = int(torch.dot(q[i].unpack_pm1(), k[j].unpack_pm1()).item())
                got = int(scores[i, j].item())
                assert got == ref, (
                    f"score[{i},{j}] = {got} but pm1-dot = {ref}; "
                    f"binary attention does not compute genuine signed scores"
                )

    def test_self_score_equals_dimension(self, gen):
        D = 128
        q = vsa.random_hypervectors(5, D, generator=gen)
        scores = brute.fast.matmul(q, q)
        # ⟨q, q⟩ = D (every bit agrees → pm1-dot = D)
        for i in range(5):
            assert int(scores[i, i].item()) == D, (
                f"self-score at {i} = {int(scores[i,i].item())}, expected {D}"
            )

    def test_score_range_is_bounded_by_d(self, gen):
        D = 128
        q = vsa.random_hypervectors(16, D, generator=gen)
        k = vsa.random_hypervectors(16, D, generator=gen)
        scores = brute.fast.matmul(q, k)
        assert int(scores.max().item()) <= D, "score exceeded D"
        assert int(scores.min().item()) >= -D, "score below -D"

    def test_antisymmetric_under_bit_flip(self, gen):
        # Flipping all bits of q (XOR with all-ones) negates the pm1 vector,
        # so the score should negate: score(~q, k) == -score(q, k).
        D = 128
        q = vsa.random_hypervectors(4, D, generator=gen)
        k = vsa.random_hypervectors(4, D, generator=gen)
        s = brute.fast.matmul(q, k)
        all_ones = brute.as_tensor(torch.ones(1, D, dtype=torch.bool), dtype=brute.bit1)
        q_neg = brute.fast.bitwise_xor(q, all_ones.expand_as(q))
        s_neg = brute.fast.matmul(q_neg, k)
        for i in range(4):
            for j in range(4):
                assert int(s_neg[i, j].item()) == -int(s[i, j].item()), (
                    f"score under bit-flip not negated at ({i},{j})"
                )


# ── ALiBi correspondence ─────────────────────────────────────────────────────────

class TestAlibiCorrespondence:
    """Integer ALiBi bias in the binary model and float ALiBi in the baseline
    both encode recency: closer key = higher adjusted score = selected by argmax."""

    def test_binary_alibi_bias_is_monotone_in_distance(self, gen):
        """For the recency-dominant head, the ALiBi penalty grows with i-j,
        so the raw bias term is monotonically increasing as j approaches i."""
        D = 64
        n = 10
        slope = 2 * D + 1               # recency-dominant: one step > any content swing
        mha = BinaryMultiHeadAttention(
            D, 1, name="m", attn_mode="hardmax", alibi=True,
            causal=True, causal_strict=True,
            alibi_slopes_override=(slope,), generator=gen,
        )
        relpos, keep = mha._geometry(n, torch.device("cpu"))
        qh = vsa.random_hypervectors(n, D, generator=gen)
        kh = vsa.random_hypervectors(n, D, generator=gen)
        ell = mha._scores_head(qh, kh, relpos, keep, slope)
        raw = brute.fast.matmul(qh, kh)   # content scores without ALiBi

        for i in range(2, n):
            # bias_j = ell[i,j] - raw[i,j]  = -slope*(i-j)
            biases = [int(ell[i, j].item()) - int(raw[i, j].item()) for j in range(i)]
            # as j increases (0 → i-1) distance shrinks, bias rises monotonically
            for a, b in zip(biases, biases[1:]):
                assert a <= b, (
                    f"ALiBi bias not monotone at row {i}: {biases}"
                )

    def test_binary_recency_head_selects_nearest_key(self, gen):
        """With a recency-dominant slope, hardmax must select the nearest causal key
        (j = i-1 for strict-causal; j = i for non-strict) for all positions."""
        D = 64
        B, n = 2, 8
        x = vsa.random_hypervectors(B * n, D, generator=gen).reshape(B, n, D)
        mha = BinaryMultiHeadAttention(
            D, 1, name="m", attn_mode="hardmax", alibi=True,
            causal=True, causal_strict=True,
            alibi_slopes_override=(2 * D + 1,), generator=gen,
        )
        mha.forward(x)
        idx = mha._cache["idx_heads"][0]   # (B, n)
        want = (torch.arange(n) - 1).clamp_min(0)   # i-1 (row0 picks 0)
        assert torch.equal(idx[:, 1:], want[1:].unsqueeze(0).expand(B, -1)), (
            "recency-dominant binary head did not select the nearest key (i-1)"
        )

    def test_float_baseline_recency_head_ordering(self):
        """Float baseline's last (recency-dominant) head: for query i>0 the
        nearest key j=i-1 has the highest attention weight among valid keys."""
        V, d_model, n_heads = 16, 64, 2
        n = 8
        bl = TransformerBaseline(V, d_model=d_model, n_heads=n_heads,
                                 n_layers=1, causal_strict=True, alibi_recency=True)
        bias = bl._bias(n, torch.device("cpu"))   # (H, n, n) with -inf above diagonal
        # For the last (recency-dominant) head and each query i >= 1,
        # j = i-1 should be the argmax among finite entries.
        h = n_heads - 1
        for i in range(1, n):
            row = bias[h, i]
            finite = row.isfinite()
            if not finite.any():
                continue
            argmax_j = int(row.masked_fill(~finite, float("-inf")).argmax().item())
            assert argmax_j == i - 1, (
                f"baseline recency head: query {i} should select j={i-1}, got {argmax_j}"
            )

    def test_binary_and_float_recency_argmax_agree(self, gen):
        """Both binary and float recency-dominant heads select the nearest key
        (j = i-1) for the same query positions — they encode the same recency
        ordering despite one using integers and one using floats."""
        D = 64
        n = 8
        # Binary head
        slope = 2 * D + 1
        mha = BinaryMultiHeadAttention(
            D, 1, name="m", attn_mode="hardmax", alibi=True,
            causal=True, causal_strict=True,
            alibi_slopes_override=(slope,), generator=gen,
        )
        x = vsa.random_hypervectors(n, D, generator=gen).reshape(1, n, D)
        mha.forward(x)
        bin_idx = mha._cache["idx_heads"][0][0]   # (n,)

        # Float baseline (same causal_strict, recency head is last)
        bl = TransformerBaseline(16, d_model=D, n_heads=2, n_layers=1,
                                 causal_strict=True, alibi_recency=True)
        bias = bl._bias(n, torch.device("cpu"))
        h = bl.n_heads - 1  # recency head
        float_argmax = torch.full((n,), 0, dtype=torch.long)
        for i in range(n):
            row = bias[h, i]
            finite = row.isfinite()
            if finite.any():
                float_argmax[i] = int(row.masked_fill(~finite, float("-inf")).argmax().item())

        # Both must agree for positions i >= 1 (i=0 is ambiguous: no j < 0)
        want = (torch.arange(n) - 1).clamp_min(0)
        assert torch.equal(bin_idx[1:], want[1:]), (
            f"binary argmax: {bin_idx.tolist()} != expected {want.tolist()}"
        )
        assert torch.equal(float_argmax[1:], want[1:]), (
            f"float argmax: {float_argmax.tolist()} != expected {want.tolist()}"
        )


# ── fast structural IO comparison (non-slow) ────────────────────────────────────

class TestStructuralIOComparison:
    """Both models share the same I/O contract; verify causal structure
    without training (fast)."""

    def test_binary_never_attends_future(self, gen):
        """For all (b, i), idx ≤ i: hardmax attention never selects a future key."""
        cfg = retrieval_config(V=16, D=128, n_heads=1, attn_mode="hardmax")
        m = BinaryTransformerLM(cfg)
        B, n = 3, 10
        ids = torch.randint(0, 16, (B, n))
        m.forward(ids)
        pos = torch.arange(n).unsqueeze(0)   # (1, n) broadcast over B
        for h, idx in enumerate(m.blocks[0].mha._cache["idx_heads"]):
            if idx is not None:
                assert bool((idx <= pos).all()), (
                    f"Binary model head {h} attended to a future key: idx={idx}"
                )

    def test_binary_output_is_2d_float(self):
        cfg = retrieval_config(V=16, D=128, n_heads=1, attn_mode="hardmax")
        m = BinaryTransformerLM(cfg)
        B, n = 2, 6
        ids = torch.randint(0, 16, (B, n))
        logits = m.forward(ids)
        assert logits.ndim == 2 and logits.shape == (B * n, 16), (
            f"binary model logits shape {logits.shape}, expected ({B*n}, 16)"
        )
        assert logits.dtype == torch.float32

    def test_float_baseline_causal_attention_zeros_above_diagonal(self):
        """Float baseline's softmax attention weights are ~0 above the
        causal diagonal (j >= i for causal_strict=True)."""
        V, d_model, n_heads = 16, 64, 2
        n = 8
        bl = TransformerBaseline(V, d_model=d_model, n_heads=n_heads,
                                 n_layers=1, causal_strict=True, alibi_recency=True)

        # Re-run the forward, capturing the attention weights
        captured: list[torch.Tensor] = []
        B = 2
        ids = torch.randint(0, V, (B, n))
        with torch.no_grad():
            x = bl.emb(ids)
            bias = bl._bias(n, ids.device)
            dh = bl.d_h
            for blk in bl.blocks:
                qkv = blk["qkv"](blk["ln1"](x)).view(B, n, 3, n_heads, dh)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
                att = (q @ k.transpose(-1, -2)) / (dh ** 0.5) + bias.unsqueeze(0)
                row_valid = (bias > -1e8).any(dim=-1, keepdim=True).float()
                att = att.softmax(dim=-1) * row_valid.unsqueeze(0)   # zero fully-masked rows
                captured.append(att)
                o = (att @ v).transpose(1, 2).reshape(B, n, n_heads * dh)
                x = x + blk["proj"](o)
                import torch.nn.functional as F
                x = x + blk["fc2"](F.gelu(blk["fc1"](blk["ln2"](x))))

        att_w = captured[0]   # (B, H, n, n)
        # causal_strict: j >= i is masked out; softmax of -inf = 0
        for b in range(B):
            for h in range(n_heads):
                for i in range(n):
                    for j in range(i, n):   # strictly-causal: j >= i is invalid
                        val = float(att_w[b, h, i, j].item())
                        assert val < 1e-6, (
                            f"baseline: non-zero attention at b={b} h={h} i={i} j={j}: {val}"
                        )

    def test_baseline_output_matches_io_contract(self):
        """Float baseline given (B, n) ids -> (B, n, V) logits; causal argmax
        is well-defined."""
        V, d_model = 16, 64
        bl = TransformerBaseline(V, d_model=d_model, n_heads=2, n_layers=1,
                                 causal_strict=True, alibi_recency=True)
        B, n = 2, 8
        ids = torch.randint(0, V, (B, n))
        with torch.no_grad():
            out = bl(ids)
        assert out.shape == (B, n, V)
        assert out.dtype == torch.float32
        # argmax is a valid token id
        pred = out.argmax(-1)
        assert pred.shape == (B, n)
        assert bool((pred >= 0).all()) and bool((pred < V).all())


# ── baseline learns prev-token copy (slow) ───────────────────────────────────────

@pytest.mark.slow
def test_baseline_prev_token_learns():
    """Float transformer baseline should reach > 0.9 on previous-token copy,
    confirming the task is well-posed for standard attention."""
    bl = TransformerBaseline(16, d_model=64, n_heads=2, n_layers=1,
                             causal_strict=True, alibi_recency=True)
    best = train_baseline(bl, prev_token_batch, steps=400)
    assert best > 0.9, (
        f"Baseline did not reach 0.9 on prev-token copy; best={best:.3f}. "
        f"The task may not be well-posed for standard attention."
    )


# ── binary vs baseline parity (slow) ────────────────────────────────────────────

@pytest.mark.slow
def test_binary_and_baseline_both_exceed_0p9_on_prev_token():
    """Both the binary transformer and the float baseline exceed 0.9 accuracy
    on previous-token copy: the binary transformer matches a real transformer
    on a non-local attention task."""
    # Binary model
    cfg = retrieval_config(V=16, D=128, n_heads=1, attn_mode="hardmax")
    binary_model = BinaryTransformerLM(cfg)
    binary_acc = train_prev_token(binary_model, steps=600)

    # Float baseline
    baseline = TransformerBaseline(16, d_model=64, n_heads=2, n_layers=1,
                                   causal_strict=True, alibi_recency=True)
    baseline_acc = train_baseline(baseline, prev_token_batch, steps=400)

    assert binary_acc > 0.9 and baseline_acc > 0.9, (
        f"Both models must exceed 0.9 on prev-token copy. "
        f"binary={binary_acc:.3f}, baseline={baseline_acc:.3f}"
    )
