"""Tests for the BEP RNN language-model example.

Runnable with::

    cd examples/bgpt && python -m pytest test_bep.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the example-local modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import pytest

import brute

from bef import generate_bef
from bep import BEPConfig, BEPLanguageModel, IGNORE_INDEX
from data import make_lm_batches, repeating_sequence


# ── Forward pass: shapes, dtypes, ±1 invariant ───────────────────────────────

class TestForward:
    def test_forward_shapes_and_dtypes(self):
        torch.manual_seed(0)
        V, K, B, T = 8, 16, 3, 5
        P = generate_bef(V, K, iters=2000, seed=0)
        model = BEPLanguageModel(V, K, P, config=BEPConfig(group_size_init=2), seed=0)

        tokens = torch.randint(0, V, (B, T), dtype=torch.long)
        state = model.forward(tokens)

        assert state.a_prev.shape == (B, T, K)
        assert state.a_curr.shape == (B, T, K)
        assert state.e_pm1.shape == (B, T, K)
        assert state.z.shape == (B, T, K)
        assert state.logits.shape == (B, T, V)
        assert state.z.dtype == torch.int32
        assert state.logits.dtype == torch.int32
        # ±1 invariant.
        for buf in (state.a_prev, state.a_curr, state.e_pm1):
            assert set(buf.unique().tolist()).issubset({-1, 1})

    def test_forward_matches_reference(self):
        """Recurrent forward must match a float ±1 reference."""
        torch.manual_seed(0)
        V, K, B, T = 6, 32, 2, 4
        P = generate_bef(V, K, iters=2000, seed=0)
        model = BEPLanguageModel(V, K, P, config=BEPConfig(group_size_init=4), seed=0)

        tokens = torch.randint(0, V, (B, T), dtype=torch.long)
        state = model.forward(tokens)

        # Reference: float ±1 arithmetic.
        W_xh = (model.H_xh >= 0).float() * 2 - 1
        W_hh = (model.H_hh >= 0).float() * 2 - 1
        Pf = model._P_pm1.float()
        a = torch.ones(B, K, dtype=torch.float32)
        emb = Pf[tokens]                                          # (B, T, K)
        for t in range(T):
            z = emb[:, t] @ W_xh.t() + a @ W_hh.t()
            a = (z > 0).float() * 2 - 1
            assert torch.equal(state.z[:, t].cpu(), z.to(torch.int32))
            assert torch.equal(state.a_curr[:, t].cpu(), a.to(torch.int8))
            ref_logits = a @ Pf.t()
            assert torch.equal(state.logits[:, t].cpu(), ref_logits.to(torch.int32))

    def test_init_state_is_plus_one(self):
        """a_prev at t=0 must be the all-ones init state."""
        torch.manual_seed(0)
        V, K = 5, 16
        P = generate_bef(V, K, iters=1000, seed=0)
        model = BEPLanguageModel(V, K, P, config=BEPConfig(group_size_init=2), seed=0)
        tokens = torch.randint(0, V, (4, 7), dtype=torch.long)
        state = model.forward(tokens)
        assert torch.equal(
            state.a_prev[:, 0],
            torch.ones(4, K, dtype=torch.int8),
        )
        # And a_prev[:, t+1] must equal a_curr[:, t].
        assert torch.equal(state.a_prev[:, 1:], state.a_curr[:, :-1])


# ── Step semantics ───────────────────────────────────────────────────────────

class TestStep:
    def test_step_no_trigger_returns_zero(self):
        """With r negative no sample triggers, weights must not change."""
        torch.manual_seed(0)
        V, K = 6, 16
        P = generate_bef(V, K, iters=2000, seed=0)
        model = BEPLanguageModel(
            V, K, P,
            config=BEPConfig(group_size_init=2, r=-1.0),
            seed=0,
        )
        H_xh_b = model.H_xh.clone()
        H_hh_b = model.H_hh.clone()

        tokens = torch.randint(0, V, (3, 5), dtype=torch.long)
        targets = torch.randint(0, V, (3, 5), dtype=torch.long)
        state = model.forward(tokens)
        info = model.step(state, targets)
        assert info["n_triggered"] == 0
        assert torch.equal(H_xh_b, model.H_xh)
        assert torch.equal(H_hh_b, model.H_hh)

    def test_step_modifies_weights_on_trigger(self):
        torch.manual_seed(0)
        V, K = 6, 16
        P = generate_bef(V, K, iters=2000, seed=0)
        # r=1.0 is aggressive: essentially every position triggers.
        model = BEPLanguageModel(
            V, K, P,
            config=BEPConfig(group_size_init=2, r=1.0),
            seed=0,
        )
        H_xh_b = model.H_xh.clone()
        H_hh_b = model.H_hh.clone()

        tokens = torch.randint(0, V, (3, 5), dtype=torch.long)
        targets = torch.randint(0, V, (3, 5), dtype=torch.long)
        state = model.forward(tokens)
        info = model.step(state, targets)
        assert info["n_triggered"] > 0
        assert not torch.equal(H_xh_b, model.H_xh) or not torch.equal(H_hh_b, model.H_hh)

    def test_ignore_index_masks_positions(self):
        """Positions with target == IGNORE_INDEX must not contribute."""
        torch.manual_seed(0)
        V, K = 6, 16
        P = generate_bef(V, K, iters=2000, seed=0)
        model = BEPLanguageModel(
            V, K, P,
            config=BEPConfig(group_size_init=2, r=1.0, use_reinforcement=False),
            seed=0,
        )
        tokens = torch.randint(0, V, (3, 5), dtype=torch.long)

        # All-ignored: weights must NOT change (no triggers, no signal).
        targets_all_ignore = torch.full((3, 5), IGNORE_INDEX, dtype=torch.long)
        state = model.forward(tokens)
        H_xh_b = model.H_xh.clone()
        H_hh_b = model.H_hh.clone()
        info = model.step(state, targets_all_ignore)
        assert info["n_triggered"] == 0
        assert info["n_seen"] == 0
        assert torch.equal(H_xh_b, model.H_xh)
        assert torch.equal(H_hh_b, model.H_hh)


# ── End-to-end learning sanity ───────────────────────────────────────────────

class TestLearning:
    def test_learns_repeating_sequence(self):
        """A binary RNN with sufficient hidden width should learn a tiny
        deterministic next-token rule far above the uniform baseline."""
        torch.manual_seed(0)
        cycle = 8
        V = 16
        K = 64
        T = 16

        ids = repeating_sequence(cycle=cycle, vocab_size=V, n_tokens=4096, seed=0)
        x, y = make_lm_batches(ids, seq_len=T, batch_size=32, seed=0, shuffle=False)

        P = generate_bef(V, K, iters=10000, seed=0)
        model = BEPLanguageModel(
            V, K, P,
            config=BEPConfig(r=0.5, nu=0.1, group_size_init=4, p_reinforce=0.5),
            seed=0,
        )

        init_acc = model.accuracy(x, y, batch_size=32)
        for ep in range(6):
            perm = torch.randperm(x.shape[0])
            xs, ys = x[perm], y[perm]
            for s in range(0, xs.shape[0], 32):
                state = model.forward(xs[s:s+32])
                model.step(state, ys[s:s+32])
        final_acc = model.accuracy(x, y, batch_size=32)
        # Uniform baseline is 1/V ≈ 0.0625; chance of the actual successor
        # is 1/cycle = 0.125 if the model knew nothing about recurrence.
        # We require clearly beating the unigram baseline.
        assert final_acc > init_acc + 0.2, (
            f"Expected learning on repeating sequence, got "
            f"{init_acc:.3f} → {final_acc:.3f}"
        )
        assert final_acc > 0.5, (
            f"Expected >50% on cycle={cycle} task, got {final_acc:.3f}"
        )


# ── brute.bit1 integration ───────────────────────────────────────────────────

class TestBruteIntegration:
    def test_packed_weights_are_bit1(self):
        torch.manual_seed(0)
        V, K = 4, 16
        P = generate_bef(V, K, iters=1000, seed=0)
        model = BEPLanguageModel(V, K, P, config=BEPConfig(group_size_init=2), seed=0)
        W_xh = model.visible_W_xh()
        W_hh = model.visible_W_hh()
        for W in (W_xh, W_hh):
            assert isinstance(W, brute.Tensor)
            assert W.dtype == brute.bit1
            assert W._packed_buf is not None
            # K=16 packs into a single 64-bit word.
            assert W._packed_buf.shape == (16, 1)

    def test_forward_z_is_int32(self):
        """Recurrent pre-activations must come back as int32 (the brute
        bit1 matmul output dtype, summed)."""
        torch.manual_seed(0)
        V, K = 4, 16
        P = generate_bef(V, K, iters=1000, seed=0)
        model = BEPLanguageModel(V, K, P, config=BEPConfig(group_size_init=2), seed=0)
        tokens = torch.randint(0, V, (2, 4), dtype=torch.long)
        state = model.forward(tokens)
        assert state.z.dtype == torch.int32
        assert state.logits.dtype == torch.int32

    def test_weight_cache_invalidates_after_update(self):
        """When H changes, the packed bit1 cache must refresh."""
        torch.manual_seed(0)
        V, K = 4, 16
        P = generate_bef(V, K, iters=1000, seed=0)
        model = BEPLanguageModel(
            V, K, P,
            config=BEPConfig(group_size_init=2, r=1.0, use_reinforcement=False),
            seed=0,
        )
        W0 = model.visible_W_xh().unpack_pm1().clone()
        # Drive an update.
        tokens = torch.randint(0, V, (4, 5), dtype=torch.long)
        targets = torch.randint(0, V, (4, 5), dtype=torch.long)
        state = model.forward(tokens)
        model.step(state, targets)
        # Force at least one sign flip in H_xh by saturating it.
        model.H_xh[0, 0] = -model.H_xh[0, 0].item() - 1
        W1 = model.visible_W_xh().unpack_pm1().clone()
        assert not torch.equal(W0, W1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
