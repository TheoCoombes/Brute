"""Tests for the BEP example — runnable with `pytest test_bep.py`.

These tests stay inside `examples/bep/` and do not modify the global brute
test suite. Run from this directory with `pytest .`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the example-local modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import pytest

import brute

from bef import bef_stats, generate_bef
from bep import BEPConfig, BEPModel
from data import random_prototypes


# ── BEF ──────────────────────────────────────────────────────────────────────

class TestBEF:
    def test_shape_and_pm1(self):
        P = generate_bef(n_classes=4, n_features=32, iters=2000, seed=0)
        assert P.shape == (4, 32)
        assert set(P.unique().tolist()).issubset({-1.0, 1.0})

    def test_improves_over_random(self):
        torch.manual_seed(0)
        random_P = (torch.randint(0, 2, (8, 64)) * 2 - 1).float()
        random_stats = bef_stats(random_P)

        bef = generate_bef(n_classes=8, n_features=64, iters=20000, seed=0)
        bef_st = bef_stats(bef)

        # The BEF should have a more negative mean off-diagonal inner product
        # (better separation) than a random ±1 codebook of the same shape.
        assert bef_st["mean"] < random_stats["mean"] - 1.0

    def test_separation_lower_bound(self):
        """With C ≤ D, the optimal pairwise IP can be as low as ≈ -D + small.
        We just check the search ends up well below the random baseline.
        """
        C, D = 4, 32
        bef = generate_bef(n_classes=C, n_features=D, iters=50000, seed=1)
        st = bef_stats(bef)
        # Random would have mean ≈ 0; optimum is highly negative.
        assert st["mean"] < -D / 4


# ── BEP forward / step ───────────────────────────────────────────────────────

class TestBEPForward:
    def test_forward_shapes_and_dtypes(self):
        torch.manual_seed(0)
        P = generate_bef(3, 16, iters=2000, seed=0)
        model = BEPModel([8, 16, 16], P, config=BEPConfig(group_size_init=2), seed=0)

        x = torch.randint(0, 2, (5, 8), dtype=torch.int8) * 2 - 1
        state = model.forward(x)
        # 1 input + L=2 hidden activations.
        assert len(state.a) == 3
        assert len(state.z) == 2
        assert state.a[0].shape == (5, 8)
        assert state.a[1].shape == (5, 16)
        assert state.a[2].shape == (5, 16)
        assert state.z[0].dtype == torch.int32
        assert state.logits.shape == (5, 3)
        assert state.logits.dtype == torch.int32
        # ±1 invariant on activations
        for a in state.a:
            assert set(a.unique().tolist()).issubset({-1, 1})

    def test_forward_matches_reference(self):
        """The bit1 matmul forward must agree, bit for bit, with the obvious
        ±1 reference computation."""
        torch.manual_seed(1)
        P = generate_bef(4, 32, iters=2000, seed=0)
        model = BEPModel([16, 32, 32], P, config=BEPConfig(group_size_init=2), seed=1)
        x = (torch.randint(0, 2, (7, 16), dtype=torch.int8) * 2 - 1)

        state = model.forward(x)

        # Reference computation using float ±1 arithmetic.
        a = x.float()
        for l in range(model.L):
            W = (model.H[l] >= 0).float() * 2 - 1
            z = a @ W.t()
            a = (z > 0).float() * 2 - 1
            assert torch.equal(state.z[l].cpu(), z.to(torch.int32))
            assert torch.equal(state.a[l + 1].cpu(), a.to(torch.int8))
        ref_logits = a @ model._P_pm1.t()
        assert torch.equal(state.logits.cpu(), ref_logits.to(torch.int32))

    def test_step_no_trigger_returns_zero(self):
        """If we set r to a negative value, no sample triggers an update."""
        torch.manual_seed(0)
        P = generate_bef(4, 16, iters=2000, seed=0)
        model = BEPModel([8, 16, 16], P, config=BEPConfig(group_size_init=2, r=-1.0), seed=0)
        H_before = [h.clone() for h in model.H]

        x = (torch.randint(0, 2, (6, 8), dtype=torch.int8) * 2 - 1)
        y = torch.tensor([0, 1, 2, 3, 0, 1])
        state = model.forward(x)
        info = model.step(state, y)
        assert info["n_triggered"] == 0
        for h_b, h_a in zip(H_before, model.H):
            assert torch.equal(h_b, h_a)

    def test_step_modifies_weights_on_trigger(self):
        torch.manual_seed(0)
        P = generate_bef(4, 16, iters=2000, seed=0)
        # Aggressive r so essentially every sample triggers.
        model = BEPModel([8, 16, 16], P, config=BEPConfig(group_size_init=2, r=1.0), seed=0)
        H_before = [h.clone() for h in model.H]

        x = (torch.randint(0, 2, (6, 8), dtype=torch.int8) * 2 - 1)
        y = torch.tensor([0, 1, 2, 3, 0, 1])
        state = model.forward(x)
        info = model.step(state, y)
        assert info["n_triggered"] > 0
        changed = any(not torch.equal(b, a) for b, a in zip(H_before, model.H))
        assert changed


# ── End-to-end training sanity ───────────────────────────────────────────────

class TestBEPLearning:
    def test_single_layer_learns_prototypes(self):
        """Tiny sanity training run: BEP must beat random on prototypes."""
        x_tr, y_tr, x_te, y_te = random_prototypes(
            n_samples=2000, n_features=200, n_classes=4, flip_prob=0.3,
            test_frac=0.25, seed=0,
        )
        n_classes = int(y_tr.max().item() + 1)
        P = generate_bef(n_classes, 64, iters=20000, seed=0)
        model = BEPModel(
            [200, 64], P,
            config=BEPConfig(r=0.5, nu=0.1, group_size_init=4, p_reinforce=0.5),
            seed=0,
        )
        init_acc = model.accuracy(x_te, y_te)
        for ep in range(5):
            perm = torch.randperm(x_tr.shape[0])
            x_sh, y_sh = x_tr[perm], y_tr[perm]
            for s in range(0, x_sh.shape[0], 50):
                state = model.forward(x_sh[s:s+50])
                model.step(state, y_sh[s:s+50])
        final_acc = model.accuracy(x_te, y_te)
        assert final_acc > init_acc + 0.2, f"Expected learning, got {init_acc:.3f}→{final_acc:.3f}"
        # Should solve this easy task to at least 70%.
        assert final_acc > 0.7, f"Expected >70% test acc, got {final_acc:.3f}"

    def test_multi_layer_learns(self):
        """Two-layer model must still learn (worse than 1-layer is fine; just
        verifying the desired-activation back-prop doesn't break learning)."""
        x_tr, y_tr, x_te, y_te = random_prototypes(
            n_samples=2000, n_features=200, n_classes=4, flip_prob=0.3,
            test_frac=0.25, seed=0,
        )
        P = generate_bef(4, 64, iters=20000, seed=0)
        model = BEPModel(
            [200, 64, 64], P,
            config=BEPConfig(r=0.5, nu=0.1, group_size_init=4, p_reinforce=0.5),
            seed=0,
        )
        init_acc = model.accuracy(x_te, y_te)
        for ep in range(5):
            perm = torch.randperm(x_tr.shape[0])
            x_sh, y_sh = x_tr[perm], y_tr[perm]
            for s in range(0, x_sh.shape[0], 50):
                state = model.forward(x_sh[s:s+50])
                model.step(state, y_sh[s:s+50])
        final_acc = model.accuracy(x_te, y_te)
        assert final_acc > init_acc + 0.15, (
            f"Multi-layer BEP not learning: {init_acc:.3f}→{final_acc:.3f}"
        )


# ── brute.bit1 integration ───────────────────────────────────────────────────

class TestBruteIntegration:
    def test_packed_weights_are_bit1(self):
        """Visible weight cache returns brute.bit1 tensors with packed storage."""
        torch.manual_seed(0)
        P = generate_bef(3, 16, iters=2000, seed=0)
        model = BEPModel([8, 16], P, config=BEPConfig(group_size_init=2), seed=0)
        W = model.visible_weight(0)
        assert isinstance(W, brute.Tensor)
        assert W.dtype == brute.bit1
        assert W._packed_buf is not None
        # Packed shape: last logical dim K_in=8 → 1 int64 word (64-bit packs).
        assert W._packed_buf.shape == (16, 1)

    def test_forward_uses_bit1_matmul(self):
        """The matmul output should be int32 (the dtype the brute bit1 kernel
        returns)."""
        torch.manual_seed(0)
        P = generate_bef(3, 16, iters=2000, seed=0)
        model = BEPModel([8, 16, 16], P, config=BEPConfig(group_size_init=2), seed=0)
        x = (torch.randint(0, 2, (4, 8), dtype=torch.int8) * 2 - 1)
        state = model.forward(x)
        for z in state.z:
            assert z.dtype == torch.int32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
