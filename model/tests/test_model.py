"""Integration tests for :class:`model.BinaryTransformerLM`.

Covers:
* forward shape / dtype contract
* all eight mode combinations (value_proj × attn_mode × residual_mode)
* checkpoint round-trip (state_dict / load_state_dict + torch.save/load)
* generate: shape, top_k, repetition_window, determinism vs sampling
* memory contract: param_bytes() == 2 * num_bit_parameters()
* config validation (ValueError for bad configs, succeeds for valid ones)
* _synth_induction_matched: hand-verified small example
* metrics: loss / acc / ppl match manual computation
* Block backward returns bit1 (B,n,D) and leaves all params as valid BepParams
* learnability probes (slow): prev-token copy with hardmax and soft attention
* fast local-identity sanity (non-slow): readout/codebook co-adaptation check
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import brute
import vsa
from model import TransformerConfig, BinaryTransformerLM, Block, IGNORE_INDEX
from bep import BepConfig, BepOptimizer, BepParam
from _helpers import (
    retrieval_config,
    train_prev_token,
    prev_token_batch,
    masked_accuracy,
)


# ── helpers ──────────────────────────────────────────────────────────────────────

def _tiny_cfg(**overrides) -> TransformerConfig:
    """A minimal valid config for fast unit-test runs."""
    kw = dict(
        vocab_size=16, D=128, n_layers=1, n_heads=1, d_ff=256,
        attn_mode="hardmax", residual_mode="mux", value_proj=False,
        causal=True, causal_strict=True, alibi=True,
        sem_weight=0.0, seed=0,
    )
    kw.update(overrides)
    return TransformerConfig(**kw)


def _rand_ids(B: int, n: int, V: int = 16) -> torch.Tensor:
    return torch.randint(0, V, (B, n))


def _train_one_step(m: BinaryTransformerLM) -> dict:
    """One forward+backward+opt-step; return the info dict."""
    B, n, V = 4, 8, m.cfg.vocab_size
    ids = _rand_ids(B, n, V)
    tgt = _rand_ids(B, n, V)
    logits = m.forward(ids)
    info = m.loss_and_backward(logits, tgt)
    BepOptimizer(m.parameters(), BepConfig(r=m.cfg.r)).step()
    return info


# ── forward shape / dtype ────────────────────────────────────────────────────────

class TestForwardShapeDtype:
    def test_single_batch(self):
        cfg = _tiny_cfg()
        m = BinaryTransformerLM(cfg)
        ids = _rand_ids(1, 8)
        logits = m.forward(ids)
        assert logits.shape == (8, cfg.vocab_size), f"expected (8, {cfg.vocab_size}), got {logits.shape}"
        assert logits.dtype == torch.float32

    def test_multi_batch(self):
        cfg = _tiny_cfg()
        m = BinaryTransformerLM(cfg)
        B, n = 4, 12
        ids = _rand_ids(B, n)
        logits = m.forward(ids)
        assert logits.shape == (B * n, cfg.vocab_size), f"expected ({B*n}, {cfg.vocab_size}), got {logits.shape}"
        assert logits.dtype == torch.float32

    def test_logits_2d_not_3d(self):
        cfg = _tiny_cfg()
        m = BinaryTransformerLM(cfg)
        logits = m.forward(_rand_ids(3, 5))
        assert logits.ndim == 2, "forward should return 2-D (B*n, V)"


# ── eight mode combinations ──────────────────────────────────────────────────────

_MODE_COMBOS = [
    (vp, am, rm)
    for vp in (True, False)
    for am in ("soft", "hardmax")
    for rm in ("mux", "majority")
]


@pytest.mark.parametrize("value_proj,attn_mode,residual_mode", _MODE_COMBOS,
                         ids=[f"vp{int(vp)}-{am}-{rm}" for vp, am, rm in _MODE_COMBOS])
def test_eight_mode_combos(value_proj, attn_mode, residual_mode):
    """Build tiny model for each of 8 mode combos, run one training step."""
    cfg = TransformerConfig(
        vocab_size=48, D=128, n_heads=2, n_layers=2, d_ff=256,
        attn_mode=attn_mode, residual_mode=residual_mode,
        value_proj=value_proj, sem_weight=0.0, seed=0,
    )
    m = BinaryTransformerLM(cfg)
    B, n = 3, 6
    ids = _rand_ids(B, n, V=48)
    tgt = _rand_ids(B, n, V=48)
    logits = m.forward(ids)
    info = m.loss_and_backward(logits, tgt)
    BepOptimizer(m.parameters(), BepConfig(r=cfg.r)).step()
    assert math.isfinite(info["loss"]), f"loss not finite: {info['loss']}"
    assert 0.0 <= info["trigger_rate"] <= 1.0, (
        f"trigger_rate out of [0,1]: {info['trigger_rate']}"
    )


# ── checkpoint round-trip ────────────────────────────────────────────────────────

class TestCheckpointRoundTrip:
    def _train_model(self) -> tuple[BinaryTransformerLM, torch.Tensor]:
        cfg = _tiny_cfg()
        m = BinaryTransformerLM(cfg)
        _train_one_step(m)
        ids = _rand_ids(2, 6)
        return m, ids

    def test_state_dict_load_state_dict(self):
        m, ids = self._train_model()
        sd = m.state_dict()
        m2 = BinaryTransformerLM(m.cfg)
        m2.load_state_dict(sd)
        out1 = m.forward(ids)
        out2 = m2.forward(ids)
        assert torch.equal(out1, out2), "outputs differ after load_state_dict"

    def test_torch_save_load(self, tmp_path: Path):
        m, ids = self._train_model()
        sd = m.state_dict()
        path = tmp_path / "ckpt.pt"
        torch.save(sd, path)
        sd2 = torch.load(path, weights_only=False)
        m2 = BinaryTransformerLM(m.cfg)
        m2.load_state_dict(sd2)
        out1 = m.forward(ids)
        out2 = m2.forward(ids)
        assert torch.equal(out1, out2), "outputs differ after torch.save/load"

    def test_train_step_counter_preserved(self):
        m, _ = self._train_model()
        step_before = m._train_step
        sd = m.state_dict()
        m2 = BinaryTransformerLM(m.cfg)
        m2.load_state_dict(sd)
        assert m2._train_step == step_before, (
            f"train_step not preserved: {m2._train_step} != {step_before}"
        )


# ── generate ─────────────────────────────────────────────────────────────────────

class TestGenerate:
    def _model(self) -> BinaryTransformerLM:
        return BinaryTransformerLM(_tiny_cfg())

    def test_output_shape_single(self):
        m = self._model()
        ids = _rand_ids(1, 4)
        out = m.generate(ids, n_new=5)
        assert out.shape == (1, 9), f"expected (1, 9), got {out.shape}"

    def test_output_shape_batch(self):
        m = self._model()
        B, n0, n_new = 3, 6, 4
        ids = _rand_ids(B, n0)
        out = m.generate(ids, n_new=n_new)
        assert out.shape == (B, n0 + n_new), f"expected ({B}, {n0+n_new}), got {out.shape}"

    def test_prefix_preserved(self):
        m = self._model()
        ids = _rand_ids(2, 5)
        out = m.generate(ids, n_new=3)
        assert torch.equal(out[:, :5], ids), "prefix tokens were mutated"

    def test_top_k(self):
        m = self._model()
        ids = _rand_ids(1, 4)
        out = m.generate(ids, n_new=4, top_k=3)
        assert out.shape == (1, 8)

    def test_repetition_window(self):
        m = self._model()
        ids = _rand_ids(2, 4)
        out = m.generate(ids, n_new=4, repetition_window=2)
        assert out.shape == (2, 8)

    def test_deterministic_low_temperature(self):
        m = self._model()
        ids = _rand_ids(2, 4)
        torch.manual_seed(7)
        out1 = m.generate(ids, n_new=5, temperature=1e-6)
        torch.manual_seed(7)
        out2 = m.generate(ids, n_new=5, temperature=1e-6)
        assert torch.equal(out1, out2), "generate is not deterministic with same RNG state"

    def test_sampling_top_k_shape(self):
        m = self._model()
        ids = _rand_ids(2, 4)
        out = m.generate(ids, n_new=3, top_k=5, temperature=1.0)
        assert out.shape == (2, 7)


# ── memory contract ──────────────────────────────────────────────────────────────

class TestMemoryContract:
    def test_param_bytes_equals_twice_num_bit_parameters(self):
        m = BinaryTransformerLM(_tiny_cfg())
        assert m.param_bytes() == 2 * m.num_bit_parameters(), (
            f"param_bytes()={m.param_bytes()} != 2*num_bit_parameters()={2*m.num_bit_parameters()}"
        )

    def test_num_bit_parameters_equals_sum_of_param_sizes(self):
        m = BinaryTransformerLM(_tiny_cfg())
        total = sum(
            int(torch.tensor(list(p.shape)).prod().item())
            for p in m.parameters()
        )
        assert m.num_bit_parameters() == total, (
            f"num_bit_parameters()={m.num_bit_parameters()} != sum of shapes={total}"
        )

    def test_int16_H_buffers(self):
        m = BinaryTransformerLM(_tiny_cfg())
        for p in m.parameters():
            assert p.H.dtype == torch.int16, (
                f"param '{p.name}' has H.dtype={p.H.dtype}, expected int16"
            )


# ── config validation ────────────────────────────────────────────────────────────

class TestConfigValidation:
    def test_n_heads_not_dividing_D_raises(self):
        with pytest.raises(ValueError, match="n_heads must divide D"):
            TransformerConfig(vocab_size=16, D=128, n_heads=3, n_layers=1, d_ff=256)

    def test_d_h_not_multiple_of_64_raises(self):
        # D=96, n_heads=3: d_h = 32, which is not a multiple of 64
        with pytest.raises(ValueError, match="multiple of 64"):
            TransformerConfig(vocab_size=16, D=96, n_heads=3, n_layers=1, d_ff=256)

    def test_bad_attn_mode_raises(self):
        with pytest.raises(ValueError, match="attn_mode"):
            TransformerConfig(vocab_size=16, D=128, n_heads=1, n_layers=1, d_ff=256, attn_mode="bad")

    def test_bad_residual_mode_raises(self):
        with pytest.raises(ValueError, match="residual_mode"):
            TransformerConfig(vocab_size=16, D=128, n_heads=1, n_layers=1, d_ff=256, residual_mode="bad")

    def test_gate_open_out_of_range_raises(self):
        with pytest.raises(ValueError, match="gate_open"):
            TransformerConfig(vocab_size=16, D=128, n_heads=1, n_layers=1, d_ff=256, gate_open=1.5)

    def test_gate_open_negative_raises(self):
        with pytest.raises(ValueError, match="gate_open"):
            TransformerConfig(vocab_size=16, D=128, n_heads=1, n_layers=1, d_ff=256, gate_open=-0.1)

    def test_valid_config_constructs(self):
        # Should not raise
        cfg = TransformerConfig(
            vocab_size=32, D=128, n_heads=2, n_layers=2, d_ff=256,
            attn_mode="soft", residual_mode="majority", gate_open=0.5
        )
        assert cfg.D == 128


# ── _synth_induction_matched ─────────────────────────────────────────────────────

class TestSynthInductionMatched:
    """Hand-verified example: targets = [3, 5, 3, 7, 3, 2] (B=1, n=6).

    pos 0 (t=3): no earlier match -> -1
    pos 1 (t=5): no earlier match -> -1
    pos 2 (t=3): earlier match at pos 0 -> 0
    pos 3 (t=7): no earlier match -> -1
    pos 4 (t=3): earlier matches at pos 0, 2; most recent is 2 -> 2
    pos 5 (t=2): no earlier match -> -1
    """

    def _get_matched(self, targets: torch.Tensor) -> torch.Tensor:
        cfg = _tiny_cfg()
        m = BinaryTransformerLM(cfg)
        B, n = targets.shape
        ids = _rand_ids(B, n)
        m.forward(ids)
        return m._synth_induction_matched(targets)

    def test_basic_pattern(self):
        targets = torch.tensor([[3, 5, 3, 7, 3, 2]])
        matched = self._get_matched(targets)
        expected = torch.tensor([[-1, -1, 0, -1, 2, -1]])
        assert torch.equal(matched, expected), (
            f"matched={matched.tolist()} != expected={expected.tolist()}"
        )

    def test_all_unique_returns_all_minus_one(self):
        # All distinct tokens: no repeated follower, every position -> -1
        targets = torch.tensor([[0, 1, 2, 3, 4, 5]])
        matched = self._get_matched(targets)
        expected = torch.full_like(targets, -1)
        assert torch.equal(matched, expected)

    def test_repeated_consecutive_selects_previous(self):
        # [A, A, A]: pos1 sees pos0, pos2 sees max(0,1)=1
        targets = torch.tensor([[7, 7, 7]])
        matched = self._get_matched(targets)
        expected = torch.tensor([[-1, 0, 1]])
        assert torch.equal(matched, expected), (
            f"consecutive: got {matched.tolist()}, expected {expected.tolist()}"
        )

    def test_shape_matches_targets(self):
        targets = torch.randint(0, 16, (3, 8))
        matched = self._get_matched(targets)
        assert matched.shape == targets.shape

    def test_ignore_index_not_matched(self):
        # A target of IGNORE_INDEX (-1) should not be matched as a key
        targets = torch.tensor([[3, IGNORE_INDEX, 3, 3]])
        matched = self._get_matched(targets)
        # pos0 t=3: no match
        # pos1 t=IGNORE: -1 (invalid)
        # pos2 t=3: only valid earlier is pos0 (pos1 is IGNORE) -> 0
        # pos3 t=3: valid eariler are 0 and 2 -> max=2
        expected = torch.tensor([[-1, -1, 0, 2]])
        assert torch.equal(matched, expected), (
            f"got {matched.tolist()}, expected {expected.tolist()}"
        )


# ── metrics ──────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_match_manual_computation(self):
        """metrics() must reproduce manual NLL/CE, accuracy and PPL."""
        V = 4
        cfg = _tiny_cfg(vocab_size=V)
        m = BinaryTransformerLM(cfg)
        ids = _rand_ids(1, 3, V)
        logits = m.forward(ids)   # shape (3, V)

        # Two valid, one IGNORE
        targets = torch.tensor([[1, 2, IGNORE_INDEX]])
        result = m.metrics(logits, targets)

        tgt = targets.reshape(-1)
        valid = tgt != IGNORE_INDEX
        scaled = logits.float() * m.inv
        logp = torch.log_softmax(scaled, dim=1)
        safe = tgt.clamp_min(0)
        nll = -logp.gather(1, safe.unsqueeze(1)).squeeze(1)[valid]
        loss_manual = float(nll.mean().item())
        acc_manual = float((scaled.argmax(1)[valid] == tgt[valid]).float().mean().item())
        ppl_manual = float(torch.exp(torch.tensor(loss_manual)).item())

        assert abs(result["loss"] - loss_manual) < 1e-5, (
            f"loss mismatch: {result['loss']} vs {loss_manual}"
        )
        assert abs(result["acc"] - acc_manual) < 1e-5, (
            f"acc mismatch: {result['acc']} vs {acc_manual}"
        )
        assert abs(result["ppl"] - ppl_manual) < 1e-3, (
            f"ppl mismatch: {result['ppl']} vs {ppl_manual}"
        )
        assert result["n_valid"] == 2

    def test_metrics_all_ignore_returns_zero_loss(self):
        cfg = _tiny_cfg()
        m = BinaryTransformerLM(cfg)
        ids = _rand_ids(1, 4)
        logits = m.forward(ids)
        targets = torch.full((1, 4), IGNORE_INDEX, dtype=torch.long)
        result = m.metrics(logits, targets)
        assert result["loss"] == 0.0
        assert result["n_valid"] == 0


# ── Block backward dtype and params ─────────────────────────────────────────────

class TestBlockBackward:
    def test_backward_returns_bit1_desired(self, gen):
        B, n, D = 2, 6, 128
        cfg = _tiny_cfg(n_layers=2)
        x = vsa.random_hypervectors(B * n, D, generator=gen).reshape(B, n, D)
        blk = Block(cfg, idx=0, generator=gen)
        blk.forward(x)
        desired = vsa.random_hypervectors(B * n, D, generator=gen).reshape(B, n, D)
        x_star = blk.backward(desired)
        assert x_star.dtype == brute.bit1, f"expected bit1, got {x_star.dtype}"
        assert list(x_star.shape) == [B, n, D], (
            f"expected [{B},{n},{D}], got {list(x_star.shape)}"
        )

    def test_all_params_remain_int16(self):
        cfg = _tiny_cfg(n_layers=2)
        m = BinaryTransformerLM(cfg)
        ids = _rand_ids(2, 6)
        tgt = _rand_ids(2, 6)
        logits = m.forward(ids)
        m.loss_and_backward(logits, tgt)
        BepOptimizer(m.parameters(), BepConfig(r=cfg.r)).step()
        for p in m.parameters():
            assert isinstance(p, BepParam), f"'{p.name}' is not a BepParam"
            assert p.H.dtype == torch.int16, (
                f"param '{p.name}' H.dtype={p.H.dtype} after step"
            )


# ── fast local-identity sanity (non-slow) ────────────────────────────────────────

def test_local_identity_readout_sanity():
    """A gate-closed config (identity init) should learn the local identity task
    (target == current token) in ~150 steps, confirming readout/codebook training.

    This is intentionally non-slow: it uses a small batch, short sequences, and
    only ~150 steps, and should reach > 0.6 comfortably.
    """
    cfg = TransformerConfig(
        vocab_size=16, D=128, n_heads=1, n_layers=1, d_ff=256,
        attn_mode="hardmax", residual_mode="mux", value_proj=True,
        causal=True, causal_strict=False, alibi=True,
        gate_open=0.05, r=0.15, readout_warmup_steps=10,
        max_trigger_rate=0.6, block_init_inertia=6, block_update_clip=1,
        margin_weight=0.0, self_supervised_induction=False,
        sem_weight=0.0, seed=0,
    )
    m = BinaryTransformerLM(cfg)
    opt = BepOptimizer(m.parameters(), BepConfig(r=cfg.r))
    g = torch.Generator(device="cpu").manual_seed(99)
    B, n, V = 8, 8, 16

    best = 0.0
    for step in range(150):
        X = torch.randint(0, V, (B, n), generator=g)
        Y = X.clone()   # target == current token
        logits = m.forward(X)
        m.loss_and_backward(logits, Y)
        opt.step()
        if (step + 1) % 30 == 0:
            Xe = torch.randint(0, V, (B, n), generator=g)
            Ye = Xe.clone()
            pred = m.forward(Xe).reshape(B, n, V).argmax(-1)
            acc = float((pred == Ye).float().mean().item())
            best = max(best, acc)

    assert best > 0.6, (
        f"Local-identity task did not reach 0.6 in 150 steps; best={best:.3f}. "
        f"Readout / codebook training may be broken."
    )


# ── learnability (slow) ──────────────────────────────────────────────────────────

@pytest.mark.slow
def test_prev_token_hardmax_learns():
    """Binary model with hardmax attention learns prev-token copy to > 0.9."""
    cfg = retrieval_config(V=16, D=128, n_heads=1, attn_mode="hardmax")
    m = BinaryTransformerLM(cfg)
    best = train_prev_token(m, steps=600)
    assert best > 0.9, (
        f"hardmax model did not reach 0.9 on prev-token; best={best:.3f}"
    )


@pytest.mark.slow
def test_prev_token_soft_learns():
    """Binary model with soft attention learns prev-token copy to > 0.9."""
    cfg = retrieval_config(V=16, D=128, n_heads=1, attn_mode="soft")
    m = BinaryTransformerLM(cfg)
    best = train_prev_token(m, steps=600)
    assert best > 0.9, (
        f"soft model did not reach 0.9 on prev-token; best={best:.3f}"
    )
