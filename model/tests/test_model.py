"""End-to-end HÆMMR model tests — forward, BOLD backward, learning, sampling."""
import torch
import pytest

import brute
from bold import BoldConfig, BoldOptimizer
from data import make_lm_batches, repeating_sequence
from model import HaemmrConfig, HaemmrLM, IGNORE_INDEX


def tiny_model(**over):
    cfg = HaemmrConfig(vocab_size=over.pop("V", 16), D=over.pop("D", 128),
                       n_layers=over.pop("L", 1), d_ff=over.pop("d_ff", 256),
                       n_slots=over.pop("slots", 16), top_k=over.pop("top_k", 3),
                       seed=0, **over)
    return HaemmrLM(cfg, device="cpu")


class TestForward:
    def test_v2_has_position_free_decode_path(self):
        m = tiny_model(use_position=True)
        ids = torch.randint(0, 16, (2, 6))
        m.forward(ids)
        assert "decode_pos_pm1" not in m._fwd_cache
        assert "pos_pm1" not in m._fwd_cache
        assert m._fwd_cache["ell_pm1"].shape == (12, m.cfg.D)

    def test_content_only_mode_keeps_neutral_episodic_position_lane(self):
        m = tiny_model(use_position=False)
        pos = m._position_lane(5)
        assert torch.equal(pos.unpack_pm1(), torch.ones(5, m.cfg.D))

    def test_shapes_and_dtype(self):
        m = tiny_model()
        ids = torch.randint(0, 16, (3, 10))
        logits = m.forward(ids)
        assert logits.shape == (30, 16)
        assert logits.dtype == torch.float32
        bound = m.cfg.D * (1.0 + m.cfg.sem_weight)
        assert -bound <= float(logits.min()) and float(logits.max()) <= bound

    def test_config_validation(self):
        with pytest.raises(ValueError):
            HaemmrConfig(D=0)
        with pytest.raises(ValueError):
            HaemmrConfig(epi_read_k=0)
        with pytest.raises(ValueError):
            HaemmrConfig(label_smoothing=1.0)

    def test_all_params_are_bit1(self):
        m = tiny_model()
        for p in m.parameters():
            assert p.bit.dtype == brute.bit1
            assert p.bit._packed_buf is not None

    def test_param_count_reasonable(self):
        m = tiny_model()
        assert m.num_bit_parameters() == sum(
            int(torch.tensor(p.shape).prod()) for p in m.parameters())


class TestBackward:
    def test_loss_finite_and_signals_flow(self):
        m = tiny_model()
        ids = torch.randint(0, 16, (4, 12))
        tgt = torch.randint(0, 16, (4, 12))
        info = m.loss_and_backward(m.forward(ids), tgt)
        assert torch.isfinite(torch.tensor(info["loss"]))
        # every parameter received a flip-signal
        n_with_signal = sum(int((p.q != 0).any()) for p in m.parameters())
        assert n_with_signal == len(m.parameters())

    def test_ignore_index_masks_targets(self):
        m = tiny_model()
        ids = torch.randint(0, 16, (2, 8))
        tgt = torch.full((2, 8), IGNORE_INDEX)
        info = m.loss_and_backward(m.forward(ids), tgt)
        assert info["n_valid"] == 0
        assert info["loss"] == 0.0

    def test_label_smoothing_and_margin_supervision_run(self):
        m = tiny_model(label_smoothing=0.1, margin_weight=0.5)
        ids = torch.randint(0, 16, (2, 6))
        tgt = torch.randint(0, 16, (2, 6))
        matched = torch.tensor([[-1, 0, 1, 2, 3, 4], [-1, 0, 1, 2, 3, 4]])
        info = m.loss_and_backward(m.forward(ids), tgt, matched=matched)
        assert torch.isfinite(torch.tensor(info["loss"]))
        assert info["margin"] >= 0.0

    def test_optimizer_step_flips_bits(self):
        m = tiny_model()
        opt = BoldOptimizer(m.parameters(), BoldConfig(eta=1.0, threshold=1.0))
        ids = torch.randint(0, 16, (4, 12))
        tgt = torch.randint(0, 16, (4, 12))
        m.loss_and_backward(m.forward(ids), tgt)
        st = opt.step()
        assert st["n_flip"] > 0


class TestLearning:
    @pytest.mark.parametrize("L", [0, 1])
    def test_learns_deterministic_next_token(self, L):
        """Memorise next=(x+1)%cycle — must beat the uniform baseline decisively."""
        torch.manual_seed(0)
        cycle, V = 8, 16
        ids = repeating_sequence(cycle=cycle, n_tokens=8192)
        X, Y = make_lm_batches(ids, seq_len=16, mask_oov=False, shuffle=True)
        m = tiny_model(V=V, D=256, L=L, d_ff=512)
        opt = BoldOptimizer(m.parameters(), BoldConfig(eta=1.0, threshold=4.0))
        bs = 32
        for step in range(180):
            i = (step * bs) % (X.shape[0] - bs)
            info = m.loss_and_backward(m.forward(X[i:i+bs]), Y[i:i+bs])
            opt.step()
        assert info["acc"] > 0.8, f"L={L}: only reached acc {info['acc']:.3f}"

    def test_block_uses_previous_token_context(self):
        """Same query token and position; previous marker decides the answer."""
        torch.manual_seed(0)

        def make_batch(bs):
            marker = torch.randint(0, 2, (bs,))
            X = torch.full((bs, 4), 7, dtype=torch.long)
            X[:, 0] = marker
            X[:, 1] = 2
            X[:, 2] = 5
            X[:, 3] = 6
            Y = torch.full((bs, 4), IGNORE_INDEX, dtype=torch.long)
            Y[:, 1] = marker + 3
            return X, Y

        m = tiny_model(V=8, D=128, L=1, d_ff=256, slots=32,
                       use_position=False, gate_open=0.05, codebook_flip_scale=0.5)
        opt = BoldOptimizer(m.parameters(), BoldConfig(eta=3.0, threshold=8.0))
        for _ in range(120):
            X, Y = make_batch(64)
            m.loss_and_backward(m.forward(X), Y)
            opt.step()

        X, Y = make_batch(512)
        X[:256, 0] = 0
        X[256:, 0] = 1
        Y[:, 1] = X[:, 0] + 3
        info = m.metrics(m.forward(X), Y)
        assert info["acc"] > 0.9, f"context disambiguation only reached {info['acc']:.3f}"

    def test_induction_pattern_at_length_64(self):
        """If A is followed by B earlier, seeing A again should predict B."""
        torch.manual_seed(0)

        def make_batch(bs, seq_len=64):
            a = torch.randint(0, 4, (bs,))
            b = a + 4
            X = torch.full((bs, seq_len), 9, dtype=torch.long)
            Y = torch.full((bs, seq_len), IGNORE_INDEX, dtype=torch.long)
            X[:, 0] = a
            X[:, 1] = b
            X[:, -2] = a
            Y[:, -2] = b
            return X, Y

        m = tiny_model(V=16, D=128, L=1, d_ff=256, slots=32,
                       use_position=False, gate_open=0.05, codebook_flip_scale=0.5)
        opt = BoldOptimizer(m.parameters(), BoldConfig(eta=3.0, threshold=8.0))
        for _ in range(80):
            X, Y = make_batch(64)
            m.loss_and_backward(m.forward(X), Y)
            opt.step()

        X, Y = make_batch(512)
        info = m.metrics(m.forward(X), Y)
        assert info["acc"] > 0.9, f"induction only reached {info['acc']:.3f}"


class TestSamplingAndCheckpoint:
    def test_generate_runs(self):
        m = tiny_model()
        ids = torch.randint(0, 16, (1, 5))
        out = m.generate(ids, n_new=7, temperature=0.9, top_k=4)
        assert out.shape == (1, 12)
        assert out[:, :5].equal(ids)               # prompt preserved

    def test_state_dict_roundtrip_preserves_forward(self):
        m = tiny_model()
        ids = torch.randint(0, 16, (2, 9))
        before = m.forward(ids).clone()
        sd = m.state_dict()
        m2 = tiny_model()
        m2.load_state_dict(sd)
        after = m2.forward(ids)
        assert torch.equal(before, after)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
