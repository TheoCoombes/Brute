"""End-to-end HÆMMR model tests — forward, BOLD backward, learning, sampling."""
import torch
import torch.nn.functional as F
import pytest

import brute
from bold import BoldConfig, BoldOptimizer
from data import make_lm_batches, repeating_sequence
from model import HaemmrConfig, HaemmrLM, IGNORE_INDEX


# ── Minimal FP transformer for parity comparison ───────────────────────────────

class _CausalAttnBlock(torch.nn.Module):
    """Pre-norm causal self-attention + MLP block (no dropout)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class _MiniTransformer(torch.nn.Module):
    """Minimal causal transformer used as the FP baseline in comparison tests."""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int = 1, max_len: int = 256):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos   = torch.nn.Embedding(max_len, d_model)
        self.blocks = torch.nn.ModuleList([
            _CausalAttnBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)
        torch.nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))  # (B, T, V)


def tiny_model(**over):
    cfg = HaemmrConfig(vocab_size=over.pop("V", 16), D=over.pop("D", 128),
                       n_layers=over.pop("L", 1), d_ff=over.pop("d_ff", 256),
                       n_slots=over.pop("slots", 16), top_k=over.pop("top_k", 3),
                       seed=0, **over)
    return HaemmrLM(cfg, device="cpu")


class TestForward:
    def test_has_position_free_decode_path(self):
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

    def test_epi_chunk_epi_registers_in_config(self):
        cfg = HaemmrConfig(epi_chunk=32, epi_registers=16)
        assert cfg.epi_chunk == 32
        assert cfg.epi_registers == 16
        with pytest.raises(ValueError):
            HaemmrConfig(epi_chunk=0)
        with pytest.raises(ValueError):
            HaemmrConfig(epi_registers=-1)

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



class TestVsFPTransformer:
    """Binary HAEMMR should solve sequence learning tasks as reliably as a FP transformer.

    Two synthetic tasks are tested:

    A→B induction  (positions 0,1 and T-2):
        Sequence A B filler… A;  predict B at position T-2.
        Solved by BSR linear recurrence (accumulates the A→B bundle association).
        EpisodicSlotMemory alone cannot do this — its payload is the matched
        token's own concept, not the following token (no V projection), so it
        retrieves A not B.  BSR fills this role analogously to RWKV/RetNet.

    Content copy  (position 0 to T-2):
        Sequence A filler…;  predict A at position T-2.
        This IS a pure content-addressed lookup — the payload IS the answer —
        so EpisodicSlotMemory handles it without BSR.
    """

    V, T = 16, 32

    def test_fp_transformer_solves_induction(self):
        """Sanity-check: a vanilla causal FP transformer solves A→B induction."""
        torch.manual_seed(0)
        V, T = self.V, self.T

        def make_batch(bs):
            a = torch.randint(0, 4, (bs,))
            b = a + 4
            X = torch.full((bs, T), 9, dtype=torch.long)
            Y = torch.full((bs, T), IGNORE_INDEX, dtype=torch.long)
            X[:, 0] = a; X[:, 1] = b; X[:, -2] = a
            Y[:, -2] = b
            return X, Y

        model = _MiniTransformer(V, d_model=64, n_heads=2, d_ff=128, n_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(200):
            X, Y = make_batch(64)
            logits = model(X)
            valid = Y != IGNORE_INDEX
            loss = F.cross_entropy(logits[valid], Y[valid])
            opt.zero_grad(); loss.backward(); opt.step()
        X, Y = make_batch(512)
        with torch.no_grad():
            logits = model(X)
        valid = Y != IGNORE_INDEX
        acc = float((logits[valid].argmax(1) == Y[valid]).float().mean())
        assert acc > 0.85, f"FP transformer induction acc={acc:.3f} < 0.85"

    def test_haemmr_full_system_matches_fp_on_induction(self):
        """Full HAEMMR (EpisodicSlotMemory + BSR + HopfieldBank) must match the
        FP transformer on A→B induction within the same step budget.

        BSR provides the "what follows A" context that makes induction work;
        the episodic memory then refines exact-recall lookups.  Together they
        match FP causal attention at this scale.
        """
        torch.manual_seed(0)
        V, T = self.V, self.T

        def make_batch(bs):
            a = torch.randint(0, 4, (bs,))
            b = a + 4
            X = torch.full((bs, T), 9, dtype=torch.long)
            Y = torch.full((bs, T), IGNORE_INDEX, dtype=torch.long)
            X[:, 0] = a; X[:, 1] = b; X[:, -2] = a
            Y[:, -2] = b
            return X, Y

        m = tiny_model(V=V, D=128, L=1, d_ff=256, slots=32,
                       use_bsr=True, use_position=False,
                       gate_open=0.05, codebook_flip_scale=0.5,
                       epi_chunk=T)
        opt = BoldOptimizer(m.parameters(), BoldConfig(eta=3.0, threshold=8.0))
        for _ in range(200):
            X, Y = make_batch(64)
            m.loss_and_backward(m.forward(X), Y)
            opt.step()
        X, Y = make_batch(512)
        info = m.metrics(m.forward(X), Y)
        assert info["acc"] > 0.85, (
            f"HAEMMR full system induction acc={info['acc']:.3f} < 0.85"
        )

    def test_episodic_memory_content_copy_matches_fp(self):
        """EpisodicSlotMemory (BSR disabled) vs FP transformer on long-range copy.

        Content copy is what EpisodicSlotMemory is designed for: the query key at
        position T-2 matches the stored key at position 0, and the returned payload
        is exactly the answer.  Both models must exceed 80 % accuracy, confirming
        the binary content-addressed lookup is as capable as FP softmax attention
        for this class of task.
        """
        torch.manual_seed(1)
        V, T = self.V, self.T

        def make_batch(bs):
            a = torch.randint(0, 4, (bs,))
            X = torch.full((bs, T), 9, dtype=torch.long)
            Y = torch.full((bs, T), IGNORE_INDEX, dtype=torch.long)
            X[:, 0] = a
            Y[:, -2] = a
            return X, Y

        # FP transformer
        fp = _MiniTransformer(V, d_model=64, n_heads=2, d_ff=128, n_layers=1)
        fp_opt = torch.optim.Adam(fp.parameters(), lr=3e-3)
        for _ in range(300):
            X, Y = make_batch(64)
            logits = fp(X)
            valid = Y != IGNORE_INDEX
            loss = F.cross_entropy(logits[valid], Y[valid])
            fp_opt.zero_grad(); loss.backward(); fp_opt.step()

        # HAEMMR — episodic memory only (BSR off)
        m = tiny_model(V=V, D=128, L=1, d_ff=256, slots=32,
                       use_bsr=False, use_position=False,
                       gate_open=0.1, codebook_flip_scale=0.5,
                       epi_chunk=T)
        bold_opt = BoldOptimizer(m.parameters(), BoldConfig(eta=3.0, threshold=6.0))
        for _ in range(300):
            X, Y = make_batch(64)
            m.loss_and_backward(m.forward(X), Y)
            bold_opt.step()

        X, Y = make_batch(512)
        valid = Y != IGNORE_INDEX
        with torch.no_grad():
            fp_logits = fp(X)
        fp_acc = float((fp_logits[valid].argmax(1) == Y[valid]).float().mean())
        bm_info = m.metrics(m.forward(X), Y)
        assert fp_acc  > 0.80, f"FP transformer copy acc={fp_acc:.3f} < 0.80"
        assert bm_info["acc"] > 0.80, (
            f"HAEMMR episodic copy acc={bm_info['acc']:.3f} < 0.80"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
