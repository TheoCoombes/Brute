"""Layer tests — bitwise forward correctness vs ±1 reference + backward wiring."""
import torch
import pytest

import brute
from bold import BoldConfig, BoldOptimizer
from layers import (BSR, BooleanLinear, DiagBind, EpisodicSlotMemory,
                    HopfieldBank, ResidualMerge, TokenCodebook, bind_mask,
                    pow2_decay_palette)
from vsa import bind, hamming_similarity, random_hypervectors, to_bit1, to_pm1


def _rand_bits(*shape, seed=0):
    return random_hypervectors(shape[0], shape[1],
                               generator=torch.Generator().manual_seed(seed))


class TestBooleanLinear:
    def test_forward_matches_pm1_reference(self):
        lin = BooleanLinear(96, 48, name="lin", generator=torch.Generator().manual_seed(1))
        x = _rand_bits(10, 96, seed=2)
        out, z = lin.forward(x)
        z_ref = to_pm1(x) @ lin.W.pm1.t()
        assert z.dtype == torch.int32
        assert torch.equal(z.float(), z_ref)
        assert torch.equal(to_pm1(out), torch.where(z_ref >= 0, 1.0, -1.0))

    def test_backward_shapes_and_signal(self):
        lin = BooleanLinear(96, 48, name="lin", generator=torch.Generator().manual_seed(1))
        x = _rand_bits(10, 96, seed=2)
        lin.forward(x)
        S = torch.randn(10, 48)
        g = lin.backward(S)
        assert g.shape == (10, 96)
        assert torch.any(lin.W.q != 0)          # accumulated weight flip-signal
        # q_W = Sᵀ · a_pm1
        assert torch.allclose(lin.W.q, S.t() @ to_pm1(x), atol=1e-4)

    def test_learns_supervised_binary_projection(self):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(21)
        x = random_hypervectors(256, 96, generator=g)
        teacher = random_hypervectors(8, 96, generator=g)
        y = hamming_similarity(x, teacher).argmax(dim=1)
        lin = BooleanLinear(96, 8, name="lin", generator=torch.Generator().manual_seed(22))
        opt = BoldOptimizer(lin.params(), BoldConfig(eta=3.0, threshold=8.0))
        inv = 1.0 / (96 ** 0.5)
        for _ in range(80):
            _, z = lin.forward(x)
            logits = z.float() * inv
            p = logits.softmax(dim=1)
            oh = torch.zeros_like(p)
            oh.scatter_(1, y.unsqueeze(1), 1.0)
            lin.backward((p - oh) * (inv / x.shape[0]))
            opt.step()
        _, z = lin.forward(x)
        acc = float((z.argmax(dim=1) == y).float().mean())
        assert acc > 0.75


class TestDiagBind:
    def test_forward_is_elementwise_xnor(self):
        db = DiagBind(128, name="db", generator=torch.Generator().manual_seed(3))
        c = _rand_bits(6, 128, seed=4)
        out = db.forward(c)
        assert torch.equal(to_pm1(out), to_pm1(c) * db.m.pm1)

    def test_bind_mask_broadcasts(self):
        c = _rand_bits(4, 200, seed=5)
        m = random_hypervectors(1, 200, generator=torch.Generator().manual_seed(6))[0]
        out = bind_mask(c, m)
        assert torch.equal(to_pm1(out), to_pm1(c) * to_pm1(m))


class TestResidualMerge:
    def test_mux_selects_per_gate(self):
        rm = ResidualMerge(128, name="rm", p_open=0.5,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(5, 128, seed=8)
        trans = _rand_bits(5, 128, seed=9)
        out = rm.forward(skip, trans)
        g_open = rm.g.pm1 > 0
        ref = torch.where(g_open, to_pm1(trans), to_pm1(skip))
        assert torch.equal(to_pm1(out), ref)

    def test_identity_when_all_gates_closed(self):
        rm = ResidualMerge(128, name="rm", p_open=0.0,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(5, 128, seed=8)
        trans = _rand_bits(5, 128, seed=9)
        out = rm.forward(skip, trans)
        assert torch.equal(to_pm1(out), to_pm1(skip))   # block ≈ identity at init

    def test_backward_routes_signal_by_gate(self):
        rm = ResidualMerge(64, name="rm", p_open=0.5,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(3, 64, seed=8)
        trans = _rand_bits(3, 64, seed=9)
        rm.forward(skip, trans)
        S = torch.randn(3, 64)
        g_skip, g_trans = rm.backward(S)
        g_open = rm.g.pm1 > 0
        assert torch.equal(g_trans, S * g_open)
        assert torch.equal(g_skip, S * (~g_open))

    def test_gate_opens_when_transform_is_better(self):
        rm = ResidualMerge(64, name="rm", p_open=0.0,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(8, 64, seed=8)
        trans = _rand_bits(8, 64, seed=9)
        target = to_pm1(trans)
        out = rm.forward(skip, trans)
        rm.backward(to_pm1(out) - target)
        n = rm.g.apply(eta=5.0, threshold=1.0)
        assert n > 0


class TestBSR:
    def test_pow2_decay_palette_uses_shift_multipliers(self):
        decay = pow2_decay_palette(10, shifts=(1, 2, 0))
        assert torch.equal(decay[:4], torch.full((4,), 0.5))
        assert torch.equal(decay[4:7], torch.full((3,), 0.75))
        assert torch.equal(decay[7:], torch.ones(3))

    def test_forward_matches_recurrence_reference(self):
        bsr = BSR(128, name="bsr", generator=torch.Generator().manual_seed(10))
        c = random_hypervectors(2 * 12, 128,
                                generator=torch.Generator().manual_seed(11)).reshape(2, 12, 128)
        r = bsr.forward(c)
        # reference ±1 recurrence using the cached Boolean projections
        k = bsr._cache["k_pm1"]
        v = bsr._cache["v_pm1"]
        q = bsr._cache["q_pm1"]
        assoc = k * v
        A = torch.zeros(2, 128)
        S = torch.empty(2, 12, 128)
        gate = torch.empty(2, 12, 128)
        for i in range(12):
            S[:, i, :] = torch.where(A >= 0, 1.0, -1.0)
            pred = k[:, i, :] * S[:, i, :]
            gate[:, i, :] = (pred != v[:, i, :]).float()
            A = bsr.decay * A + gate[:, i, :] * assoc[:, i, :]
        r_ref = q * S
        assert torch.equal(bsr._cache["gate"], gate)
        assert torch.equal(to_pm1(r), r_ref)

    def test_recurrence_is_sensitive_to_prefix_content(self):
        """Identical query position, different prefix token → different BSR read.

        This catches the degenerate diagonal K/V case where
        (c⊗W_K)⊗(c⊗W_V) cancels c and the state cannot store content.
        """
        bsr = BSR(128, name="bsr", generator=torch.Generator().manual_seed(10))
        hv = random_hypervectors(3, 128, generator=torch.Generator().manual_seed(11))
        c_pm1 = torch.stack([
            torch.stack([to_pm1(hv[0]), to_pm1(hv[2])]),
            torch.stack([to_pm1(hv[1]), to_pm1(hv[2])]),
        ])
        c = to_bit1(c_pm1)
        r = bsr.forward(c)
        same_query_reads = to_pm1(r)[:, 1, :]
        assert not torch.equal(same_query_reads[0], same_query_reads[1])

    def test_streaming_matches_batched_forward(self):
        bsr = BSR(96, name="bsr", generator=torch.Generator().manual_seed(10))
        c = random_hypervectors(2 * 16, 96,
                                generator=torch.Generator().manual_seed(11)).reshape(2, 16, 96)
        batched = bsr.forward(c)
        bsr.reset_stream(2)
        streamed = [bsr.step(c[:, i, :]) for i in range(16)]
        streamed = to_bit1(torch.stack([to_pm1(x) for x in streamed], dim=1))
        assert torch.equal(to_pm1(streamed), to_pm1(batched))

    @pytest.mark.parametrize("length", [8, 16, 32, 64])
    def test_long_prefix_changes_late_read(self, length):
        bsr = BSR(128, name="bsr", generator=torch.Generator().manual_seed(10))
        hv = random_hypervectors(4, 128, generator=torch.Generator().manual_seed(11))
        seq_a = torch.stack([to_pm1(hv[0])] * (length - 1) + [to_pm1(hv[3])])
        seq_b = torch.stack([to_pm1(hv[1])] * (length - 1) + [to_pm1(hv[3])])
        c = to_bit1(torch.stack([seq_a, seq_b]))
        r = bsr.forward(c)
        assert not torch.equal(to_pm1(r)[0, -1], to_pm1(r)[1, -1])

    def test_backward_returns_concept_signal(self):
        bsr = BSR(64, name="bsr", generator=torch.Generator().manual_seed(10))
        c = random_hypervectors(2 * 8, 64,
                                generator=torch.Generator().manual_seed(11)).reshape(2, 8, 64)
        bsr.forward(c)
        g = bsr.backward(torch.randn(2, 8, 64))
        assert g.shape == (2, 8, 64)
        for p in bsr.params():
            assert torch.any(p.q != 0)


class _FixedProjection:
    def __init__(self, out_bit):
        self.out_bit = out_bit

    def forward(self, a_bit):
        flat = self.out_bit.reshape(a_bit.shape[0], a_bit.shape[-1])
        return flat, torch.zeros(a_bit.shape[0], a_bit.shape[-1])

    def backward(self, S):
        return torch.zeros_like(S)

    def params(self):
        return []


class TestEpisodicSlotMemory:
    def test_controlled_top1_read_is_causal_and_exact(self):
        D, n = 64, 4
        g = torch.Generator().manual_seed(41)
        payload = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        keys = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        queries = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        queries_pm1 = to_pm1(queries)
        queries_pm1[0, 3] = to_pm1(keys)[0, 1]    # t=3 should retrieve slot 1
        queries = to_bit1(queries_pm1)
        pos = to_bit1(torch.ones(n, D))
        neutral = to_bit1(torch.ones(1, n, D))

        mem = EpisodicSlotMemory(D, name="epi", read_k=1)
        mem.Kc = _FixedProjection(keys)
        mem.Qc = _FixedProjection(queries)
        mem.Qp = _FixedProjection(neutral)

        out = mem.forward(payload, pos)
        out_pm1 = to_pm1(out)
        assert torch.equal(out_pm1[0, 0], to_pm1(payload)[0, 0])  # no prior slot -> identity
        assert torch.equal(out_pm1[0, 3], to_pm1(payload)[0, 1])

    def test_batched_read_respects_configured_window(self):
        D, n = 64, 4
        g = torch.Generator().manual_seed(42)
        payload = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        keys = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        queries = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        queries_pm1 = to_pm1(queries)
        queries_pm1[0, 3] = to_pm1(keys)[0, 1]    # outside a one-slot window at t=3
        queries = to_bit1(queries_pm1)
        pos = to_bit1(torch.ones(n, D))
        neutral = to_bit1(torch.ones(1, n, D))

        mem = EpisodicSlotMemory(D, name="epi", read_k=1, n_slots=1)
        mem.Kc = _FixedProjection(keys)
        mem.Qc = _FixedProjection(queries)
        mem.Qp = _FixedProjection(neutral)

        out = mem.forward(payload, pos)
        assert not torch.equal(to_pm1(out)[0, 3], to_pm1(payload)[0, 1])

    def test_streaming_matches_batched_forward(self):
        D, n = 96, 12
        g = torch.Generator().manual_seed(43)
        mem = EpisodicSlotMemory(D, name="epi", read_k=1, generator=g)
        c = random_hypervectors(2 * n, D, generator=g).reshape(2, n, D)
        pos = random_hypervectors(n, D, generator=g)

        batched = mem.forward(c, pos)
        mem.reset_stream(2, n)
        streamed = [mem.step(c[:, i, :], pos[i], n) for i in range(n)]
        streamed = to_bit1(torch.stack([to_pm1(x) for x in streamed], dim=1))
        assert torch.equal(to_pm1(streamed), to_pm1(batched))

    def test_backward_and_margin_update_address_projections(self):
        D, n = 64, 5
        g = torch.Generator().manual_seed(44)
        mem = EpisodicSlotMemory(D, name="epi", read_k=1, generator=g)
        c = random_hypervectors(2 * n, D, generator=g).reshape(2, n, D)
        pos = random_hypervectors(n, D, generator=g)
        mem.forward(c, pos)
        g_c = mem.backward(torch.randn(2, n, D))
        assert g_c.shape == (2, n, D)
        matched = torch.tensor([[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]])
        margin = mem.margin_loss(matched, theta_pos=1.0, theta_neg=-1.0)
        assert margin >= 0.0
        assert torch.any(mem.Kc.W.q != 0)
        assert torch.any(mem.Qc.W.q != 0)
        assert torch.any(mem.Qp.W.q != 0)


class TestHopfield:
    def test_read_shape_and_determinism(self):
        hb = HopfieldBank(96, n_slots=32, top_k=5, name="hop",
                          generator=torch.Generator().manual_seed(12))
        q = _rand_bits(7, 96, seed=13)
        r1 = hb.forward(q)
        r2 = hb.forward(q)
        assert r1.shape == (7, 96) and r1.dtype == brute.bit1
        assert torch.equal(to_pm1(r1), to_pm1(r2))

    def test_backward_trains_bank_not_query(self):
        hb = HopfieldBank(96, n_slots=32, top_k=5, name="hop",
                          generator=torch.Generator().manual_seed(12))
        q = _rand_bits(7, 96, seed=13)
        hb.forward(q)
        g_q = hb.backward(torch.randn(7, 96))
        assert torch.equal(g_q, torch.zeros_like(g_q))   # hard selection ⇒ no query signal
        assert torch.any(hb.U.q != 0)                    # payloads get loss signal
        assert torch.any(hb.P.q != 0)                    # keys get Hebbian signal

    def test_hebbian_key_signal_pulls_selected_key_toward_query(self):
        hb = HopfieldBank(64, n_slots=1, top_k=1, name="hop",
                          generator=torch.Generator().manual_seed(12))
        q = _rand_bits(1, 64, seed=13)
        hb.P.bit = _rand_bits(1, 64, seed=14)
        hb.P._pm1 = None
        hb.forward(q)
        hb.backward(torch.zeros(1, 64))
        q_pm1 = to_pm1(q)
        p_pm1 = hb.P.pm1
        disagree = p_pm1 != q_pm1
        agree = ~disagree
        assert torch.equal(hb.P.q[disagree], p_pm1[disagree])
        assert torch.equal(hb.P.q[agree], -p_pm1[agree])

    def test_controlled_slot_retrieval(self):
        hb = HopfieldBank(128, n_slots=8, top_k=1, name="hop",
                          generator=torch.Generator().manual_seed(12))
        keys = random_hypervectors(8, 128, generator=torch.Generator().manual_seed(13))
        payloads = random_hypervectors(8, 128, generator=torch.Generator().manual_seed(14))
        hb.P.bit = keys
        hb.P._pm1 = None
        hb.U.bit = payloads
        hb.U._pm1 = None
        out = hb.forward(keys[3:4])
        assert torch.equal(to_pm1(out), to_pm1(payloads[3:4]))


class TestCodebook:
    def test_embed_and_decode_roundtrip_shapes(self):
        cb = TokenCodebook(40, 128, name="E", generator=torch.Generator().manual_seed(14))
        ids = torch.randint(0, 40, (3, 9))
        emb = cb.embed(ids)
        assert emb.shape == (3, 9, 128) and emb.dtype == brute.bit1
        logits = cb.decode(emb.reshape(27, 128))
        assert logits.shape == (27, 40) and logits.dtype == torch.int32
        # a token's embedding is maximally similar to its own codebook row
        single = cb.decode(cb.E.bit[5:6])
        assert int(single.argmax().item()) == 5

    def test_positioned_code_decodes_after_unbind(self):
        cb = TokenCodebook(64, 1024, name="E", generator=torch.Generator().manual_seed(14))
        base = (torch.randint(0, 2, (1024,), generator=torch.Generator().manual_seed(15)) * 2 - 1).float()
        pos = to_bit1(torch.roll(base, 7).unsqueeze(0))
        dressed = bind(cb.E.bit[11:12], pos)
        raw_target_sim = int(cb.decode(dressed)[0, 11].item())
        recovered = bind(dressed, pos)
        assert int(cb.decode(recovered).argmax().item()) == 11
        assert raw_target_sim < 512


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
