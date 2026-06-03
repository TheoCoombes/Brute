"""Layer tests — bitwise forward correctness vs ±1 reference + BEP backward wiring.

Forward math is unchanged (compared to a ±1 reference); backward now threads
**binary desired activations** (bit1) and accumulates **integer** ``ΔH`` into the
parameters' hidden weights ``H`` (no float ``q``).
"""
import torch
import pytest

import brute
import bep
from bep import BepConfig, BepOptimizer, pm1_int
from layers import (BSR, BooleanLinear, DiagBind, EpisodicSlotMemory,
                    HopfieldBank, ResidualMerge, TokenCodebook, bind_mask,
                    pow2_decay_palette, pow2_decay_shifts)
from vsa import bind, hamming_similarity, random_hypervectors, to_bit1


def _rand_bits(*shape, seed=0):
    return random_hypervectors(shape[0], shape[1],
                               generator=torch.Generator().manual_seed(seed))


class TestBooleanLinear:
    def test_forward_matches_pm1_reference(self):
        lin = BooleanLinear(96, 48, name="lin", generator=torch.Generator().manual_seed(1))
        x = _rand_bits(10, 96, seed=2)
        out, z = lin.forward(x)
        z_ref = pm1_int(x, torch.int32).float() @ pm1_int(lin.W.bit, torch.int32).float().t()
        assert z.dtype == torch.int32
        assert torch.equal(z.float(), z_ref)
        assert torch.equal(pm1_int(out).float(), torch.where(z_ref >= 0, 1.0, -1.0))

    def test_backward_binary_desired_and_integer_update(self):
        lin = BooleanLinear(96, 48, name="lin", generator=torch.Generator().manual_seed(1))
        x = _rand_bits(10, 96, seed=2)
        lin.forward(x)
        a_star = _rand_bits(10, 48, seed=3)
        H0 = lin.W.H.clone()
        a_in = lin.backward(a_star)
        assert a_in.dtype == brute.bit1 and a_in.shape == (10, 96)
        assert torch.any(lin.W.H != H0)
        dref = 2 * (pm1_int(a_star, torch.int32).t() @ pm1_int(x, torch.int32))
        assert torch.equal((lin.W.H - H0).to(torch.int32), dref)

    def test_learns_supervised_binary_projection(self):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(21)
        x = random_hypervectors(256, 96, generator=g)
        proto = random_hypervectors(8, 96, generator=g)
        y = hamming_similarity(x, proto).argmax(dim=1)
        lin = BooleanLinear(96, 96, name="lin", generator=torch.Generator().manual_seed(22))
        opt = BepOptimizer(lin.params(), BepConfig())
        from bep import mux
        for _ in range(100):
            ell, _ = lin.forward(x)
            logits = brute.fast.matmul(ell, proto)          # (256, 8)
            logit_t = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            other = logits.clone().float(); other.scatter_(1, y.unsqueeze(1), float("-inf"))
            trig = (logit_t - other.max(1).values) < 0.1 * 96
            sel = to_bit1(trig.unsqueeze(1).expand(256, 96))
            des = mux(sel, proto[y], ell)                   # desired ℓ̂ = target prototype
            bep.set_active(trig); lin.backward(des); bep.set_active(None)
            opt.step()
        ell, _ = lin.forward(x)
        acc = float((brute.fast.matmul(ell, proto).argmax(1) == y).float().mean())
        assert acc > 0.75


class TestDiagBind:
    def test_forward_is_elementwise_xnor(self):
        db = DiagBind(128, name="db", generator=torch.Generator().manual_seed(3))
        c = _rand_bits(6, 128, seed=4)
        out = db.forward(c)
        assert torch.equal(pm1_int(out), pm1_int(c) * pm1_int(db.m.bit))

    def test_backward_returns_binary_concept_and_updates_mask(self):
        db = DiagBind(64, name="db", generator=torch.Generator().manual_seed(3))
        c = _rand_bits(5, 64, seed=4)
        db.forward(c)
        a_star = _rand_bits(5, 64, seed=7)
        H0 = db.m.H.clone()
        g_c = db.backward(a_star)
        assert g_c.dtype == brute.bit1 and g_c.shape == (5, 64)
        assert torch.equal(pm1_int(g_c), pm1_int(a_star) * pm1_int(db.m.bit))
        assert torch.any(db.m.H != H0)

    def test_bind_mask_broadcasts(self):
        c = _rand_bits(4, 200, seed=5)
        m = random_hypervectors(1, 200, generator=torch.Generator().manual_seed(6))[0]
        out = bind_mask(c, m)
        assert torch.equal(pm1_int(out), pm1_int(c) * pm1_int(m))


class TestResidualMerge:
    def test_mux_selects_per_gate(self):
        rm = ResidualMerge(128, name="rm", p_open=0.5,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(5, 128, seed=8)
        trans = _rand_bits(5, 128, seed=9)
        out = rm.forward(skip, trans)
        g_open = pm1_int(rm.g.bit) > 0
        ref = torch.where(g_open, pm1_int(trans), pm1_int(skip))
        assert torch.equal(pm1_int(out), ref)

    def test_identity_when_all_gates_closed(self):
        rm = ResidualMerge(128, name="rm", p_open=0.0,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(5, 128, seed=8)
        trans = _rand_bits(5, 128, seed=9)
        out = rm.forward(skip, trans)
        assert torch.equal(pm1_int(out), pm1_int(skip))   # block ≈ identity at init

    def test_backward_routes_desired_to_both_branches(self):
        rm = ResidualMerge(64, name="rm", p_open=0.5,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(3, 64, seed=8)
        trans = _rand_bits(3, 64, seed=9)
        rm.forward(skip, trans)
        a_star = _rand_bits(3, 64, seed=10)
        a_skip, a_trans = rm.backward(a_star)
        # the transform keeps learning even where its gate is closed
        assert torch.equal(pm1_int(a_skip), pm1_int(a_star))
        assert torch.equal(pm1_int(a_trans), pm1_int(a_star))

    def test_gate_opens_when_transform_is_better(self):
        rm = ResidualMerge(64, name="rm", p_open=0.0,
                           generator=torch.Generator().manual_seed(7))
        skip = _rand_bits(8, 64, seed=8)
        trans = _rand_bits(8, 64, seed=9)
        rm.forward(skip, trans)
        before = pm1_int(rm.g.bit).clone()
        rm.backward(trans)                      # desired == transform → gate should open
        opt = BepOptimizer(rm.params(), BepConfig())
        for _ in range(20):
            rm.forward(skip, trans); rm.backward(trans); opt.step()
        assert int((pm1_int(rm.g.bit) != before).sum()) > 0


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
        k = pm1_int(bsr._cache["k_bit"], torch.int32)
        v = pm1_int(bsr._cache["v_bit"], torch.int32)
        q = pm1_int(bsr._cache["q_bit"], torch.int32)
        assoc = k * v
        shifts = pow2_decay_shifts(128)
        A = torch.zeros(2, 128, dtype=torch.int32)
        S = torch.empty(2, 12, 128, dtype=torch.int32)
        gate = torch.empty(2, 12, 128, dtype=torch.bool)
        for i in range(12):
            S[:, i, :] = torch.where(A >= 0, 1, -1)
            pred = k[:, i, :] * S[:, i, :]
            gate[:, i, :] = pred != v[:, i, :]
            shifted = torch.bitwise_right_shift(A, shifts)
            decayed = torch.where(shifts == 0, A, A - shifted)
            A = decayed + gate[:, i, :].to(torch.int32) * assoc[:, i, :]
        r_ref = q * S
        assert torch.equal(bsr._cache["gate"], gate)
        assert torch.equal(pm1_int(r, torch.int32), r_ref)

    def test_recurrence_is_sensitive_to_prefix_content(self):
        bsr = BSR(128, name="bsr", generator=torch.Generator().manual_seed(10))
        hv = random_hypervectors(3, 128, generator=torch.Generator().manual_seed(11))
        c_pm1 = torch.stack([
            torch.stack([pm1_int(hv[0]).float(), pm1_int(hv[2]).float()]),
            torch.stack([pm1_int(hv[1]).float(), pm1_int(hv[2]).float()]),
        ])
        c = to_bit1(c_pm1)
        r = bsr.forward(c)
        same_query_reads = pm1_int(r)[:, 1, :]
        assert not torch.equal(same_query_reads[0], same_query_reads[1])

    def test_streaming_matches_batched_forward(self):
        bsr = BSR(96, name="bsr", generator=torch.Generator().manual_seed(10))
        c = random_hypervectors(2 * 16, 96,
                                generator=torch.Generator().manual_seed(11)).reshape(2, 16, 96)
        batched = bsr.forward(c)
        bsr.reset_stream(2)
        streamed = [bsr.step(c[:, i, :]) for i in range(16)]
        streamed = to_bit1(torch.stack([pm1_int(x).float() for x in streamed], dim=1))
        assert torch.equal(pm1_int(streamed), pm1_int(batched))

    @pytest.mark.parametrize("length", [8, 16, 32, 64])
    def test_long_prefix_changes_late_read(self, length):
        bsr = BSR(128, name="bsr", generator=torch.Generator().manual_seed(10))
        hv = random_hypervectors(4, 128, generator=torch.Generator().manual_seed(11))
        seq_a = torch.stack([pm1_int(hv[0]).float()] * (length - 1) + [pm1_int(hv[3]).float()])
        seq_b = torch.stack([pm1_int(hv[1]).float()] * (length - 1) + [pm1_int(hv[3]).float()])
        c = to_bit1(torch.stack([seq_a, seq_b]))
        r = bsr.forward(c)
        assert not torch.equal(pm1_int(r)[0, -1], pm1_int(r)[1, -1])

    def test_backward_returns_concept_signal(self):
        bsr = BSR(64, name="bsr", generator=torch.Generator().manual_seed(10))
        c = random_hypervectors(2 * 8, 64,
                                generator=torch.Generator().manual_seed(11)).reshape(2, 8, 64)
        bsr.forward(c)
        a_star = random_hypervectors(2 * 8, 64,
                                     generator=torch.Generator().manual_seed(12)).reshape(2, 8, 64)
        H0 = [p.H.clone() for p in bsr.params()]
        g = bsr.backward(a_star)
        assert g.dtype == brute.bit1 and g.shape == (2, 8, 64)
        assert any(torch.any(p.H != h0) for p, h0 in zip(bsr.params(), H0))


class _FixedProjection:
    def __init__(self, out_bit):
        self.out_bit = out_bit

    def forward(self, a_bit):
        flat = self.out_bit.reshape(a_bit.shape[0], a_bit.shape[-1])
        return flat, torch.zeros(a_bit.shape[0], a_bit.shape[-1], dtype=torch.int32)

    def backward(self, a_star):
        return a_star

    def params(self):
        return []


class TestEpisodicSlotMemory:
    def test_controlled_top1_read_is_causal_and_exact(self):
        D, n = 64, 4
        g = torch.Generator().manual_seed(41)
        payload = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        keys = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        queries = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        q_pm1 = pm1_int(queries).float()
        q_pm1[0, 3] = pm1_int(keys).float()[0, 1]
        queries = to_bit1(q_pm1)
        pos = to_bit1(torch.ones(n, D))

        mem = EpisodicSlotMemory(D, name="epi", read_k=1)
        mem.Kc = _FixedProjection(keys)
        mem.Qc = _FixedProjection(queries)
        mem.Qp = _FixedProjection(to_bit1(torch.ones(1, n, D)))

        out = mem.forward(payload, pos)
        assert torch.equal(pm1_int(out)[0, 0], pm1_int(payload)[0, 0])
        assert torch.equal(pm1_int(out)[0, 3], pm1_int(payload)[0, 1])

    def test_batched_read_respects_configured_window(self):
        D, n = 64, 4
        g = torch.Generator().manual_seed(42)
        payload = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        keys = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        queries = random_hypervectors(n, D, generator=g).reshape(1, n, D)
        q_pm1 = pm1_int(queries).float()
        q_pm1[0, 3] = pm1_int(keys).float()[0, 1]
        queries = to_bit1(q_pm1)
        pos = to_bit1(torch.ones(n, D))

        mem = EpisodicSlotMemory(D, name="epi", read_k=1, n_slots=1)
        mem.Kc = _FixedProjection(keys)
        mem.Qc = _FixedProjection(queries)
        mem.Qp = _FixedProjection(to_bit1(torch.ones(1, n, D)))

        out = mem.forward(payload, pos)
        assert not torch.equal(pm1_int(out)[0, 3], pm1_int(payload)[0, 1])

    def test_streaming_matches_batched_forward(self):
        D, n = 96, 12
        g = torch.Generator().manual_seed(43)
        mem = EpisodicSlotMemory(D, name="epi", read_k=1, generator=g)
        c = random_hypervectors(2 * n, D, generator=g).reshape(2, n, D)
        pos = random_hypervectors(n, D, generator=g)

        batched = mem.forward(c, pos)
        mem.reset_stream(2, n)
        streamed = [mem.step(c[:, i, :], pos[i], n) for i in range(n)]
        streamed = to_bit1(torch.stack([pm1_int(x).float() for x in streamed], dim=1))
        assert torch.equal(pm1_int(streamed), pm1_int(batched))

    def test_backward_and_margin_update_address_projections(self):
        D, n = 64, 5
        g = torch.Generator().manual_seed(44)
        mem = EpisodicSlotMemory(D, name="epi", read_k=1, generator=g)
        c = random_hypervectors(2 * n, D, generator=g).reshape(2, n, D)
        pos = random_hypervectors(n, D, generator=g)
        mem.forward(c, pos)
        g_c = mem.backward(random_hypervectors(2 * n, D, generator=g).reshape(2, n, D))
        assert g_c.dtype == brute.bit1 and g_c.shape == (2, n, D)
        H0 = {p.name: p.H.clone() for p in mem.params()}
        matched = torch.tensor([[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]])
        margin = mem.margin_loss(matched, theta_pos=1.0, theta_neg=-1.0)
        assert margin >= 0.0
        assert any(torch.any(p.H != H0[p.name]) for p in mem.params())


class TestHopfield:
    def test_read_shape_and_determinism(self):
        hb = HopfieldBank(96, n_slots=32, top_k=5, name="hop",
                          generator=torch.Generator().manual_seed(12))
        q = _rand_bits(7, 96, seed=13)
        r1 = hb.forward(q)
        r2 = hb.forward(q)
        assert r1.shape == (7, 96) and r1.dtype == brute.bit1
        assert torch.equal(pm1_int(r1), pm1_int(r2))

    def test_backward_trains_bank_not_query(self):
        hb = HopfieldBank(96, n_slots=32, top_k=5, name="hop",
                          generator=torch.Generator().manual_seed(12))
        q = _rand_bits(7, 96, seed=13)
        hb.forward(q)
        U0, P0 = hb.U.H.clone(), hb.P.H.clone()
        g_q = hb.backward(_rand_bits(7, 96, seed=15))
        assert g_q is None                                   # hard selection ⇒ no query signal
        assert torch.any(hb.U.H != U0)                       # payloads get loss signal
        assert torch.any(hb.P.H != P0)                       # keys get Hebbian signal

    def test_controlled_slot_retrieval(self):
        hb = HopfieldBank(128, n_slots=8, top_k=1, name="hop",
                          generator=torch.Generator().manual_seed(12))
        keys = random_hypervectors(8, 128, generator=torch.Generator().manual_seed(13))
        payloads = random_hypervectors(8, 128, generator=torch.Generator().manual_seed(14))
        hb.P.bit = keys
        hb.U.bit = payloads
        out = hb.forward(keys[3:4])
        assert torch.equal(pm1_int(out), pm1_int(payloads[3:4]))


class TestCodebook:
    def test_embed_and_decode_roundtrip_shapes(self):
        cb = TokenCodebook(40, 128, name="E", generator=torch.Generator().manual_seed(14))
        ids = torch.randint(0, 40, (3, 9))
        emb = cb.embed(ids)
        assert emb.shape == (3, 9, 128) and emb.dtype == brute.bit1
        logits = cb.decode(emb.reshape(27, 128))
        assert logits.shape == (27, 40) and logits.dtype == torch.int32
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
