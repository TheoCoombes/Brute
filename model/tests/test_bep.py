"""BEP training tests — integer hidden weight, binary errors, learning, checkpoint."""
import torch
import pytest

import brute
import bep
from bep import (BepConfig, BepOptimizer, BepParam, random_bit_param,
                 linear_backward, combine_desired, mux, pm1_int, signed_batch_sum)
from layers import BooleanLinear, TokenCodebook
from vsa import random_hypervectors, to_bit1


def test_param_is_integer_H_with_sign_visible_weight():
    """The only buffer is the integer ``H``; the visible weight is ``sign(H)``."""
    H = torch.tensor([3, -2, 0, -7], dtype=torch.int16)
    p = BepParam(H, name="w")
    assert p.H.dtype == torch.int16
    assert not hasattr(p, "q")
    assert not hasattr(p, "_pm1")
    # sign(0) = +1 (True); negative → False
    assert pm1_int(p.bit).tolist() == [1, -1, 1, -1]
    # param holds exactly one large buffer == H.nbytes
    assert p.param_bytes() == H.nbytes


def test_flip_happens_when_H_crosses_zero():
    H = torch.tensor([1, 1, -1, -1], dtype=torch.int16)
    p = BepParam(H, name="w")
    assert pm1_int(p.bit).tolist() == [1, 1, -1, -1]
    p.accumulate(torch.tensor([-3, 0, +3, 0], dtype=torch.int16))   # cross zero on 0 and 2
    assert pm1_int(p.bit).tolist() == [-1, 1, 1, -1]


def test_step_reports_sign_flips_and_clips():
    p = random_bit_param((64,), "w", generator=torch.Generator().manual_seed(0), bits=4)
    p.step()                                   # establish baseline sign snapshot
    p.accumulate(pm1_int(p.bit).to(torch.int16) * -10)   # push every bit across zero
    n = p.step()
    assert n == 64                             # every visible bit flipped
    assert int(p.H.abs().max()) <= (1 << (4 - 1))        # clipped to bit-width


def test_linear_backward_updates_H_and_returns_binary_desired():
    lin = BooleanLinear(32, 16, name="lin", generator=torch.Generator().manual_seed(1))
    x = random_hypervectors(8, 32, generator=torch.Generator().manual_seed(2))
    lin.forward(x)
    a_star = random_hypervectors(8, 16, generator=torch.Generator().manual_seed(3))
    H0 = lin.W.H.clone()
    a_in = lin.backward(a_star)
    assert a_in.dtype == brute.bit1 and a_in.shape == (8, 32)
    assert torch.any(lin.W.H != H0)            # integer weight moved
    # ΔH == 2 · (a*_out ⊗ a_inᵀ)  (binary outer product, Eq. 8-9)
    dref = 2 * (pm1_int(a_star, torch.int32).t() @ pm1_int(x, torch.int32))
    assert torch.equal((lin.W.H - H0).to(torch.int32), dref)


def test_combine_and_mux_are_packed_binary():
    a = to_bit1(torch.tensor([[True, True, False]]))
    b = to_bit1(torch.tensor([[True, False, False]]))
    tie = to_bit1(torch.tensor([[False, False, True]]))
    out = combine_desired(a, b, tie)           # agree→a; disagree(coord1)→tie
    assert pm1_int(out).tolist() == [[1, -1, -1]]
    sel = to_bit1(torch.tensor([[True, False, True]]))
    assert pm1_int(mux(sel, a, b)).tolist() == [[1, -1, -1]]
    assert combine_desired(a, None, tie) is a   # "no opinion"


def test_signed_batch_sum_matches_pm1_colsum():
    x = random_hypervectors(7, 12, generator=torch.Generator().manual_seed(5))
    assert torch.equal(signed_batch_sum(x), pm1_int(x, torch.int32).sum(0))


def test_bep_learns_classifier_teacher():
    """End-to-end: BEP margin-trigger + fixed prototypes trains a projection to
    imitate a fixed argmax-teacher classifier."""
    torch.manual_seed(0)
    N, D, C = 512, 256, 8
    g = torch.Generator().manual_seed(1)
    x = random_hypervectors(N, D, generator=g)
    proto = random_hypervectors(C, D, generator=g)          # fixed prototypes (codebook)
    y = (pm1_int(x, torch.int32).float() @ pm1_int(proto, torch.int32).float().t()).argmax(1)

    lin = BooleanLinear(D, D, name="lin", generator=torch.Generator().manual_seed(2))

    def acc():
        ell, _ = lin.forward(x)
        logits = brute.fast.matmul(ell, proto)
        return float((logits.argmax(1) == y).float().mean())

    a0 = acc()
    opt = BepOptimizer(lin.params(), BepConfig())
    for _ in range(120):
        ell, _ = lin.forward(x)
        logits = brute.fast.matmul(ell, proto)              # (N, C)
        logit_t = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        other = logits.clone().float()
        other.scatter_(1, y.unsqueeze(1), float("-inf"))
        trigger = (logit_t - other.max(1).values) < 0.1 * D
        sel = to_bit1(trigger.unsqueeze(1).expand(N, D))
        des = mux(sel, proto[y], ell)                       # desired ℓ̂ = target prototype
        bep.set_active(trigger)
        lin.backward(des)
        bep.set_active(None)
        opt.step()
    a1 = acc()
    assert a1 > a0 + 0.4, f"BEP failed to learn: {a0:.3f} → {a1:.3f}"
    assert a1 > 0.8, f"expected > 0.8 (nearest-centroid ceiling), got {a1:.3f}"


def test_param_checkpoint_roundtrips_H():
    p = random_bit_param((32, 16), "w", generator=torch.Generator().manual_seed(0))
    p.accumulate(torch.randint(-3, 3, (32, 16), dtype=torch.int16))
    sd = p.state_dict()
    assert set(sd) == {"H", "shape"}                        # integer H only
    q = random_bit_param((32, 16), "w", generator=torch.Generator().manual_seed(99))
    q.load_state_dict(sd)
    assert torch.equal(q.H, p.H)
    assert torch.equal(pm1_int(q.bit), pm1_int(p.bit))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
