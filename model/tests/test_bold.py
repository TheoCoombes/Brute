"""BOLD optimizer tests — flip-rule correctness, learning, plasticity, checkpoint."""
import torch
import pytest

import brute
from bold import BoldConfig, BoldOptimizer, BoldParam, random_bit_param, signal_scale
from layers import TokenCodebook
from vsa import random_hypervectors, to_pm1


def test_flip_rule_on_agreeing_signal():
    """A bit flips iff accumulated evidence agrees in sign with it (Eq. 9)."""
    bit = brute.as_tensor(torch.tensor([True, False, True, False]), dtype=brute.bit1)
    p = BoldParam(bit, name="w")                    # pm1 = [+1, -1, +1, -1]
    # q same sign as the weight on coords 0,1 (agree → flip); opposite on 2,3.
    p.add_signal(torch.tensor([+5.0, -5.0, -5.0, +5.0]))
    p.apply(eta=5.0, threshold=1.0)                 # threshold reached in one step
    assert p.pm1.tolist() == [-1.0, +1.0, +1.0, -1.0]


def test_accumulator_is_int8_and_gates_subtle_flips():
    """Nothing is discarded: a small consistent signal accumulates and *eventually*
    flips once it crosses the threshold (no flip on the first step)."""
    bit = brute.as_tensor(torch.tensor([True, True, True, True]), dtype=brute.bit1)
    p = BoldParam(bit, name="w")
    assert p.m.dtype == torch.int8                  # integer tracker buffer
    # small agreeing signal (pm1 = +1): one step adds ≈eta, threshold=10 ⇒ no flip yet
    p.add_signal(torch.ones(4)); n1 = p.apply(eta=3.0, threshold=10.0)
    assert n1 == 0 and p.pm1.tolist() == [1, 1, 1, 1]
    # keep feeding the same consistent signal — it accumulates and eventually flips
    flipped = False
    for _ in range(10):
        p.add_signal(torch.ones(4))
        if p.apply(eta=3.0, threshold=10.0) > 0:
            flipped = True
            break
    assert flipped, "consistent sub-threshold signal never accumulated to a flip"


def test_accumulator_reset_on_flip():
    p = random_bit_param((100,), "w", generator=torch.Generator().manual_seed(0))
    p.add_signal(p.pm1 * 10.0)                       # all agree
    n = p.apply(eta=5.0, threshold=1.0)
    assert n == 100                                  # all flip (no cap)
    assert int((p.m == 0).sum().item()) == 100       # reset on flip (error feedback)
    assert abs(p.beta - 0.0) < 1e-6                  # β = unchanged/total


def test_signal_scale():
    assert abs(signal_scale(2) - 1.0) < 1e-6
    assert abs(signal_scale(8) - 0.5) < 1e-6


def test_bold_learns_classifier_teacher():
    """End-to-end: BOLD trains a codebook to imitate a fixed argmax-teacher."""
    torch.manual_seed(0)
    N, D, C = 512, 256, 8
    g = torch.Generator().manual_seed(1)
    chat = random_hypervectors(N, D, generator=g)
    teacher = random_hypervectors(C, D, generator=g)
    y = (chat @ teacher).argmax(dim=1)

    cb = TokenCodebook(C, D, name="E", generator=torch.Generator().manual_seed(2))
    opt = BoldOptimizer(cb.params(), BoldConfig(eta=1.0, threshold=6.0))
    inv = 1.0 / (D ** 0.5)

    def acc():
        return float(((cb.decode(chat).float() * inv).argmax(1) == y).float().mean())

    a0 = acc()
    for _ in range(120):
        logits = cb.decode(chat).float() * inv
        p = logits.softmax(1)
        oh = torch.zeros_like(p); oh.scatter_(1, y.unsqueeze(1), 1.0)
        cb.decode_backward((p - oh) * (inv / N), to_pm1(chat))
        opt.step()
    a1 = acc()
    assert a1 > a0 + 0.4, f"BOLD failed to learn: {a0:.3f} → {a1:.3f}"
    assert a1 > 0.8, f"expected > 0.8 (nearest-centroid ceiling), got {a1:.3f}"


def test_param_checkpoint_roundtrip():
    p = random_bit_param((32, 16), "w", generator=torch.Generator().manual_seed(0))
    p.add_signal(torch.randn(32, 16))
    p.apply(eta=1.0)
    sd = p.state_dict()
    q = random_bit_param((32, 16), "w", generator=torch.Generator().manual_seed(99))
    q.load_state_dict(sd)
    assert torch.equal(q.pm1, p.pm1)
    assert torch.equal(q.m, p.m)
    assert abs(q.beta - p.beta) < 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
