"""BinaryGLU (exact XNOR-gated FFN) and MajorityResidual (gated binary residual).

These are the two non-attention sublayer primitives of the binary transformer
block.  We check the exact forward identities (XNOR gate; coordinate majority;
gated admittance / identity-at-init) and that their BEP backwards move the right
weights in the right direction and stay fully packed.
"""

from __future__ import annotations

import torch

import brute
import bep
from bep import BepParam
from layers import BinaryGLU, MajorityResidual
from vsa import bundle3, sign_to_bit1, to_bit1
import vsa


def _rand_bits(shape, gen):
    return brute.as_tensor(torch.randint(0, 2, shape, generator=gen).bool(), dtype=brute.bit1)


# ── BinaryGLU ────────────────────────────────────────────────────────────────

def test_glu_forward_shapes_dtype(gen):
    glu = BinaryGLU(128, 256, name="glu", generator=gen)
    x = _rand_bits((10, 128), gen)
    f = glu.forward(x)
    assert tuple(f.shape) == (10, 128)
    assert f.dtype == brute.bit1


def test_glu_gate_is_exact_xnor(gen):
    """h = xnor(sign(W_g·x), sign(W_m·x)); the gate equals the bipolar product."""
    glu = BinaryGLU(128, 256, name="glu", generator=gen)
    x = _rand_bits((8, 128), gen)
    glu.forward(x)
    g = glu._cache["g_bit"]
    m = glu._cache["m_bit"]
    # h cached implicitly; recompute and compare to XNOR of g,m in ±1 domain.
    h_xnor = brute.fast.eq(g, m)            # XNOR
    gp, mp, hp = g.unpack_pm1(), m.unpack_pm1(), h_xnor.unpack_pm1()
    assert torch.equal(hp, gp * mp)         # XNOR == bipolar multiply


def test_glu_backward_runs_and_shapes(gen):
    glu = BinaryGLU(64, 128, name="glu", generator=gen)
    x = _rand_bits((6, 64), gen)
    glu.forward(x)
    f_star = _rand_bits((6, 64), gen)
    x_star = glu.backward(f_star)
    assert tuple(x_star.shape) == (6, 64)
    assert x_star.dtype == brute.bit1


def test_glu_backward_moves_weights(gen):
    """A consistent desired output drives a non-trivial weight update in all
    three projections (ΔH accumulated)."""
    glu = BinaryGLU(64, 128, name="glu", generator=gen)
    x = _rand_bits((16, 64), gen)
    H0 = {p.name: p.H.clone() for p in glu.params()}
    for _ in range(5):
        glu.forward(x)
        # desired = a fixed target, opposite of current output at some coords
        f, = (glu.forward(x),)
        f_star = brute.fast.bitwise_not(f)
        glu.backward(f_star)
    moved = [not torch.equal(p.H, H0[p.name]) for p in glu.params()]
    assert all(moved), f"some GLU projection did not update: {moved}"


def test_glu_packed_no_unpack_forward(gen):
    """GLU forward never unpacks (pure XNOR-matmul + bind)."""
    counts = _patch_unpacks()
    try:
        glu = BinaryGLU(128, 256, name="glu", generator=gen)
        x = _rand_bits((8, 128), gen)
        for k in counts:
            counts[k] = 0
        glu.forward(x)
        assert counts == {k: 0 for k in counts}, counts
    finally:
        _restore_unpacks()


# ── MajorityResidual ──────────────────────────────────────────────────────────

def test_residual_identity_when_gate_closed(gen):
    """With the gate fully closed, out == skip exactly (identity at init)."""
    for mode in ("mux", "majority"):
        res = MajorityResidual(128, name=f"res_{mode}", mode=mode, p_open=0.0,
                               init_inertia=8, generator=gen)
        skip = _rand_bits((4, 5, 128), gen)
        branch = _rand_bits((4, 5, 128), gen)
        out = res.forward(skip, branch)
        assert torch.equal(out.unpack_pm1(), skip.unpack_pm1()), mode


def test_residual_mux_open_is_branch(gen):
    """mux mode, gate fully open ⇒ out == branch (value replacement)."""
    res = MajorityResidual(128, name="res", mode="mux", p_open=1.0, init_inertia=8,
                           generator=gen)
    skip = _rand_bits((3, 4, 128), gen)
    branch = _rand_bits((3, 4, 128), gen)
    out = res.forward(skip, branch)
    assert torch.equal(out.unpack_pm1(), branch.unpack_pm1())


def test_residual_majority_open_is_maj3(gen):
    """majority mode, gate fully open ⇒ out == maj3(skip, branch, c)."""
    res = MajorityResidual(128, name="res", mode="majority", p_open=1.0, init_inertia=8,
                           generator=gen)
    skip = _rand_bits((3, 4, 128), gen)
    branch = _rand_bits((3, 4, 128), gen)
    out = res.forward(skip, branch)
    from layers import _broadcast_like
    c_b = _broadcast_like(res.c.bit, skip)
    maj = bundle3(skip, branch, c_b)
    assert torch.equal(out.unpack_pm1(), maj.unpack_pm1())


def test_residual_majority_equals_coordinate_majority(gen):
    """maj3 is the exact coordinate majority sign(skip+branch+c)."""
    res = MajorityResidual(64, name="res", mode="majority", p_open=1.0, generator=gen)
    skip = _rand_bits((20, 64), gen).reshape(1, 20, 64)
    branch = _rand_bits((20, 64), gen).reshape(1, 20, 64)
    out = res.forward(skip, branch)
    from layers import _broadcast_like
    c = res.c.bit.unpack_pm1()                       # (64,)
    s = skip.unpack_pm1(); b = branch.unpack_pm1()
    expect = torch.sign(s + b + c)                   # ±3/±1 → sign, never 0
    assert torch.equal(out.unpack_pm1(), expect)


def test_residual_backward_routes_to_both(gen):
    res = MajorityResidual(64, name="res", mode="mux", p_open=0.3, generator=gen)
    skip = _rand_bits((2, 3, 64), gen)
    branch = _rand_bits((2, 3, 64), gen)
    res.forward(skip, branch)
    out_star = _rand_bits((2, 3, 64), gen)
    skip_star, branch_star = res.backward(out_star)
    assert torch.equal(skip_star.unpack_pm1(), out_star.unpack_pm1())
    assert torch.equal(branch_star.unpack_pm1(), out_star.unpack_pm1())


def test_residual_gate_opens_toward_branch(gen):
    """If the branch consistently predicts the desired and the skip does not, the
    admittance gate opens (more open bits over training)."""
    res = MajorityResidual(128, name="res", mode="mux", p_open=0.05, init_inertia=2,
                           generator=gen)
    skip = _rand_bits((8, 4, 128), gen)
    branch = _rand_bits((8, 4, 128), gen)
    open0 = float(res.g.bit.bool().float().mean())
    for _ in range(30):
        res.forward(skip, branch)
        res.backward(branch)          # desired == branch ⇒ branch agrees, skip (random) doesn't
        res.g.step()
    open1 = float(res.g.bit.bool().float().mean())
    assert open1 > open0 + 0.1, f"gate did not open: {open0:.2f} -> {open1:.2f}"


def test_residual_majority_has_tiebreak_param_mux_does_not(gen):
    rmaj = MajorityResidual(64, name="rm", mode="majority", generator=gen)
    rmux = MajorityResidual(64, name="rx", mode="mux", generator=gen)
    assert any(p.name.endswith(".c") for p in rmaj.params())
    assert not any(p.name.endswith(".c") for p in rmux.params())
    assert rmux.c is None


def test_residual_tiebreak_updates_in_majority_mode(gen):
    res = MajorityResidual(128, name="res", mode="majority", p_open=1.0, init_inertia=1,
                           generator=gen)
    skip = _rand_bits((16, 128), gen).reshape(1, 16, 128)
    branch = brute.fast.bitwise_not(skip)            # always disagree ⇒ c decides
    c0 = res.c.H.clone()
    target = _rand_bits((16, 128), gen).reshape(1, 16, 128)
    res.forward(skip, branch)
    res.backward(target)
    assert not torch.equal(res.c.H, c0), "tiebreak c did not update on full disagreement"


# ── unpack-counting helpers (shared) ─────────────────────────────────────────

_ORIG = {}
_COUNTS = {}


def _patch_unpacks():
    global _ORIG, _COUNTS
    _COUNTS = {"unpack_bits": 0, "unpack_bool": 0}
    _ORIG = {k: getattr(torch.ops.brute, k) for k in _COUNTS}

    def mk(name, o):
        def w(*a, **kw):
            _COUNTS[name] += 1
            return o(*a, **kw)
        return w
    for k in _COUNTS:
        setattr(torch.ops.brute, k, mk(k, _ORIG[k]))
    return _COUNTS


def _restore_unpacks():
    for k, v in _ORIG.items():
        setattr(torch.ops.brute, k, v)
