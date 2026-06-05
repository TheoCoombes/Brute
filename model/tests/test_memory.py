"""Memory-footprint contracts — keep the model inside the project's scope.

The whole point of a 1-bit, BEP-trained model is a tiny, fixed memory budget:

* **Parameters** are *only* the int8 hidden weights ``H`` (~1 byte / 1-bit
  weight) — no float copies.
* **Training overhead is ~zero**: BEP carries no per-parameter optimiser state
  (no Adam moments, no float momentum), unlike a float transformer which needs
  ~4 bytes (fp32 weight) + ~8 bytes (Adam m, v) = ~12 bytes per parameter.
* The soft-attention vote **accumulator is int8** (1 byte), not int32/-16, and
  the score / weight registers are transient.
* No forward/backward intermediate scales like O(B·H·n²·d_h) (the dense
  attention tensor) — heads stay packed and the value combine reduces in place.

These tests assert those invariants concretely so the footprint can never
silently regress.
"""

from __future__ import annotations

import gc

import torch

import attention
from attention import signed_bundle, _I8_MIN, _I8_MAX
from bep import BepConfig, BepOptimizer
from model import TransformerConfig, BinaryTransformerLM


def _model(attn_mode="soft", **kw):
    base = dict(vocab_size=256, D=256, n_layers=2, n_heads=4, d_ff=512,
                attn_mode=attn_mode, seed=0)
    base.update(kw)
    return BinaryTransformerLM(TransformerConfig(**base))


# ── parameter footprint ──────────────────────────────────────────────────────

def test_parameters_are_int8_only():
    m = _model()
    total = 0
    for p in m.parameters():
        assert p.H.dtype == torch.int8
        total += p.H.nbytes
    assert total == m.param_bytes() == m.num_bit_parameters()


def test_no_float_buffers_anywhere():
    """Not a single float tensor is held by any parameter (no STE shadow weights,
    no float ``q``/``pm1``)."""
    m = _model()
    for p in m.parameters():
        for attr in ("H", "_bit"):
            t = getattr(p, attr, None)
            if isinstance(t, torch.Tensor):
                assert not t.is_floating_point(), (p.name, attr)
            elif t is not None and hasattr(t, "_packed_buf"):
                assert t._packed_buf.dtype == torch.int64


# ── training overhead: BEP holds no extra per-parameter state ────────────────

def test_optimizer_has_no_per_parameter_float_state():
    m = _model()
    opt = BepOptimizer(m.parameters(), BepConfig())
    before = m.param_bytes()
    ids = torch.randint(0, 256, (4, 8))
    tgt = torch.randint(0, 256, (4, 8))
    for _ in range(3):
        m.loss_and_backward(m.forward(ids), tgt)
        opt.step()
    # The only growth allowed is the (already-allocated) int8 H; nothing float.
    after = m.param_bytes()
    assert after == before, "param footprint changed during training"
    # The optimiser stores only references to BepParams, no momentum tensors.
    for p in opt.params:
        assert not hasattr(p, "m") and not hasattr(p, "v")
        assert not hasattr(p, "momentum")


def test_training_overhead_far_below_float_adam():
    """The binary model's *total* train-time parameter memory (int8 H) is a small
    fraction of an equivalent fp32 + Adam transformer (weight + 2 moments)."""
    m = _model()
    n = m.num_bit_parameters()
    binary_bytes = m.param_bytes()                  # int8 H
    float_adam_bytes = n * (4 + 4 + 4)              # fp32 weight + Adam m + v
    assert binary_bytes <= float_adam_bytes / 4, (binary_bytes, float_adam_bytes)


# ── int8 vote accumulator ────────────────────────────────────────────────────

def test_soft_attention_weights_are_int8():
    m = _model(attn_mode="soft")
    ids = torch.randint(0, 256, (3, 6))
    m.forward(ids)
    for blk in m.h:
        for w in blk.attn._cache["w_heads"]:
            assert w is not None and w.dtype == torch.int8


def test_accumulator_rails_are_int8():
    assert (_I8_MIN, _I8_MAX) == (-127, 127)


def test_signed_bundle_int8_saturation_is_sign_exact():
    """The int8 accumulator clamps to ±127 but clamping preserves sign, so the
    bundled output sign is exact even when the true weighted sum overflows int8."""
    import brute
    B, M, N, D = 1, 1, 200, 64
    # all values +1, all weights +1 → true sum = +200 (> 127) → saturates to +127
    W = torch.ones(B, M, N, dtype=torch.int8)
    Vb = brute.as_tensor(torch.ones(B, N, D, dtype=torch.bool), dtype=brute.bit1)
    out = signed_bundle(W, Vb, prefer_kernel=False)
    assert bool(out.bool().all()), "saturated positive sum must sign to +1 everywhere"
    # flip values to -1 → true sum = -200 → saturates to -127 → sign -1
    Vneg = brute.as_tensor(torch.zeros(B, N, D, dtype=torch.bool), dtype=brute.bit1)
    out_neg = signed_bundle(W, Vneg, prefer_kernel=False)
    assert not bool(out_neg.bool().any())


def test_signed_bundle_matches_dense_reference():
    """signed_bundle == sign(Σ_n W·pm1(V)) computed densely (small, in range)."""
    import brute
    g = torch.Generator().manual_seed(3)
    B, M, N, D = 2, 5, 7, 64
    W = torch.randint(-3, 4, (B, M, N), generator=g, dtype=torch.int8)
    Vb = brute.as_tensor(torch.randint(0, 2, (B, N, D), generator=g).bool(), dtype=brute.bit1)
    out = signed_bundle(W, Vb, prefer_kernel=False)
    v_pm1 = Vb.unpack_pm1().reshape(B, N, D)
    dense = torch.sign(torch.bmm(W.float(), v_pm1))
    dense = torch.where(dense == 0, torch.ones_like(dense), dense)   # sign(0)=+1
    assert torch.equal(out.unpack_pm1(), dense)


# ── no dense O(B·H·n²·d_h) intermediate ──────────────────────────────────────

def test_attention_does_not_materialise_dense_value_tensor():
    """A forward on a long sequence must not allocate an O(B·H·n²·d_h) tensor
    (the classic dense-attention blowup).  We bound the largest *new* float/int
    tensor created during the forward by tracking allocations."""
    big = []
    orig_empty = torch.empty
    orig_zeros = torch.zeros

    def track(fn):
        def wrapped(*a, **kw):
            t = fn(*a, **kw)
            try:
                big.append(t.numel())
            except Exception:
                pass
            return t
        return wrapped

    B, n, D, Hh = 2, 48, 256, 4
    cfg = TransformerConfig(vocab_size=64, D=D, n_layers=1, n_heads=Hh, d_ff=512,
                            attn_mode="soft", seed=0)
    m = BinaryTransformerLM(cfg)
    ids = torch.randint(0, 64, (B, n))
    torch.empty, torch.zeros = track(orig_empty), track(orig_zeros)
    try:
        m.forward(ids)
    finally:
        torch.empty, torch.zeros = orig_empty, orig_zeros
    dense_attn = B * Hh * n * n * (D // Hh)          # the forbidden blowup
    assert max(big) < dense_attn, (max(big), dense_attn)
