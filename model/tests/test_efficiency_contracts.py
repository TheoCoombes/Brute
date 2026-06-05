"""Hot-path efficiency contracts — proves the binary transformer stays packed.

The defining requirement of the rewrite: every *persistent* tensor stays packed
``brute.bit1`` (uint64) end-to-end, and only *transient register* integers
(scores, vote accumulators) are allowed.  The disease being cured is the old
mixers' constant unpacking (``pm1_int``, ``bmm``/``einsum`` on dense ±1).

We assert, by instrumenting the two C++ decoders ``unpack_bits`` (→ dense ±1
float) and ``unpack_bool`` (→ bool bytes; the substrate of every ``.bool()`` /
``pm1_int``):

* **Forward (hardmax)** performs **zero** unpacks of any kind — the sequence-mixing
  hot path is fully packed.
* A **full training step** performs **zero** ``unpack_bits`` — no dense ±1 float is
  ever materialised, forward *or* backward.  ``unpack_bool`` is permitted only at
  BEP weight-gradient / codebook granularity (never per token-pair), so its count
  is bounded far below the O(B·n²) sequence core.
* The parameter footprint is exactly the int8 hidden weights ``H`` (≈ 1 byte per
  1-bit parameter) — no float copies anywhere.
"""

from __future__ import annotations

import contextlib

import torch

import brute
import bep
from bep import BepConfig, BepOptimizer
from model import TransformerConfig, BinaryTransformerLM, IGNORE_INDEX


# ── instrumentation: wrap the C++ unpack ops on the brute op namespace ──────────

@contextlib.contextmanager
def _unpack_guard(raise_on=(), count=True):
    """Context manager that wraps ``unpack_bits`` / ``unpack_bool``.

    Ops named in ``raise_on`` raise immediately when called (proving they are
    never hit); the rest are counted.  Yields the live counts dict.
    """
    names = ("unpack_bits", "unpack_bool")
    orig = {k: getattr(torch.ops.brute, k) for k in names}
    counts = {k: 0 for k in names}

    def make(name, fn):
        def wrapped(*a, **kw):
            counts[name] += 1
            if name in raise_on:
                raise AssertionError(f"hot path unpacked via {name}()")
            return fn(*a, **kw)
        return wrapped

    for k in names:
        setattr(torch.ops.brute, k, make(k, orig[k]))
    try:
        yield counts
    finally:
        for k, v in orig.items():
            setattr(torch.ops.brute, k, v)


def _model(attn_mode="hardmax", value_proj=True, n_layers=2):
    cfg = TransformerConfig(vocab_size=48, D=128, n_layers=n_layers, n_heads=2,
                            d_ff=256, attn_mode=attn_mode, value_proj=value_proj,
                            seed=0)
    return BinaryTransformerLM(cfg)


# ── forward hot path: zero unpacks ──────────────────────────────────────────────

def test_forward_hardmax_never_unpacks():
    m = _model(attn_mode="hardmax")
    ids = torch.randint(0, 48, (4, 8))
    with _unpack_guard(raise_on=("unpack_bits", "unpack_bool")):
        m.forward(ids)               # must complete with zero unpacks


def test_forward_hardmax_value_proj_false_never_unpacks():
    m = _model(attn_mode="hardmax", value_proj=False)
    ids = torch.randint(0, 48, (4, 8))
    with _unpack_guard(raise_on=("unpack_bits", "unpack_bool")):
        m.forward(ids)


# ── full training step: zero dense-±1 (unpack_bits) materialisation ─────────────

def test_training_step_never_materialises_dense_pm1():
    m = _model(attn_mode="hardmax")
    opt = BepOptimizer(m.parameters(), BepConfig())
    ids = torch.randint(0, 48, (4, 8))
    tgt = torch.randint(0, 48, (4, 8))
    with _unpack_guard(raise_on=("unpack_bits",)) as counts:
        logits = m.forward(ids)
        m.loss_and_backward(logits, tgt)
        opt.step()
    assert counts["unpack_bits"] == 0


def test_backward_unpack_bool_is_bounded():
    """unpack_bool (the .bool()/pm1_int substrate) only appears at weight /
    active-row granularity in backward — never per token-pair (O(B·n²))."""
    B, n = 4, 8
    m = _model(attn_mode="hardmax", n_layers=2)
    opt = BepOptimizer(m.parameters(), BepConfig())
    ids = torch.randint(0, 48, (B, n))
    tgt = torch.randint(0, 48, (B, n))
    with _unpack_guard() as counts:
        m.loss_and_backward(m.forward(ids), tgt)
        opt.step()
    # The sequence core is O(B·n²·H) = 4·64·2 = 512 token-pairs; a per-pair unpack
    # would dwarf this. Bound well under it.
    assert counts["unpack_bool"] < B * n * n, counts


# ── parameter footprint: int8 H only ────────────────────────────────────────────

def test_param_bytes_is_int8_hidden_weight():
    m = _model()
    n_bits = m.num_bit_parameters()
    pbytes = m.param_bytes()
    assert pbytes == n_bits, f"{pbytes} != {n_bits}"


def test_all_params_are_int8_no_floats():
    m = _model()
    for p in m.parameters():
        assert p.H.dtype == torch.int8, (p.name, p.H.dtype)
        assert not p.H.is_floating_point()


def test_visible_weight_is_one_bit():
    m = _model()
    for p in m.parameters():
        assert p.bit.dtype == brute.bit1
        # packed footprint ≈ 1 bit/param (uint64 words)
        assert p.bit._packed_buf.dtype == torch.int64


def test_soft_forward_unpack_only_when_no_kernel():
    """Soft attention's Stage-A reference unpacks the value transiently (±1) once
    per head/layer; the hardmax path never does.  This documents (and bounds)
    the single Stage-A transient unpack the kernel removes."""
    from attention import _kernel_available
    m = _model(attn_mode="soft", n_layers=1)
    ids = torch.randint(0, 48, (4, 8))
    with _unpack_guard() as counts:
        m.forward(ids)
    if _kernel_available():
        assert counts["unpack_bits"] == 0          # kernel: zero unpacks
    else:
        # reference: one ±1 unpack per head (n_heads) — small, bounded, transient.
        assert counts["unpack_bits"] <= m.cfg.n_heads * m.cfg.n_layers + 1
