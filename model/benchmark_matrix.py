"""Research comparison matrix for the native binary transformer.

A small, reproducible matrix that reports, side by side, the things that make
this model interesting as a research artefact:

  1. **geometry**   — the exact bit1/VSA identities the model is built on
                       (XNOR = bipolar multiply; ``⟨u,v⟩ = D − 2·Hamming``).
  2. **mechanism**  — attention exactness: integer scores equal ``d_h − 2H``, and
                       hardmax transports the argmax value bit-for-bit.
  3. **learning**   — previous-token copy (a genuinely non-local attention task):
                       the binary transformer vs a float transformer baseline.
  4. **efficiency** — the binary model's forward does zero unpacks, and its
                       train-time parameter footprint vs an fp32 + Adam transformer.

Run as a script (not ``-m model.benchmark_matrix`` — the ``model`` directory would
shadow the ``model.py`` module)::

    .venv/bin/python model/benchmark_matrix.py              # full matrix
    .venv/bin/python model/benchmark_matrix.py --quick      # skip the learning probe
    .venv/bin/python model/benchmark_matrix.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

import torch

import brute
from attention import BinaryMultiHeadAttention
from model import TransformerConfig, BinaryTransformerLM
from vsa import bind, hamming_similarity, random_hypervectors
from _helpers import (TransformerBaseline, prev_token_batch, retrieval_config,
                      train_prev_token, train_baseline)


def _row(section, name, value, unit="", notes=""):
    return {"section": section, "name": name, "value": value, "unit": unit, "notes": notes}


# ── 1. geometry ──────────────────────────────────────────────────────────────

def run_geometry(g):
    rows, D = [], 1024
    a = random_hypervectors(64, D, generator=g)
    b = random_hypervectors(64, D, generator=g)
    xnor_ok = bool((bind(a, b).unpack_pm1() == a.unpack_pm1() * b.unpack_pm1()).all())
    rows.append(_row("geometry", "XNOR == bipolar multiply", xnor_ok, "", "exact"))
    sim = hamming_similarity(a[:1], b)                       # (1, 64) int
    pm = (a[:1].unpack_pm1() @ b.unpack_pm1().t())
    rows.append(_row("geometry", "<u,v> == D - 2*Hamming",
                     bool((sim.float() == pm).all()), "", "exact"))
    return rows


# ── 2. mechanism exactness ────────────────────────────────────────────────────

def run_mechanism(g):
    rows = []
    D, H, B, n = 128, 2, 2, 8
    x = random_hypervectors(B * n, D, generator=g).reshape(B, n, D)
    mha = BinaryMultiHeadAttention(D, H, name="m", attn_mode="hardmax",
                                   causal_strict=True, generator=g)
    a = mha.forward(x)
    qh, kh = mha._cache["q_head_bits"][0], mha._cache["k_head_bits"][0]
    score = brute.fast.matmul(qh[0], kh[0])
    dot = (qh[0].unpack_pm1() @ kh[0].unpack_pm1().t())
    rows.append(_row("mechanism", "score == d_h - 2H",
                     bool((score.float() == dot).all()), "", "exact"))
    rows.append(_row("mechanism", "forward stays packed bit1", a.dtype == brute.bit1, "", ""))
    return rows


# ── 3. learning: binary vs float baseline on previous-token copy ──────────────

def run_learning(g, steps=600):
    rows = []
    cfg = retrieval_config(V=16, D=128, n_heads=1, attn_mode="hardmax")
    binary = BinaryTransformerLM(cfg)
    t0 = time.time(); b_acc = train_prev_token(binary, steps=steps); b_t = time.time() - t0
    base = TransformerBaseline(16, d_model=64, n_heads=2, n_layers=1,
                               causal_strict=True, alibi_recency=True)
    t0 = time.time(); f_acc = train_baseline(base, prev_token_batch, steps=max(steps // 2, 200))
    f_t = time.time() - t0
    rows.append(_row("learning", "binary prev-token acc", round(b_acc, 3), "",
                     f"{b_t:.0f}s, chance 0.0625"))
    rows.append(_row("learning", "float baseline prev-token acc", round(f_acc, 3), "", f"{f_t:.0f}s"))
    rows.append(_row("learning", "binary matches baseline (>0.9)",
                     bool(b_acc > 0.9 and f_acc > 0.9), "", ""))
    return rows


# ── 4. efficiency ──────────────────────────────────────────────────────────────

def run_efficiency(g):
    rows = []
    cfg = TransformerConfig(vocab_size=256, D=256, n_layers=2, n_heads=4, d_ff=512,
                            attn_mode="hardmax", seed=0)
    m = BinaryTransformerLM(cfg)
    ids = torch.randint(0, 256, (4, 16))
    names = ("unpack_bits", "unpack_bool")
    orig = {k: getattr(torch.ops.brute, k) for k in names}
    counts = {k: 0 for k in names}
    for k in names:
        def mk(name, o):
            return lambda *a, **kw: (counts.__setitem__(name, counts[name] + 1) or o(*a, **kw))
        setattr(torch.ops.brute, k, mk(k, orig[k]))
    try:
        m.forward(ids)
    finally:
        for k, v in orig.items():
            setattr(torch.ops.brute, k, v)
    rows.append(_row("efficiency", "forward unpacks (hardmax)", sum(counts.values()), "",
                     "fully packed"))
    n = m.num_bit_parameters()
    rows.append(_row("efficiency", "param footprint", round(m.param_bytes() / 1e6, 3), "MB",
                     f"{n:,} bit-params, int16 H"))
    rows.append(_row("efficiency", "vs fp32+Adam footprint", round((n * 12) / 1e6, 3), "MB",
                     "weight+m+v; ~6x larger"))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=600, help="prev-token probe steps")
    p.add_argument("--quick", action="store_true", help="skip the learning probe")
    p.add_argument("--json", default=None, help="write the matrix as JSON")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    rows = []
    rows += run_geometry(g)
    rows += run_mechanism(g)
    rows += run_efficiency(g)
    if not args.quick:
        rows += run_learning(g, steps=args.steps)

    width = max(len(r["name"]) for r in rows)
    section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            print(f"\n-- {section} --")
        val = r["value"]
        val = ("OK" if val else "FAIL") if isinstance(val, bool) else val
        line = f"  {r['name']:<{width}}  {val} {r['unit']}".rstrip()
        if r["notes"]:
            line += f"   ({r['notes']})"
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
