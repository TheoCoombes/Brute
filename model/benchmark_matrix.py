"""Benchmark matrix for HÆMMR research comparisons.

The matrix is intentionally small and reproducible: it measures binary/VSA
geometry, component behavior, synthetic attention tasks against a tiny
Transformer baseline, forward-path bitwise efficiency, and optionally ingests a
WikiText training CSV produced by ``train.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import brute
from layers import BSR, HopfieldBank, TokenCodebook
from model import HaemmrConfig, HaemmrLM
from probe_synthetic import run_haemmr, run_transformer
from profile_bitwise import count_hot_ops
from vsa import bind, hamming_similarity, position_codes, random_hypervectors, sign_to_bit1, to_pm1


def _metric(section, name, value, unit="", notes=""):
    return {"section": section, "name": name, "value": value, "unit": unit, "notes": notes}


def run_geometry(args):
    rows = []
    g = torch.Generator().manual_seed(args.seed + 10)
    D = 1024

    x = random_hypervectors(128, D, generator=g)
    r = random_hypervectors(128, D, generator=g)
    recovered = bind(bind(x, r), r)
    bind_acc = float((to_pm1(recovered) == to_pm1(x)).float().mean())
    rows.append(_metric("geometry", "bind_unbind_bit_accuracy", bind_acc, "fraction"))

    fillers = random_hypervectors(64 * 5, D, generator=g).reshape(64, 5, D)
    roles = random_hypervectors(64 * 5, D, generator=g).reshape(64, 5, D)
    correct = 0
    for i in range(64):
        records = to_pm1(bind(fillers[i].reshape(5, D), roles[i].reshape(5, D)))
        bundle = sign_to_bit1(records.sum(dim=0, keepdim=True))
        target = int(torch.randint(0, 5, (1,), generator=g).item())
        query = bind(bundle, roles[i, target:target + 1])
        pred = int(hamming_similarity(query, fillers[i]).argmax().item())
        correct += int(pred == target)
    rows.append(_metric("geometry", "bundle_5_record_retrieval_accuracy", correct / 64, "fraction"))

    base = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()
    pos = position_codes(base, 16)
    dressed = bind(x[:16], pos)
    undressed = bind(dressed, pos)
    pos_acc = float((to_pm1(undressed) == to_pm1(x[:16])).float().mean())
    rows.append(_metric("geometry", "position_bind_unbind_bit_accuracy", pos_acc, "fraction"))

    cb = TokenCodebook(256, D, name="bench.E", generator=g)
    token_rows = cb.E.bit[:128]
    pos128 = position_codes(base, 128)
    positioned = bind(token_rows, pos128)
    raw_decode = cb.decode(positioned).argmax(dim=1)
    unbound_decode = cb.decode(bind(positioned, pos128)).argmax(dim=1)
    target = torch.arange(128)
    rows.append(_metric("geometry", "positioned_raw_decode_accuracy",
                        float((raw_decode == target).float().mean()), "fraction",
                        notes="Should be low: positioned concepts are not raw codebook rows."))
    rows.append(_metric("geometry", "positioned_unbound_decode_accuracy",
                        float((unbound_decode == target).float().mean()), "fraction"))

    for dim in (512, 4096):
        hv = random_hypervectors(128, dim, generator=g)
        sim = hamming_similarity(hv[:1], hv[1:]).float().abs()
        rows.append(_metric("geometry", f"mean_abs_random_similarity_D{dim}",
                            float((sim / dim).mean()), "normalized_dot"))
    return rows


def run_components(args):
    rows = []
    g = torch.Generator().manual_seed(args.seed + 20)
    D = 128

    bsr = BSR(D, name="bench.bsr", generator=g)
    c = random_hypervectors(2 * 64, D, generator=g).reshape(2, 64, D)
    batched = bsr.forward(c)
    bsr.reset_stream(2)
    streamed_pm1 = torch.stack([to_pm1(bsr.step(c[:, i])) for i in range(64)], dim=1)
    mismatch = float((streamed_pm1 != to_pm1(batched)).float().mean())
    rows.append(_metric("component_bsr", "streaming_vs_batched_mismatch", mismatch, "fraction"))

    hv = random_hypervectors(4, D, generator=g)
    seq_a = torch.stack([to_pm1(hv[0])] * 63 + [to_pm1(hv[3])])
    seq_b = torch.stack([to_pm1(hv[1])] * 63 + [to_pm1(hv[3])])
    late = to_pm1(bsr.forward(brute.as_tensor(torch.stack([seq_a, seq_b]) > 0, dtype=brute.bit1)))[:, -1]
    late_diff = float((late[0] != late[1]).float().mean())
    rows.append(_metric("component_bsr", "persistent_prefix_late_read_difference", late_diff, "fraction"))

    hop = HopfieldBank(D, n_slots=16, top_k=1, name="bench.hop", generator=g)
    keys = random_hypervectors(16, D, generator=g)
    payloads = random_hypervectors(16, D, generator=g)
    hop.P.bit = keys
    hop.U.bit = payloads
    out = hop.forward(keys)
    hop_acc = float((to_pm1(out) == to_pm1(payloads)).float().mean())
    rows.append(_metric("component_hopfield", "controlled_top1_payload_bit_accuracy", hop_acc, "fraction"))

    cfg = HaemmrConfig(vocab_size=128, D=256, n_layers=1, d_ff=512, n_slots=64,
                       top_k=5, use_position=True, seed=args.seed)
    model = HaemmrLM(cfg, device="cpu")
    rows.append(_metric("component_model", "bit_parameters", model.num_bit_parameters(), "bits"))
    rows.append(_metric("component_model", "approx_parameter_storage", model.num_bit_parameters() / 8 / 1e6, "MB"))
    return rows


def run_synthetic(args):
    rows = []
    synth_args = SimpleNamespace(
        seed=args.seed, D=args.synthetic_D, layers=1, d_ff=2 * args.synthetic_D,
        slots=32, top_k=3, epi_slots=None, epi_read_k=1, decay_shifts=(1, 2, 3, 4, 0),
        device="cpu", gate_open=0.05, r=0.1, bits=15, flip_dropout=0.0,
        codebook_mode="structured", bef_sweeps=30, sem_weight=0.5,
        use_position=True, use_bsr=True, use_episodic=True, use_hopfield=True,
        margin_supervision=True, steps=args.haemmr_steps,
        batch_size=64, baseline_steps=args.transformer_steps, baseline_dim=64,
        baseline_layers=2, baseline_lr=3e-3,
    )
    for task in ("induction", "marker", "copy"):
        for length in (16, 64):
            h = run_haemmr(task, length, synth_args)
            rows.append({
                "section": "synthetic_attention",
                "task": task,
                "length": length,
                "model": "haemmr",
                **h,
            })
            t = run_transformer(task, length, synth_args)
            rows.append({
                "section": "synthetic_attention",
                "task": task,
                "length": length,
                "model": "tiny_transformer",
                **t,
            })
    return rows


def run_efficiency(args):
    cfg = HaemmrConfig(vocab_size=512, D=256, n_layers=1, d_ff=512, n_slots=64,
                       top_k=5, use_position=True, seed=args.seed)
    model = HaemmrLM(cfg, device="cpu")
    ids = torch.randint(0, 512, (4, 32))
    model.forward(ids)
    with count_hot_ops() as counts:
        t0 = time.perf_counter()
        for _ in range(args.profile_iters):
            model.forward(ids)
        elapsed = time.perf_counter() - t0
    tokens = 4 * 32 * args.profile_iters
    rows = [
        _metric("efficiency", "forward_tokens_per_second", tokens / max(elapsed, 1e-9), "tokens/s"),
        _metric("efficiency", "fast_matmul_calls", counts["fast_matmul"], "calls"),
        _metric("efficiency", "unpack_pm1_calls", counts["unpack_pm1"], "calls"),
        _metric("efficiency", "to_bit1_pack_calls", counts["to_bit1_pack"], "calls"),
    ]
    return rows


def run_wikitext(csv_path: str | None):
    if not csv_path:
        return []
    path = Path(csv_path)
    if not path.exists():
        return [_metric("wikitext", "csv_missing", str(path))]
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        step = int(r["step"])
        out.extend([
            _metric("wikitext", f"step_{step}_val_loss", float(r["val_loss"])),
            _metric("wikitext", f"step_{step}_val_ppl", float(r["val_ppl"])),
            _metric("wikitext", f"step_{step}_val_acc", float(r["val_acc"]), "fraction"),
        ])
    if rows:
        best = min(rows, key=lambda r: float(r["val_loss"]))
        out.append(_metric("wikitext", "best_val_loss", float(best["val_loss"])))
        out.append(_metric("wikitext", "best_val_ppl", float(best["val_ppl"])))
        out.append(_metric("wikitext", "best_step", int(best["step"])))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic-D", type=int, default=128)
    p.add_argument("--haemmr-steps", type=int, default=80)
    p.add_argument("--transformer-steps", type=int, default=300)
    p.add_argument("--profile-iters", type=int, default=5)
    p.add_argument("--wikitext-csv", default=None)
    p.add_argument("--json-out", default="/private/tmp/haemmr_benchmark_matrix.json")
    args = p.parse_args()

    t0 = time.time()
    results = {
        "meta": {
            "seed": args.seed,
            "haemmr_steps": args.haemmr_steps,
            "transformer_steps": args.transformer_steps,
            "synthetic_D": args.synthetic_D,
        },
        "rows": (
            run_geometry(args)
            + run_components(args)
            + run_synthetic(args)
            + run_efficiency(args)
            + run_wikitext(args.wikitext_csv)
        ),
    }
    results["meta"]["elapsed_s"] = time.time() - t0
    Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
