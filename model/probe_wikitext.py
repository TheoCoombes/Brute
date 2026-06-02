"""Run small WikiText probes for position/no-position HÆMMR settings.

This script wraps train.py so probes use the same training code as normal runs.
By default it runs two short probes:
  1. no position binding
  2. position binding with next-position unbind decode

Use --final to run a longer final model with the better probe setting.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def last_val_loss(csv_path: Path) -> float:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("inf")
    return float(rows[-1]["val_loss"])


def train_cmd(args, name: str, *, positioned: bool, steps: int, D: int, layers: int):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    ckpt_path = out_dir / f"{name}.pt"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--dataset", args.dataset,
        "--data-root", args.data_root,
        "--vocab-cap", str(args.vocab_cap),
        "--max-train-tokens", str(args.max_train_tokens),
        "--D", str(D),
        "--layers", str(layers),
        "--d-ff", str(args.d_ff),
        "--slots", str(args.slots),
        "--top-k", str(args.top_k),
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--steps", str(steps),
        "--eta", str(args.eta),
        "--threshold", str(args.threshold),
        "--eta-end", str(args.eta_end),
        "--eval-every", str(args.eval_every),
        "--sample-every", str(args.sample_every),
        "--sample-len", str(args.sample_len),
        "--temperature", str(args.temperature),
        "--sample-top-k", str(args.sample_top_k),
        "--log-csv", str(csv_path),
        "--ckpt", str(ckpt_path),
        "--device", args.device,
    ]
    if not positioned:
        cmd.append("--no-position")
    else:
        cmd.extend(["--position-decode", "next_unbind"])
    return cmd, csv_path, ckpt_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    p.add_argument("--data-root", default="./.data")
    p.add_argument("--vocab-cap", type=int, default=1024)
    p.add_argument("--max-train-tokens", type=int, default=500000)
    p.add_argument("--probe-steps", type=int, default=350)
    p.add_argument("--final", action="store_true")
    p.add_argument("--final-steps", type=int, default=1500)
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--final-D", type=int, default=768)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--final-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--slots", type=int, default=128)
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eta", type=float, default=3.0)
    p.add_argument("--eta-end", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=250)
    p.add_argument("--sample-len", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--sample-top-k", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    probes = []
    for name, positioned in [("probe_no_position", False), ("probe_position_next_unbind", True)]:
        cmd, csv_path, ckpt_path = train_cmd(
            args, name, positioned=positioned, steps=args.probe_steps,
            D=args.D, layers=args.layers,
        )
        run(cmd)
        probes.append((name, positioned, last_val_loss(csv_path), ckpt_path))

    probes.sort(key=lambda x: x[2])
    best_name, best_positioned, best_loss, _ = probes[0]
    print(f"best_probe={best_name} positioned={best_positioned} val_loss={best_loss:.4f}")

    if args.final:
        cmd, csv_path, ckpt_path = train_cmd(
            args, "final_tiny", positioned=best_positioned, steps=args.final_steps,
            D=args.final_D, layers=args.final_layers,
        )
        run(cmd)
        print(f"final_csv={csv_path}")
        print(f"final_ckpt={ckpt_path}")


if __name__ == "__main__":
    main()

