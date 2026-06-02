"""Lightweight bitwise efficiency audit for HÆMMR forward passes."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import brute
import brute.fast as bfast
from brute.tensor import Tensor as BruteTensor

from model import HaemmrConfig, HaemmrLM


@contextlib.contextmanager
def count_hot_ops():
    counts = {"fast_matmul": 0, "unpack_pm1": 0, "to_bit1_pack": 0}
    orig_matmul = bfast.matmul
    orig_unpack = BruteTensor.unpack_pm1
    orig_as_tensor = brute.as_tensor

    def matmul_wrap(*args, **kwargs):
        counts["fast_matmul"] += 1
        return orig_matmul(*args, **kwargs)

    def unpack_wrap(self, *args, **kwargs):
        counts["unpack_pm1"] += 1
        return orig_unpack(self, *args, **kwargs)

    def as_tensor_wrap(data, *, dtype=None, device=None):
        if dtype is brute.bit1:
            counts["to_bit1_pack"] += 1
        return orig_as_tensor(data, dtype=dtype, device=device)

    bfast.matmul = matmul_wrap
    brute.fast.matmul = matmul_wrap
    BruteTensor.unpack_pm1 = unpack_wrap
    brute.as_tensor = as_tensor_wrap
    try:
        yield counts
    finally:
        bfast.matmul = orig_matmul
        brute.fast.matmul = orig_matmul
        BruteTensor.unpack_pm1 = orig_unpack
        brute.as_tensor = orig_as_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--slots", type=int, default=128)
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-position", dest="use_position", action="store_false", default=True)
    args = p.parse_args()

    cfg = HaemmrConfig(vocab_size=args.vocab_size, D=args.D, n_layers=args.layers,
                       d_ff=args.d_ff, n_slots=args.slots, top_k=args.top_k,
                       use_position=args.use_position, seed=0)
    model = HaemmrLM(cfg, device=args.device)
    ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=args.device)
    model.forward(ids)

    with count_hot_ops() as counts:
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model.forward(ids)
        dt = time.perf_counter() - t0

    n_tok = args.batch_size * args.seq_len * args.iters
    print(f"elapsed_s={dt:.4f}")
    print(f"tokens={n_tok}")
    print(f"tokens_per_s={n_tok / max(dt, 1e-9):.1f}")
    for k, v in counts.items():
        print(f"{k}={v}")
    print("known_unpack_hotspots=Boolean backward caches, BSR vote state, Hopfield majority")
    print("next_fusion_target=packed association-to-int accumulator / packed majority")


if __name__ == "__main__":
    main()

