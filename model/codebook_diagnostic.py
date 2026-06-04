"""Compare GPT-2 SimHash and balanced-hash binary codebook geometry.

Reports the checks requested in the training-stability plan:

* sampled pairwise Hamming-distance histogram
* effective rank of the V x D +/-1 code matrix
* exact collisions and sampled near-collisions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from data import wikitext
from vsa import balanced_hash_frame_bool


def _effective_rank(bits: torch.Tensor) -> float:
    x = bits.to(torch.float32).mul_(2).sub_(1)
    gram = x.t().matmul(x)
    eig = torch.linalg.eigvalsh(gram).clamp_min_(0)
    denom = eig.square().sum()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((eig.sum().square() / denom).item())


def _sample_hamming(bits: torch.Tensor, *, n_pairs: int, seed: int) -> torch.Tensor:
    V = bits.shape[0]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randint(0, V, (n_pairs,), generator=g)
    b = torch.randint(0, V, (n_pairs,), generator=g)
    same = a == b
    if bool(same.any()):
        b[same] = (b[same] + 1) % V
    return (bits[a] != bits[b]).sum(dim=1)


def _stats(name: str, bits: torch.Tensor, *, n_pairs: int, seed: int,
           near_frac: float) -> dict:
    bits = bits.cpu().bool()
    D = bits.shape[1]
    dist = _sample_hamming(bits, n_pairs=n_pairs, seed=seed)
    hist = torch.histc(dist.to(torch.float32), bins=16, min=0, max=D)
    unique = torch.unique(bits, dim=0).shape[0]
    return {
        "name": name,
        "vocab": int(bits.shape[0]),
        "D": int(D),
        "effective_rank": _effective_rank(bits),
        "exact_collisions": int(bits.shape[0] - unique),
        "sampled_near_collisions": int((dist < near_frac * D).sum().item()),
        "sampled_pairs": int(n_pairs),
        "hamming": {
            "min": int(dist.min().item()),
            "p05": float(torch.quantile(dist.to(torch.float32), 0.05).item()),
            "mean": float(dist.to(torch.float32).mean().item()),
            "p95": float(torch.quantile(dist.to(torch.float32), 0.95).item()),
            "max": int(dist.max().item()),
            "hist16": [int(x) for x in hist.to(torch.int64).tolist()],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    p.add_argument("--data-root", default="./.data")
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", type=int, default=200_000)
    p.add_argument("--near-frac", type=float, default=0.10)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    corpus = wikitext(name=args.dataset, data_root=args.data_root)

    from codebook import from_corpus

    simhash_pm1 = from_corpus(corpus, args.D, seed=args.seed,
                              cache_dir=str(Path(args.data_root) / "codebook"))
    simhash_bits = simhash_pm1 > 0
    balanced_bits = balanced_hash_frame_bool(corpus.vocab_size, args.D, seed=args.seed)

    rows = [
        _stats("gpt2_simhash", simhash_bits, n_pairs=args.pairs,
               seed=args.seed, near_frac=args.near_frac),
        _stats("balanced_hash", balanced_bits, n_pairs=args.pairs,
               seed=args.seed, near_frac=args.near_frac),
    ]
    text = json.dumps(rows, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text)


if __name__ == "__main__":
    main()
