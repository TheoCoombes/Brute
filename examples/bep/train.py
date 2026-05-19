"""Train a fully binary BEP MLP.

Targets (Sec. 4.2 of the BEP paper):

* Random Prototypes, L=2: BEP reaches ≈76–78% test accuracy.
* FashionMNIST, L=2     : BEP reaches ≈86–88% test accuracy.

Quick run (a few minutes on a laptop CPU):

    python train.py --dataset prototypes --epochs 30

FashionMNIST (downloads the dataset on first run):

    python train.py --dataset fashion_mnist --epochs 40 --hidden 256 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a script without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import brute  # noqa: F401 — ensures brute is importable in the env

from bef import generate_bef
from bep import BEPConfig, BEPModel
from data import fashion_mnist, random_prototypes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["prototypes", "fashion_mnist"], default="prototypes")
    p.add_argument("--hidden", type=int, nargs="*", default=[256, 256],
                   help="Hidden layer widths (defines K_1..K_L).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--r", type=float, default=0.5, help="Update margin (Eq. 1).")
    p.add_argument("--nu", type=float, default=0.05, help="Backward gating threshold (Eq. 5).")
    p.add_argument("--group-size", type=int, default=4,
                   help="Initial neuron group size (winner-takes-update).")
    p.add_argument("--p-reinforce", type=float, default=0.5)
    p.add_argument("--weight-clip", type=int, default=2048)
    p.add_argument("--bef-iters", type=int, default=None,
                   help="Coordinate-flip iterations for the BEF classifier.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()

    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    torch.manual_seed(args.seed)

    # ── Data ────────────────────────────────────────────────────────────────
    if args.dataset == "prototypes":
        x_tr, y_tr, x_te, y_te = random_prototypes(seed=args.seed)
    else:
        x_tr, y_tr, x_te, y_te = fashion_mnist(seed=args.seed)
    x_tr, x_te = x_tr.to(device), x_te.to(device)
    y_tr, y_te = y_tr.to(device), y_te.to(device)
    K0 = int(x_tr.shape[1])
    n_classes = int(y_tr.max().item() + 1)
    print(f"dataset={args.dataset}  N_train={x_tr.shape[0]}  N_test={x_te.shape[0]}  "
          f"K0={K0}  C={n_classes}")

    # ── Model ───────────────────────────────────────────────────────────────
    K_L = int(args.hidden[-1])
    P = generate_bef(n_classes, K_L, iters=args.bef_iters, seed=args.seed, device=device)
    cfg = BEPConfig(
        r=args.r,
        nu=args.nu,
        group_size_init=args.group_size,
        p_reinforce=args.p_reinforce,
        weight_clip=args.weight_clip,
    )
    model = BEPModel([K0] + list(args.hidden), P, config=cfg, device=device, seed=args.seed)
    print(model)

    # Initial accuracy
    init_te = model.accuracy(x_te, y_te)
    print(f"epoch  0 / {args.epochs}   test_acc = {init_te:.4f}")

    best_te = init_te
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(x_tr.shape[0], generator=g).to(device)
        x_sh = x_tr[perm]
        y_sh = y_tr[perm]

        n_correct = 0
        n_trig = 0
        n_seen = 0
        n_batches = (x_sh.shape[0] + args.batch_size - 1) // args.batch_size
        for b_idx in range(n_batches):
            s = b_idx * args.batch_size
            e = min(s + args.batch_size, x_sh.shape[0])
            xb = x_sh[s:e]
            yb = y_sh[s:e]
            state = model.forward(xb)
            info = model.step(state, yb)
            n_correct += info["n_correct"]
            n_trig += info["n_triggered"]
            n_seen += info["batch_size"]

        tr_acc = n_correct / max(n_seen, 1)
        te_acc = model.accuracy(x_te, y_te)
        best_te = max(best_te, te_acc)
        dt = time.time() - t0
        print(
            f"epoch {ep:2d} / {args.epochs}   "
            f"train_acc = {tr_acc:.4f}   test_acc = {te_acc:.4f}   "
            f"triggered = {n_trig / max(n_seen,1):.3f}   "
            f"({dt:.1f}s)"
        )

    print(f"\nbest test accuracy: {best_te:.4f}")


if __name__ == "__main__":
    main()
