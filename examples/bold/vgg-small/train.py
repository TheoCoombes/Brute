"""Train VGG-SMALL on CIFAR-10 using BOLD (Boolean Logic Deep Learning).

Reference
---------
Nguyen et al., "BOLD: Boolean Logic Deep Learning", NeurIPS 2024.

Target accuracies (Table 2 of the paper, modified VGG-SMALL with 1 FC head
— Table 9 footnote 2):

  * BOLD w/ BN       : 92.37 ± 0.01 % (5 reps)
  * BOLD w/o BN      : 90.29 ± 0.09 % (6 reps)
  * FP baseline      : 93.80 %

Training recipe (Appendix D.1.1 + D.1.2):
  * Optimisers
      - Real-valued params (1st conv, BN, final FC): Adam, lr ``1e-3``.
      - Boolean params      : BooleanOptimizer, lr ``150`` (with BN)
                              or ``12`` (without BN).
  * LR schedule        : cosine to 0 over the full 300 epochs (both optimisers).
  * Data augmentation  : random crop (padding 4) + random horizontal flip.
                         RandAugment + Mixup are enabled by ``--strong-aug``
                         (recommended for the final ~3 pt of accuracy per
                         Appendix D.1.3 ablation).
  * Batch size         : 128 by default; the paper uses 300 (override with
                         ``--batch-size 300`` if you have the VRAM).

Usage
-----
Full reproduction (matches paper):

    python train.py --use-bn --epochs 300 --batch-size 300 --strong-aug \\
        --data-root ./data

Quick smoke test (~3 minutes on a laptop):

    python train.py --use-bn --epochs 2 --batch-size 128 --max-train-batches 50
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import brute.optim as bopt
from vgg_small import VGGSmall, count_parameters


# ── Data ─────────────────────────────────────────────────────────────────────

# CIFAR-10 channel statistics (computed on the train split, widely used).
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)


def build_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    strong_aug: bool = False,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    base_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if strong_aug:
        # RandAugment + ColorJitter approximates the paper's RandAugment +
        # lighting recipe. Mixup is applied in the training loop on tensors.
        base_train += [
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    train_tf = transforms.Compose(base_train + [
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(256, batch_size), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


# ── Mixup (Zhang et al. 2018, "mixup: Beyond Empirical Risk Minimization") ──

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Returns (x_mixed, y_a, y_b, lam)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.shape[0], device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[idx]
    return x_mixed, y, y[idx], lam


def mixup_ce(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    return lam * F.cross_entropy(logits, y_a) + (1.0 - lam) * F.cross_entropy(logits, y_b)


# ── Train / eval loops ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    n, n_correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n_correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, n_correct / n


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    bool_opt: bopt.BooleanOptimizer,
    real_opt: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    use_mixup: bool,
    log_interval: int = 50,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    model.train()
    loss_sum, n_correct, n_seen, flips = 0.0, 0, 0, 0
    t0 = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_mixup:
            x_in, y_a, y_b, lam = mixup(x, y, alpha=0.2)
            logits = model(x_in)
            loss = mixup_ce(logits, y_a, y_b, lam)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        bool_opt.zero_grad(set_to_none=True)
        real_opt.zero_grad(set_to_none=True)
        loss.backward()
        bool_opt.step()
        real_opt.step()

        flips += bool_opt.nb_flips
        loss_sum += loss.item() * y.numel()
        # Use UN-mixed labels for the running accuracy (just a coarse signal).
        n_correct += (logits.argmax(dim=1) == y).sum().item()
        n_seen += y.numel()

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  ep {epoch}  it {batch_idx+1:>4d}/{len(loader)}  "
                f"loss {loss.item():.4f}  "
                f"acc {n_correct / n_seen:6.3f}  "
                f"flips/it {flips // (batch_idx+1):>6d}  "
                f"t/it {(time.time() - t0) / (batch_idx+1) * 1000:5.1f}ms",
                flush=True,
            )
    return loss_sum / max(n_seen, 1), n_correct / max(n_seen, 1), flips


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-bn", action="store_true",
                   help="Include BatchNorm (paper target: 92.37%%). "
                        "Without this flag we target the no-BN result: 90.29%%.")
    p.add_argument("--lr-real", type=float, default=1e-3,
                   help="Adam learning rate for full-precision params.")
    p.add_argument("--lr-bool", type=float, default=None,
                   help="BooleanOptimizer learning rate. Defaults to the "
                        "BOLD paper values: 150 with BN, 12 without.")
    p.add_argument("--strong-aug", action="store_true",
                   help="Enable RandAugment + ColorJitter + Mixup. Adds the "
                        "final ~3 pts of accuracy per Appendix D.1.3.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="cpu / cuda / mps. Auto-detected if omitted.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional checkpoint path. Saves best test accuracy.")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Cap train iterations per epoch (smoke-test mode).")
    p.add_argument("--log-interval", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    # Paper defaults for the Boolean LR (Appendix D.1.1).
    if args.lr_bool is None:
        args.lr_bool = 150.0 if args.use_bn else 12.0
    print(f"lr_real={args.lr_real}, lr_bool={args.lr_bool}, "
          f"use_bn={args.use_bn}, strong_aug={args.strong_aug}")

    model = VGGSmall(use_bn=args.use_bn).to(device)
    n_b, n_r = count_parameters(model)
    print(f"params: {n_b:,} boolean + {n_r:,} real = {n_b + n_r:,} total")

    # Split parameters into boolean (tagged by BitLinear/BitConv2d) and real.
    bool_params, real_params = bopt.split_parameters(model)
    bool_opt = bopt.BooleanOptimizer(bool_params, lr=args.lr_bool)
    real_opt = torch.optim.Adam(real_params, lr=args.lr_real)

    bool_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bool_opt, T_max=args.epochs)
    real_sched = torch.optim.lr_scheduler.CosineAnnealingLR(real_opt, T_max=args.epochs)

    pin = device.type == "cuda"
    train_loader, test_loader = build_loaders(
        args.data_root, args.batch_size,
        num_workers=args.num_workers, strong_aug=args.strong_aug,
        pin_memory=pin,
    )
    print(f"train batches/epoch: {len(train_loader)}  test batches: {len(test_loader)}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, flips = train_one_epoch(
            model, train_loader, bool_opt, real_opt, device, epoch,
            use_mixup=args.strong_aug,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
        )
        te_loss, te_acc = evaluate(model, test_loader, device)
        bool_sched.step()
        real_sched.step()

        print(
            f"epoch {epoch:>3d}/{args.epochs}  "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  "
            f"test loss {te_loss:.4f} acc {te_acc:.4f}  "
            f"flips_this_epoch {flips:,}  "
            f"lr_real {real_opt.param_groups[0]['lr']:.2e}  "
            f"lr_bool {bool_opt.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if te_acc > best_acc:
            best_acc = te_acc
            if args.checkpoint:
                os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "test_acc": te_acc,
                     "use_bn": args.use_bn},
                    args.checkpoint,
                )

    print(f"\nbest test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
