"""BGPT-1 Phase-1 training script.

Standalone — depends only on ``brute``, ``torch``, ``tokenizers`` and
the local ``examples.bgpt1`` package. Trains a small (~few-hundred-K
binary param) BGPT-1 on Tiny Shakespeare under the streaming flip rule.

Usage::

    python -m examples.bgpt1.train \
        --dim 128 --n-layers 4 --n-heads 8 --context 64 \
        --batch-size 16 --steps 2000 --tokenizer char --device cpu

CSV log is written to ``examples/bgpt1/.runs/<timestamp>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict

import torch

import brute
from brute.optim import FlipRule
from brute.nn import signed_int_error

# Local package
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.bgpt1.model import BGPT1, BGPT1Config
from examples.bgpt1.data import (
    load_corpus_text,
    load_wikitext,
    load_tokenizer,
    tokenize_corpus,
    batch_iter,
)
from examples.bgpt1.memory import memory_report

_RUN_DIR = HERE / ".runs"
_RUN_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--dim",         type=int, default=128)
    p.add_argument("--n-layers",    type=int, default=4)
    p.add_argument("--n-heads",     type=int, default=8)
    p.add_argument("--context",     type=int, default=64)
    p.add_argument("--hidden-mult", type=int, default=4)
    p.add_argument("--nu",          type=float, default=0.05)
    p.add_argument("--p-drop-max",  type=float, default=0.05)
    p.add_argument("--err-k",       type=int, default=8)
    p.add_argument("--err-clip",    type=int, default=7)
    p.add_argument("--bias-accum",  type=int, default=32,
                   help="Threshold for LM head bias bounded-accumulator updates")
    # Tokenizer / data
    p.add_argument("--tokenizer", type=str, default="char",
                   help='Tokenizer: "char" or HF tokenizer name (e.g. "gpt2")')
    p.add_argument("--max-vocab", type=int, default=None,
                   help="Cap on the vocab size; truncates HF tokens to ids < cap")
    p.add_argument("--corpus", type=str, default="tinyshakespeare",
                   choices=["tinyshakespeare", "wikitext-2", "wikitext-103"],
                   help="Training corpus")
    # Training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps",      type=int, default=2000)
    p.add_argument("--t-init",     type=float, default=0.5)
    p.add_argument("--t-final",    type=float, default=0.01)
    p.add_argument("--target-flip-rate", type=float, default=0.005)
    p.add_argument("--decisive-threshold", type=float, default=0.05)
    p.add_argument("--n-ref-scale", type=float, default=4.0,
                   help="Manual n_ref scaling: n_ref = weight.numel() / n_ref_scale")
    p.add_argument("--mu-init", type=float, default=1.0)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--device",     type=str, default="cpu")
    # Logging
    p.add_argument("--log-every",  type=int, default=50)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-tokens",type=int, default=200)
    p.add_argument("--run-name",   type=str, default=None)
    return p.parse_args()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Cross-entropy treating int logits as raw scores. Used for diagnostics
    only — the model's training signal is the signed-integer error, not
    the gradient of this loss."""
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
    ).item()


def compute_logit_entropy(logits: torch.Tensor) -> float:
    """Mean Shannon entropy of softmax(logits) — diagnoses output collapse."""
    p = torch.softmax(logits.reshape(-1, logits.shape[-1]).float(), dim=-1)
    h = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    return h.mean().item()


def compute_logit_tie_rate(logits: torch.Tensor) -> float:
    """Fraction of predicting positions whose top-2 logits are tied (int)."""
    flat = logits.reshape(-1, logits.shape[-1])
    top2, _ = flat.topk(2, dim=-1)
    return (top2[:, 0] == top2[:, 1]).float().mean().item()


@torch.no_grad()
def generate(
    model: BGPT1,
    tokenizer,
    prompt_ids: torch.Tensor,
    n_new: int,
    temperature: float = 1.0,
) -> list[int]:
    """Greedy / sampled completion from the model."""
    cfg = model.config
    device = next(iter(model.buffers())).device
    ctx = prompt_ids.to(device)
    out = ctx.tolist()
    model.eval()
    for _ in range(n_new):
        crop = ctx[-cfg.max_context :].unsqueeze(0)
        logits, _ = model(crop, return_tape=False)
        last = logits[0, -1].float() / max(temperature, 1e-3)
        # Add jitter equal to 0.5 to break integer ties without changing
        # the top-k ordering above tie groups.
        last = last + 0.5 * torch.rand_like(last)
        next_id = int(last.argmax().item())
        out.append(next_id)
        ctx = torch.cat([ctx, torch.tensor([next_id], device=device)])
    return out


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = args.device

    # Data
    print(f"loading corpus ({args.corpus}) ...")
    if args.corpus == "tinyshakespeare":
        text = load_corpus_text()
    elif args.corpus == "wikitext-2":
        text = load_wikitext("wikitext-2-raw-v1", "train")
    elif args.corpus == "wikitext-103":
        text = load_wikitext("wikitext-103-raw-v1", "train")
    else:
        raise ValueError(f"unknown corpus {args.corpus!r}")
    tokenizer = load_tokenizer(args.tokenizer, text=text)
    ids, V = tokenize_corpus(text, tokenizer, max_vocab=args.max_vocab)
    print(f"  corpus tokens: {ids.numel():,}, vocab: {V}")

    # Model
    cfg = BGPT1Config(
        vocab_size=V,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_context=args.context,
        hidden_mult=args.hidden_mult,
        nu=args.nu,
        p_drop_max=args.p_drop_max,
        err_k=args.err_k,
        err_clip=args.err_clip,
        bias_accum=args.bias_accum,
    )
    print("BGPT1 config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    model = BGPT1(cfg, device=device)

    # Optimizer
    optim = FlipRule(
        t_init=args.t_init,
        t_final=args.t_final,
        total_steps=args.steps,
        target_flip_rate=args.target_flip_rate,
        decisive_threshold=args.decisive_threshold,
    )
    model.register_with_optimizer(optim)
    # Phase-1 n_ref override: divide by `n_ref_scale` to amplify flip
    # probability in small toy models (default 4 → ~25% of weights flip
    # per batch at T_base=0.5 on small layers).
    for ls in optim.layers.values():
        ls.n_ref = max(ls.weight.numel() / args.n_ref_scale, 8.0)
        ls.mu = args.mu_init

    n_params = sum(ls.weight.numel() for ls in optim.layers.values())
    print(f"  total binary params: {n_params:,} ({n_params / 1e6:.2f} M)")
    print(f"  n_ref by layer:")
    for n, ls in optim.layers.items():
        print(f"    {n}: shape={tuple(ls.weight.shape)}, n_ref={ls.n_ref:.0f}")

    # Logging
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    log_path = _RUN_DIR / f"{run_name}.csv"
    log_fields = [
        "step", "loss", "entropy", "tie_rate",
        "T_base", "mu_mean", "f_mean", "flips_step", "flips_total",
        "n_bias_updates",
    ]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()
    print(f"  logging to {log_path}")

    # Train loop
    model.train()
    batches = batch_iter(ids, args.batch_size, args.context, seed=args.seed)
    t0 = time.time()
    flips_total = 0
    printed_mem_report = False
    for step in range(args.steps):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device)
        prev_total = sum(ls.flip_total for ls in optim.layers.values())
        logits, tape = model(x, return_tape=True, return_diag=False)
        # Memory report after the first forward (tape is materialized).
        if not printed_mem_report:
            print("\n" + memory_report(model, trainer=optim, tape=tape))
            print()
            printed_mem_report = True
        diag = model.backward_step(tape, y, optim)
        optim.step_global()
        flips_step = sum(ls.flip_total for ls in optim.layers.values()) - prev_total
        flips_total += flips_step

        if step % args.log_every == 0 or step == args.steps - 1:
            loss = compute_loss(logits, y)
            entropy = compute_logit_entropy(logits)
            tie = compute_logit_tie_rate(logits)
            snap = optim.snapshot()
            layer_keys = [k for k in snap.keys() if k != "__global__"]
            mu_mean = sum(snap[k]["mu"] for k in layer_keys) / len(layer_keys)
            f_mean = sum(snap[k]["f"] for k in layer_keys) / len(layer_keys)
            row = {
                "step": step,
                "loss": f"{loss:.4f}",
                "entropy": f"{entropy:.4f}",
                "tie_rate": f"{tie:.4f}",
                "T_base": f"{snap['__global__']['T_base']:.4f}",
                "mu_mean": f"{mu_mean:.4f}",
                "f_mean": f"{f_mean:.5f}",
                "flips_step": flips_step,
                "flips_total": flips_total,
                "n_bias_updates": int(diag["n_bias_updates"]),
            }
            log_writer.writerow(row)
            log_file.flush()
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{args.steps}  loss {loss:7.3f}  "
                  f"H {entropy:5.2f}  tie {tie:.3f}  T {snap['__global__']['T_base']:.3f}  "
                  f"mu {mu_mean:.3f}  f {f_mean:.4f}  flips {flips_step:6d}  "
                  f"bias_upd {int(diag['n_bias_updates']):3d}  ({elapsed:.1f}s)")

        if step > 0 and step % args.eval_every == 0:
            print(f"\n  sample @ step {step}:")
            model.eval()
            prompt = ids[: args.context // 2].to(device)
            gen = generate(model, tokenizer, prompt, n_new=args.eval_tokens)
            text_out = tokenizer.decode(gen[args.context // 2 :])
            print(f"    {text_out!r}\n")
            model.train()

    log_file.close()
    print(f"\ntraining done. wall: {time.time() - t0:.1f}s, total flips: {flips_total:,}")
    print(f"log: {log_path}")

    # Final generation
    print("\nfinal sample:")
    model.eval()
    prompt = ids[: args.context // 2].to(device)
    gen = generate(model, tokenizer, prompt, n_new=args.eval_tokens)
    print(tokenizer.decode(gen[args.context // 2 :]))


if __name__ == "__main__":
    train(parse_args())
