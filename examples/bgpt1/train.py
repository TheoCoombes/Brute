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
    p.add_argument("--dim",         type=int, default=256)
    p.add_argument("--n-layers",    type=int, default=4)
    p.add_argument("--n-heads",     type=int, default=8)
    p.add_argument("--context",     type=int, default=128)
    p.add_argument("--hidden-mult", type=int, default=4)
    p.add_argument("--nu",          type=float, default=0.05)
    p.add_argument("--p-drop-max",  type=float, default=0.05)
    p.add_argument("--err-k",       type=int, default=8)
    p.add_argument("--err-clip",    type=int, default=7)
    # Bias args are no-ops in v2 (the LM-head bias was removed). Accepted
    # for backward-compat with old scripts; ignored by the model.
    p.add_argument("--bias-accum",  type=int, default=0,
                   help="(deprecated, ignored) LM-head bias was removed in v2")
    p.add_argument("--bias-clip",   type=int, default=0,
                   help="(deprecated, ignored) LM-head bias was removed in v2")
    # Tokenizer / data
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help='Tokenizer: "char" or HF tokenizer name (default: "gpt2")')
    p.add_argument("--max-vocab", type=int, default=None,
                   help="Cap on the vocab size; truncates HF tokens to ids < cap")
    p.add_argument("--corpus", type=str, default="tinyshakespeare",
                   choices=["tinyshakespeare", "wikitext-2-en", "wikitext-2", "wikitext-103"],
                   help="Training corpus (wikitext-2-en = English-only filtered)")
    # Training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps",      type=int, default=2000)
    p.add_argument("--t-init",     type=float, default=0.5)
    p.add_argument("--t-final",    type=float, default=0.05)
    p.add_argument("--target-flip-rate", type=float, default=0.005)
    p.add_argument("--decisive-threshold", type=float, default=0.02,
                   help="(legacy, diagnostic only — no longer gates flips)")
    p.add_argument("--mu-adjust",  type=float, default=0.02,
                   help="Multiplicative mu feedback step size (smaller = smoother)")
    p.add_argument("--acc-threshold", type=int, default=8,
                   help="BOLD accumulator |m| threshold to trigger a flip "
                        "(higher = slower/cleaner)")
    p.add_argument("--step-scale", type=float, default=8.0,
                   help="Accumulator step size per call = max(1, round(T*mu*step_scale))")
    p.add_argument("--target-flip-rate-final", type=float, default=None,
                   help="If set, target_flip_rate is cosine-annealed from "
                        "--target-flip-rate to this value over the run "
                        "(parallel to the temperature schedule).")
    p.add_argument("--keep-best", action="store_true",
                   help="Track best acc1 during training; at end, rewind "
                        "all binary weights to the best-acc1 snapshot.")
    p.add_argument("--keep-best-window", type=int, default=3,
                   help="Number of consecutive evals an acc1 improvement "
                        "must hold to count as a new 'best' (debounces noise).")
    p.add_argument("--depth-scale", type=float, default=1.0,
                   help="Per-layer threshold multiplier by depth. With ds=1.5, "
                        "embedding=1×acc_threshold, blk0=1.5×, blk1=2.25×, blk2=3.4×, ...")
    p.add_argument("--freeze-schedule", type=str, default="none",
                   choices=["none", "warmup", "sequential"],
                   help="Layer freeze schedule. 'none' = all layers train from step 0. "
                        "'warmup' = freeze body, train embedding+lm_head only until --freeze-warmup-steps. "
                        "'sequential' = train embedding only first, then unfreeze block by block.")
    p.add_argument("--freeze-warmup-steps", type=int, default=300,
                   help="For --freeze-schedule warmup: number of embedding-only training steps")
    p.add_argument("--sequential-block-steps", type=int, default=200,
                   help="For --freeze-schedule sequential: steps per block before unfreezing the next")
    p.add_argument("--n-ref-scale", type=float, default=1.0,
                   help="Manual n_ref scaling: n_ref = weight.numel() / n_ref_scale "
                        "(only affects diagnostic prints in v2)")
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


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 next-token accuracy (the only diagnostic that's robust to integer-
    logit sharpness — cross-entropy is uninformative because softmax over int
    logits is almost 1-hot regardless of how 'right' the prediction is)."""
    flat = logits.reshape(-1, logits.shape[-1])
    pred = flat.argmax(dim=-1)
    return (pred == targets.reshape(-1)).float().mean().item()


def compute_top5_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-5 next-token accuracy."""
    flat = logits.reshape(-1, logits.shape[-1])
    _, topk = flat.topk(5, dim=-1)
    tgt = targets.reshape(-1, 1)
    return (topk == tgt).any(dim=-1).float().mean().item()


def diagnose_logits(logits: torch.Tensor, targets: torch.Tensor) -> str:
    """Detailed diagnostic for integer-logit distribution and tie patterns.

    Returns a human-readable string with:
    - Logit value range and number of distinct values
    - Average tokens per logit value (= tie multiplicity)
    - Target rank distribution: where in the sorted top-K is the true target?
    - Tie analysis: of positions where argmax ≠ target, how many have
      logit[argmax] == logit[target] (i.e., they would have been correct
      under a different tiebreak)?
    """
    flat = logits.reshape(-1, logits.shape[-1])  # (B*C, V)
    tgts = targets.reshape(-1)                    # (B*C,)
    M, V = flat.shape

    # 1. Logit value distribution.
    lmin = int(flat.min().item())
    lmax = int(flat.max().item())
    n_unique = int(torch.unique(flat).numel())
    avg_ties = V / max(n_unique, 1)

    # 2. Target rank: how many tokens have a logit STRICTLY greater than target?
    # Rank 0 = target is the top-1.
    target_logit = flat.gather(1, tgts.unsqueeze(1))            # (M, 1)
    rank = (flat > target_logit).sum(dim=1)                      # (M,)
    rank_at_top = (rank == 0).float().mean().item()              # = acc1 (no tiebreak)
    rank_lt_5 = (rank < 5).float().mean().item()
    rank_lt_20 = (rank < 20).float().mean().item()
    median_rank = float(rank.float().median().item())

    # 3. Tied-at-top analysis. How often is target tied with the argmax?
    argmax = flat.argmax(dim=1)
    argmax_logit = flat.gather(1, argmax.unsqueeze(1)).squeeze(1)
    target_logit_1d = target_logit.squeeze(1)
    tied_with_top = (argmax_logit == target_logit_1d).float().mean().item()
    # Same but only for positions where argmax is wrong:
    wrong_mask = argmax != tgts
    n_wrong = int(wrong_mask.sum().item())
    if n_wrong > 0:
        tied_among_wrong = (
            (argmax_logit[wrong_mask] == target_logit_1d[wrong_mask]).float().mean().item()
        )
    else:
        tied_among_wrong = 0.0

    # 4. Argmax multiplicity — when argmax is "right", how many other tokens
    # tie with it (i.e., a flipped coin could have picked another one)?
    n_tied_with_argmax = (flat == argmax_logit.unsqueeze(1)).sum(dim=1).float()
    avg_argmax_ties = float(n_tied_with_argmax.mean().item())

    lines = [
        f"  ╭─ logit distribution",
        f"  │  range:           [{lmin}, {lmax}]  ({lmax - lmin + 1} possible int values)",
        f"  │  distinct used:   {n_unique}  /  V={V}  →  avg {avg_ties:.2f} tokens per logit value",
        f"  ├─ target rank (would-be acc1 with perfect tiebreak)",
        f"  │  rank=0:           {rank_at_top:.3f}",
        f"  │  rank<5:           {rank_lt_5:.3f}  (=acc5)",
        f"  │  rank<20:          {rank_lt_20:.3f}",
        f"  │  median rank:      {median_rank:.1f}",
        f"  ├─ tie analysis",
        f"  │  positions where target's logit == argmax's logit:",
        f"  │    overall:        {tied_with_top:.3f}",
        f"  │    among wrong:    {tied_among_wrong:.3f}  ← upper bound on tie-break gains",
        f"  │  avg tokens tied with argmax:  {avg_argmax_ties:.2f}",
        f"  ╰─",
    ]
    return "\n".join(lines)


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
    elif args.corpus in ("wikitext-2", "wikitext-2-en"):
        text = load_wikitext("wikitext-2-raw-v1", "train", english_only=(args.corpus == "wikitext-2-en"))
    elif args.corpus == "wikitext-103":
        text = load_wikitext("wikitext-103-raw-v1", "train", english_only=True)
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
        target_flip_rate_final=args.target_flip_rate_final,
        decisive_threshold=args.decisive_threshold,
        mu_adjust=args.mu_adjust,
        acc_threshold=args.acc_threshold,
        step_scale=args.step_scale,
    )
    model.register_with_optimizer(optim)
    for ls in optim.layers.values():
        ls.mu = args.mu_init
    if args.n_ref_scale != 1.0:
        optim._finalize_ref_sizes()
        for ls in optim.layers.values():
            ls.n_ref = max(ls.n_ref / args.n_ref_scale, 8.0)

    # Depth-scaled per-layer thresholds. Embedding+lm_head (tied) sees the
    # strongest error signal and can train at the base threshold. Each
    # deeper transformer block accumulates the previous block's updates as
    # noise, so we give it ``depth_scale ** depth`` more patience.
    def _layer_depth(name: str) -> int:
        if "blk" in name:
            try:
                return int(name.split("blk")[1].split(".")[0]) + 1
            except (ValueError, IndexError):
                return 0
        return 0  # embedding / lm_head

    if args.depth_scale != 1.0:
        print(f"  per-layer thresholds (depth_scale={args.depth_scale}):")
        for name, ls in optim.layers.items():
            depth = _layer_depth(name)
            thresh = int(round(args.acc_threshold * (args.depth_scale ** depth)))
            ls.acc_threshold_override = thresh
            print(f"    {name:24s} depth={depth} threshold={thresh}")

    # Optional freeze schedule. The schedule is applied dynamically in the
    # training loop (see below); here we only set the initial state.
    if args.freeze_schedule == "warmup":
        n_active = optim.freeze_all_except(["embedding"])
        print(f"  freeze warmup: training only embedding+lm_head ({n_active} layers) "
              f"for {args.freeze_warmup_steps} steps")
    elif args.freeze_schedule == "sequential":
        n_active = optim.freeze_all_except(["embedding"])
        print(f"  sequential unfreeze: starting with embedding only, then "
              f"unfreezing 1 block every {args.sequential_block_steps} steps")

    n_params = sum(ls.weight.numel() for ls in optim.layers.values())
    print(f"  total binary params: {n_params:,} ({n_params / 1e6:.2f} M)")
    print(f"  n_ref by layer:")
    for n, ls in optim.layers.items():
        print(f"    {n}: shape={tuple(ls.weight.shape)}, n_ref={ls.n_ref:.0f}")

    # Logging
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    log_path = _RUN_DIR / f"{run_name}.csv"
    log_fields = [
        "step", "loss", "acc1", "acc5", "entropy", "tie_rate",
        "T_base", "mu_mean", "f_mean", "beta_mean",
        "flips_step", "flips_total",
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

    # Best-state tracker (option --keep-best). We snapshot the packed weight
    # buffers of every registered FlipRule layer in CPU RAM. Cost = sum of
    # weights, which is the inference footprint. For our 10M-param target,
    # that's ~1 MB. Cheap.
    best_acc1 = -1.0
    best_step = -1
    best_snapshot: Dict[str, torch.Tensor] = {}
    # Debounce: only accept "new best" after acc1 holds above prior best
    # for ``keep_best_window`` consecutive evals. Avoids checkpointing noise.
    cand_acc1 = -1.0
    cand_step = -1
    cand_snapshot: Dict[str, torch.Tensor] = {}
    cand_count = 0

    for step in range(args.steps):
        # Dynamic freeze schedule.
        if args.freeze_schedule == "warmup" and step == args.freeze_warmup_steps:
            optim.unfreeze_all()
            print(f"  >> step {step}: unfreezing all body layers")
        elif args.freeze_schedule == "sequential":
            # After every ``sequential_block_steps`` steps, unfreeze one
            # more block (in order: blk0, blk1, ..., blk{L-1}). Embedding
            # stays trainable throughout.
            blocks_to_unfreeze = step // args.sequential_block_steps
            allowed = ["embedding"] + [f"blk{i}." for i in range(min(blocks_to_unfreeze, cfg.n_layers))]
            # Recompute frozen state — cheap and idempotent.
            if step > 0 and step % args.sequential_block_steps == 0 and blocks_to_unfreeze <= cfg.n_layers:
                n_active = optim.freeze_all_except(allowed)
                print(f"  >> step {step}: unfreezing up to blk{blocks_to_unfreeze - 1} "
                      f"({n_active} active layers)")

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
            acc1 = compute_accuracy(logits, y)
            acc5 = compute_top5_accuracy(logits, y)
            entropy = compute_logit_entropy(logits)
            tie = compute_logit_tie_rate(logits)
            snap = optim.snapshot()
            layer_keys = [k for k in snap.keys() if k != "__global__"]
            mu_mean = sum(snap[k]["mu"] for k in layer_keys) / len(layer_keys)
            f_mean = sum(snap[k]["f"] for k in layer_keys) / len(layer_keys)
            beta_mean = sum(snap[k]["beta"] for k in layer_keys) / len(layer_keys)
            row = {
                "step": step,
                "loss": f"{loss:.4f}",
                "acc1": f"{acc1:.4f}",
                "acc5": f"{acc5:.4f}",
                "entropy": f"{entropy:.4f}",
                "tie_rate": f"{tie:.4f}",
                "T_base": f"{snap['__global__']['T_base']:.4f}",
                "mu_mean": f"{mu_mean:.4f}",
                "f_mean": f"{f_mean:.5f}",
                "beta_mean": f"{beta_mean:.5f}",
                "flips_step": flips_step,
                "flips_total": flips_total,
            }
            log_writer.writerow(row)
            log_file.flush()
            elapsed = time.time() - t0
            # acc@1 of 1/V is uniform-random — anything above is real signal.
            uniform_acc = 1.0 / max(model.vocab_size, 1)
            print(f"  step {step:5d}/{args.steps}  loss {loss:7.2f}  "
                  f"acc1 {acc1:.3f} ({acc1/uniform_acc:5.1f}x)  acc5 {acc5:.3f}  "
                  f"H {entropy:4.2f}  tie {tie:.3f}  "
                  f"T {snap['__global__']['T_base']:.3f}  mu {mu_mean:.2f}  "
                  f"f {f_mean:.4f}  beta {beta_mean:.4f}  "
                  f"flips {flips_step:6d}  ({elapsed:.1f}s)")

            # Best-state tracker.
            if args.keep_best:
                if acc1 > cand_acc1 + 1e-4:
                    # New candidate best. Snapshot weights immediately.
                    cand_acc1 = acc1
                    cand_step = step
                    cand_count = 1
                    cand_snapshot = {
                        n: ls.weight._packed_buf.detach().clone()
                        for n, ls in optim.layers.items()
                    }
                else:
                    cand_count += 1
                # Promote candidate to "best" once it has held through the window.
                if cand_count >= args.keep_best_window and cand_acc1 > best_acc1:
                    best_acc1 = cand_acc1
                    best_step = cand_step
                    best_snapshot = cand_snapshot
                    print(f"    ★ new best acc1={best_acc1:.3f} (step {best_step}) "
                          f"promoted after holding {cand_count} evals")

        if step > 0 and step % args.eval_every == 0:
            # Per-layer flip distribution — diagnose which weights are
            # actually being updated.
            snap_dbg = optim.snapshot()
            print(f"  flip distribution @ step {step}:")
            for k in sorted(snap_dbg.keys()):
                if k == "__global__":
                    continue
                ls_n = optim.layers[k]
                pct = 100.0 * snap_dbg[k]["flip_total"] / max(ls_n.weight.numel(), 1)
                print(f"    {k:24s}  flips={snap_dbg[k]['flip_total']:8d}  "
                      f"({pct:5.1f}% of bits)  mu={snap_dbg[k]['mu']:.2f}  "
                      f"c_mean={snap_dbg[k]['c_mean']:.2f}")
            # Logit-distribution diagnostic — the key for understanding ties.
            print(f"  logit diagnostic @ step {step}:")
            print(diagnose_logits(logits, y))
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

    # Rewind to best-acc1 snapshot, if --keep-best was set and we found one.
    if args.keep_best and best_snapshot:
        print(f"\nrewinding weights to best-acc1 snapshot: "
              f"step {best_step}, acc1={best_acc1:.3f}")
        for name, packed in best_snapshot.items():
            optim.layers[name].weight._packed_buf.copy_(packed)
            optim.layers[name].weight.__dict__["_bool_dirty"] = True
        # Re-evaluate on a fresh batch to confirm the snapshot is good.
        x_eval, y_eval = next(batches)
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)
        logits_eval, _ = model(x_eval, return_tape=False)
        acc1_eval = compute_accuracy(logits_eval, y_eval)
        acc5_eval = compute_top5_accuracy(logits_eval, y_eval)
        print(f"  rewound model:  acc1 {acc1_eval:.3f}  acc5 {acc5_eval:.3f}")
    elif args.keep_best:
        print("  (no acc1 snapshot held long enough to promote — none restored)")

    # Final generation
    print("\nfinal sample:")
    model.eval()
    prompt = ids[: args.context // 2].to(device)
    gen = generate(model, tokenizer, prompt, n_new=args.eval_tokens)
    print(tokenizer.decode(gen[args.context // 2 :]))


if __name__ == "__main__":
    train(parse_args())
