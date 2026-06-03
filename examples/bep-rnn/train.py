"""Train a fully binary BEP RNN language model on TinyShakespeare.

The model is a single Elman-style binary recurrent cell unrolled through
time, with input embedding and output classifier sharing one fixed
Binary Equiangular Frame (BEF) codebook over a capped GPT-2 vocabulary.

Quick run (a few minutes on a laptop CPU)::

    python train.py --epochs 3 --hidden 256 --vocab-cap 512 --seq-len 64

Useful flags::

    --vocab-cap   N      cap the (otherwise 50k-sized) GPT-2 vocab to top-N
    --hidden      K_h    recurrent-state width (must be divisible by group)
    --seq-len     T      BPTT context window
    --r           0.5    BEP trigger margin (Eq. 1)
    --nu          0.05   backward gating threshold (Eq. 5)
    --group-size  4      neurons per winner-takes-update group

A sanity-check report at the start lists the uniform-baseline accuracy
(``1 / vocab_size``) and the unigram-baseline (always predict the most
frequent token) so it's easy to see when the model actually beats them.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Allow running as a script without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import brute  # noqa: F401 — ensures brute is importable in the env

from bef import generate_bef
from bep import BEPConfig, BEPLanguageModel, IGNORE_INDEX
from data import OOV_ID, make_lm_batches, tinyshakespeare


def _unigram_baseline(targets: torch.Tensor, vocab_size: int) -> float:
    """Accuracy of always predicting the most-frequent target token."""
    flat = targets.reshape(-1)
    flat = flat[flat != IGNORE_INDEX]
    counts = torch.bincount(flat, minlength=vocab_size)
    return float(counts.max().item()) / max(int(flat.numel()), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=256,
                   help="Hidden-state width K_h (must be divisible by --group-size).")
    p.add_argument("--vocab-cap", type=int, default=512,
                   help="Cap on the GPT-2 vocabulary (top-N most frequent tokens).")
    p.add_argument("--seq-len", type=int, default=64, help="BPTT context window T.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--r", type=float, default=0.5, help="Update margin (Eq. 1).")
    p.add_argument("--nu", type=float, default=0.05,
                   help="Backward gating threshold (Eq. 5).")
    p.add_argument("--group-size", type=int, default=4,
                   help="Initial neuron group size (winner-takes-update).")
    p.add_argument("--p-reinforce", type=float, default=0.5)
    p.add_argument("--weight-clip", type=int, default=2048)
    p.add_argument("--bef-iters", type=int, default=None,
                   help="Coordinate-flip iterations for the BEF codebook.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--data-root", default="./.data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    p.add_argument("--sample-every", type=int, default=0,
                   help="If >0, emit a greedy sample after every N epochs.")
    p.add_argument("--sample-len", type=int, default=80)
    p.add_argument("--mask-oov", action="store_true", default=True,
                   help="Replace OOV targets with IGNORE_INDEX so the model "
                        "doesn't waste updates predicting the OOV bucket.")
    p.add_argument("--no-mask-oov", dest="mask_oov", action="store_false")
    p.add_argument("--ban-oov-sampling", action="store_true", default=True,
                   help="At decode time, suppress the OOV token in argmax.")
    p.add_argument("--prompt", default="First Citizen:\n",
                   help="Text prompt used for periodic sampling.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-k sampling at decode (1 = greedy).")
    p.add_argument("--temperature", type=float, default=1.0)
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
    print(f"loading TinyShakespeare (cap={args.vocab_cap}) …")
    corpus = tinyshakespeare(
        data_root=args.data_root,
        vocab_cap=args.vocab_cap,
        val_frac=args.val_frac,
    )
    V = corpus.vocab_size
    print(f"  corpus: train={corpus.train_ids.numel():,} tokens   "
          f"val={corpus.val_ids.numel():,} tokens   vocab={V}")

    tr_in, tr_tg = make_lm_batches(
        corpus.train_ids, seq_len=args.seq_len, batch_size=args.batch_size,
        seed=args.seed, shuffle=False,
        mask_target_id=(OOV_ID if args.mask_oov else None),
    )
    va_in, va_tg = make_lm_batches(
        corpus.val_ids, seq_len=args.seq_len, batch_size=args.batch_size,
        seed=args.seed, shuffle=False,
        mask_target_id=(OOV_ID if args.mask_oov else None),
    )
    tr_in = tr_in.to(device); tr_tg = tr_tg.to(device)
    va_in = va_in.to(device); va_tg = va_tg.to(device)
    print(f"  chunks: train={tr_in.shape[0]:,}   val={va_in.shape[0]:,}   "
          f"T={args.seq_len}")

    # ── Baselines ───────────────────────────────────────────────────────────
    uniform = 1.0 / V
    unigram = _unigram_baseline(va_tg.cpu(), V)
    print(f"  baselines on val:   uniform = {uniform:.4f}   "
          f"unigram = {unigram:.4f}")

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"generating BEF codebook ({V} × {args.hidden}) …")
    P = generate_bef(V, args.hidden, iters=args.bef_iters, seed=args.seed,
                     device=device)
    cfg = BEPConfig(
        r=args.r,
        nu=args.nu,
        group_size_init=args.group_size,
        p_reinforce=args.p_reinforce,
        weight_clip=args.weight_clip,
    )
    model = BEPLanguageModel(
        vocab_size=V, hidden_size=args.hidden, classifier=P,
        config=cfg, device=device, seed=args.seed,
    )
    print(model)

    # Initial accuracy.
    init_va = model.accuracy(va_in, va_tg, batch_size=args.batch_size)
    print(f"epoch 0 / {args.epochs}   val_acc = {init_va:.4f}")

    # ── Training loop ───────────────────────────────────────────────────────
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    best_va = init_va
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(tr_in.shape[0], generator=g).to(device)
        tr_in_e = tr_in[perm]
        tr_tg_e = tr_tg[perm]

        n_correct = n_seen = n_trig = 0
        n_batches = (tr_in_e.shape[0] + args.batch_size - 1) // args.batch_size
        for b in range(n_batches):
            s = b * args.batch_size
            e = min(s + args.batch_size, tr_in_e.shape[0])
            xb = tr_in_e[s:e]
            yb = tr_tg_e[s:e]
            state = model.forward(xb)
            info = model.step(state, yb)
            n_correct += info["n_correct"]
            n_seen += info["n_seen"]
            n_trig += info["n_triggered"]

        tr_acc = n_correct / max(n_seen, 1)
        va_acc = model.accuracy(va_in, va_tg, batch_size=args.batch_size)
        best_va = max(best_va, va_acc)
        dt = time.time() - t0
        # "Bit-perplexity" — interpret the top-1 accuracy as a (very crude)
        # lower bound on the implied log-likelihood; here we just report
        # the equivalent geometric-mean uncertainty 1/acc, which is what
        # you'd care about most when comparing to a vocab baseline.
        eff_choices = 1.0 / max(va_acc, 1e-9)
        print(
            f"epoch {ep:2d} / {args.epochs}   "
            f"train_acc = {tr_acc:.4f}   val_acc = {va_acc:.4f}   "
            f"triggered = {n_trig / max(n_seen,1):.3f}   "
            f"eff_choices = {eff_choices:.1f}/{V}   "
            f"({dt:.1f}s)"
        )

        if args.sample_every and ep % args.sample_every == 0:
            prompt_ids = _encode_prompt(corpus, args.prompt).to(device).unsqueeze(0)
            print("  sample:", _sample(
                model, corpus, prompt_ids, args.sample_len,
                ban_oov=args.ban_oov_sampling,
                top_k=args.top_k, temperature=args.temperature,
            ))

    print(f"\nbest val accuracy: {best_va:.4f}   "
          f"(uniform {uniform:.4f} | unigram {unigram:.4f})")


def _encode_prompt(corpus, prompt: str) -> torch.Tensor:
    """Encode a prompt string into compact ids. Tokens outside the capped
    vocabulary remain as the OOV id (they still produce a valid embedding,
    just from the OOV prototype)."""
    gpt2_ids = corpus.tokenizer.encode(prompt)
    table = torch.full((int(corpus.compact_to_gpt2.max().item()) + 1,),
                       OOV_ID, dtype=torch.long)
    table[corpus.compact_to_gpt2] = torch.arange(corpus.vocab_size, dtype=torch.long)
    ids = table[torch.tensor(gpt2_ids, dtype=torch.long)]
    if ids.numel() == 0:
        ids = torch.tensor([1], dtype=torch.long)
    return ids


def _sample(model: BEPLanguageModel, corpus, prompt: torch.Tensor,
            n_new: int, *, ban_oov: bool = True, top_k: int = 5,
            temperature: float = 1.0,
            repetition_window: int = 3) -> str:
    """Decode ``n_new`` tokens given a prompt.

    Uses top-``k`` softmax sampling over the integer ±1 logits (logits are
    in ``[-K_h, K_h]``; we treat them as raw scores and apply a temperature
    rescale before softmax). ``top_k=1`` gives greedy decoding. Sampling
    avoids the well-known repetition trap of greedy RNN decoding.
    """
    seq = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            state = model.forward(seq)
            lg = state.logits[:, -1].to(torch.float32)
            if ban_oov:
                lg[:, OOV_ID] = float("-inf")
            # Cheap anti-attractor: forbid emitting any of the last
            # ``repetition_window`` tokens — a binary RNN's greedy
            # distribution is spiky and falls into "x x x x x ..." loops
            # otherwise. Pure sampling/temperature alone is not enough at
            # this model capacity.
            if repetition_window > 0 and seq.shape[1] >= 1:
                recent = seq[0, -repetition_window:].tolist()
                for r in recent:
                    lg[:, int(r)] = float("-inf")
            if top_k > 1:
                topv, topi = torch.topk(lg, k=top_k, dim=1)
                probs = torch.softmax(topv / max(temperature, 1e-3), dim=1)
                pick = torch.multinomial(probs, num_samples=1)
                nxt = topi.gather(1, pick)
            else:
                nxt = lg.argmax(dim=1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
    # Decode only the newly generated tail — the prompt is already known.
    new_ids = seq[0, prompt.shape[1]:].cpu()
    text = corpus.decode(new_ids)
    # Quote whitespace so collapsed-to-newline samples are still visible.
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    return repr(text)[1:-1]  # strip outer quotes from repr()


if __name__ == "__main__":
    main()
