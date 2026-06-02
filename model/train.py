"""Train a HÆMMR binary language model on WikiText with the GPT-2 tokenizer.

Everything is 1-bit: the codebook, binding masks, Hopfield slots, channel-mix
weights and residual gates are all ``brute.bit1`` and trained by BOLD bit-flips
(no floating-point latent weights).  Runs locally on a Mac CPU.

Quick start (a few minutes)::

    python train.py --steps 1500 --D 1024 --layers 2 --vocab-cap 2048 --seq-len 64

Useful flags::

    --D            concept hypervector dim (bigger ⇒ better VSA geometry)
    --layers       number of Boolean blocks
    --vocab-cap    cap the 50k GPT-2 vocab to the top-N tokens (keeps decode fast)
    --eta          BOLD accumulation rate η
    --threshold    integrated evidence needed before a bit flips
    --device       cpu | mps | cuda  (default: auto)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import brute  # noqa: F401 — ensure the extension is importable

from bold import BoldConfig, BoldOptimizer
from data import OOV_ID, make_lm_batches, wikitext
from model import HaemmrConfig, HaemmrLM, IGNORE_INDEX


def auto_device(choice):
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unigram_baseline(targets: torch.Tensor, V: int) -> float:
    flat = targets.reshape(-1)
    flat = flat[flat != IGNORE_INDEX]
    if flat.numel() == 0:
        return 0.0
    counts = torch.bincount(flat, minlength=V)
    return float(counts.max().item()) / float(flat.numel())


@torch.no_grad()
def evaluate(model, X, Y, batch_size, max_batches=40):
    tot_loss = tot_acc = tot = 0
    nb = min((X.shape[0] + batch_size - 1) // batch_size, max_batches)
    for b in range(nb):
        s = b * batch_size
        xb, yb = X[s:s + batch_size], Y[s:s + batch_size]
        info = model.metrics(model.forward(xb), yb)
        nv = info["n_valid"]
        tot_loss += info["loss"] * nv
        tot_acc += info["acc"] * nv
        tot += nv
    loss = tot_loss / max(tot, 1)
    return {"loss": loss, "acc": tot_acc / max(tot, 1),
            "ppl": float(torch.exp(torch.tensor(loss)).item())}


def sample_demo(model, corpus, prompt, n_new, *, temperature, top_k, ban_oov, rep_window):
    ids = corpus.encode(prompt).unsqueeze(0).to(model.device)
    ban = [OOV_ID] if ban_oov else None
    out = model.generate(ids, n_new, temperature=temperature, top_k=top_k,
                         ban_ids=ban, repetition_window=rep_window)
    new_ids = out[0, ids.shape[1]:].cpu()
    text = corpus.decode(new_ids).replace("\n", "\\n")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    p.add_argument("--data-root", default="./.data")
    p.add_argument("--vocab-cap", type=int, default=2048)
    p.add_argument("--max-train-tokens", type=int, default=None,
                   help="Truncate the training stream (faster local demo).")
    p.add_argument("--D", type=int, default=1024, help="Concept hypervector dim.")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--slots", type=int, default=256, help="Hopfield bank slots M.")
    p.add_argument("--top-k", type=int, default=15, help="Hopfield WTA width (odd).")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500, help="number of optimiser (flip) steps.")
    p.add_argument("--eta", type=float, default=3.0, help="BOLD accumulation rate η.")
    p.add_argument("--threshold", type=float, default=8.0,
                   help="integrated evidence needed to flip a bit (accumulator hysteresis).")
    p.add_argument("--m-clip", type=int, default=127, help="int8 accumulator saturation (±).")
    p.add_argument("--eta-decay", type=float, default=1.0)
    p.add_argument("--eta-end", type=float, default=None,
                   help="anneal η geometrically to this value by the last step.")
    p.add_argument("--log-csv", default=None, help="append the loss curve to this CSV file.")
    p.add_argument("--no-position", dest="use_position", action="store_false", default=True,
                   help="don't bind ρ^i(POS); let order come from the BSR decay (RWKV-style).")
    p.add_argument("--position-decode", default=None, choices=["none", "next_unbind"],
                   help="decode positioned concepts safely; default is next_unbind when positions are on.")
    p.add_argument("--gate-open", type=float, default=0.05,
                   help="residual-gate init openness (higher ⇒ context flows sooner).")
    p.add_argument("--codebook-flip-scale", type=float, default=0.3,
                   help="codebook flip-rate relative to transforms (lower = more stable).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--prompt", default="The history of")
    p.add_argument("--sample-len", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--sample-top-k", type=int, default=20)
    p.add_argument("--rep-window", type=int, default=3)
    p.add_argument("--ckpt", default="./haemmr.pt")
    args = p.parse_args()

    device = auto_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    print(f"loading {args.dataset} (vocab cap {args.vocab_cap}) …")
    corpus = wikitext(name=args.dataset, data_root=args.data_root,
                      vocab_cap=args.vocab_cap, max_train_tokens=args.max_train_tokens)
    V = corpus.vocab_size
    Xtr, Ytr = make_lm_batches(corpus.train_ids, seq_len=args.seq_len, mask_oov=True,
                               seed=args.seed, shuffle=True)
    Xva, Yva = make_lm_batches(corpus.val_ids, seq_len=args.seq_len, mask_oov=True,
                               seed=args.seed, shuffle=False)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)
    print(f"  train tokens={corpus.train_ids.numel():,}  val tokens={corpus.val_ids.numel():,}"
          f"  vocab={V}  chunks: train={Xtr.shape[0]:,} val={Xva.shape[0]:,}")
    print(f"  baselines (val):  uniform={1.0/V:.4f}  unigram={unigram_baseline(Yva.cpu(), V):.4f}")

    # Optional η annealing: explore more early, then let the integer accumulator
    # settle into smaller, steadier evidence updates late in training.
    eta_decay = args.eta_decay
    if args.eta_end is not None and args.steps > 0:
        eta_decay = (args.eta_end / args.eta) ** (1.0 / args.steps)

    cfg = HaemmrConfig(vocab_size=V, D=args.D, n_layers=args.layers, d_ff=args.d_ff,
                       n_slots=args.slots, top_k=args.top_k, seed=args.seed,
                       use_position=args.use_position, gate_open=args.gate_open,
                       codebook_flip_scale=args.codebook_flip_scale,
                       position_decode=args.position_decode)
    model = HaemmrLM(cfg, device=device)
    opt = BoldOptimizer(model.parameters(),
                        BoldConfig(eta=args.eta, eta_decay=eta_decay,
                                   threshold=args.threshold, m_clip=args.m_clip))
    n_bits = model.num_bit_parameters()
    print(f"model: D={cfg.D} layers={cfg.n_layers} d_ff={cfg.d_ff} slots={cfg.n_slots}"
          f"  |  {n_bits:,} bit-params ≈ {n_bits/8/1e6:.2f} MB")
    init = evaluate(model, Xva, Yva, args.batch_size)
    print(f"step 0    val loss {init['loss']:.3f}  ppl {init['ppl']:.1f}  acc {init['acc']:.4f}")

    csv_f = None
    if args.log_csv:
        csv_f = open(args.log_csv, "a")
        if csv_f.tell() == 0:
            csv_f.write("step,train_ema,val_loss,val_ppl,val_acc,flip_frac,eta\n")

    bs = args.batch_size
    n_chunks = Xtr.shape[0]
    order = torch.randperm(n_chunks)
    ptr = 0
    run_loss = run_acc = run_n = 0.0
    ema = None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if ptr + bs > n_chunks:
            order = torch.randperm(n_chunks)
            ptr = 0
        idx = order[ptr:ptr + bs]
        ptr += bs
        xb, yb = Xtr[idx], Ytr[idx]
        info = model.loss_and_backward(model.forward(xb), yb)
        step_loss = info["loss"] * info["n_valid"]
        step_acc = info["acc"] * info["n_valid"]
        step_n = info["n_valid"]

        st = opt.step()
        b_loss = step_loss / max(step_n, 1)
        run_loss += step_loss
        run_acc += step_acc
        run_n += step_n
        ema = b_loss if ema is None else 0.98 * ema + 0.02 * b_loss

        if step % args.eval_every == 0 or step == args.steps:
            tr_loss = run_loss / max(run_n, 1)
            tr_acc = run_acc / max(run_n, 1)
            run_loss = run_acc = run_n = 0.0
            va = evaluate(model, Xva, Yva, bs)
            dt = time.time() - t0
            print(f"step {step:5d}  train loss {tr_loss:.3f} (ema {ema:.3f}) acc {tr_acc:.4f}  |  "
                  f"val loss {va['loss']:.3f} ppl {va['ppl']:.1f} acc {va['acc']:.4f}  |  "
                  f"flip {st['flip_frac']*100:.3f}% η{st['eta']:.2f}  ({dt:.0f}s)")
            if csv_f:
                csv_f.write(f"{step},{ema:.4f},{va['loss']:.4f},{va['ppl']:.2f},"
                            f"{va['acc']:.4f},{st['flip_frac']:.5f},{st['eta']:.4f}\n")
                csv_f.flush()

        if args.sample_every and step % args.sample_every == 0:
            txt = sample_demo(model, corpus, args.prompt, args.sample_len,
                              temperature=args.temperature, top_k=args.sample_top_k,
                              ban_oov=True, rep_window=args.rep_window)
            print(f"  sample[{args.prompt!r}]: {txt}")

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "vocab_size": V, "compact_to_gpt2": corpus.compact_to_gpt2},
               args.ckpt)
    print(f"saved checkpoint → {args.ckpt}")
    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
