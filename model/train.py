"""Train a HÆMMR binary language model on WikiText with the GPT-2 tokenizer.

Everything is 1-bit: the codebook, binding masks, episodic address projections,
Hopfield slots, channel-mix weights and residual gates are all ``brute.bit1``
and trained by BOLD bit-flips (no floating-point latent weights). Runs locally
on a Mac CPU.

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
from data import OOV_ID, make_lm_batches, wikitext, tiny_shakespeare
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
    total_batches = (X.shape[0] + batch_size - 1) // batch_size
    nb = total_batches if max_batches is None or max_batches <= 0 else min(total_batches, max_batches)
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


def save_checkpoint(path, model, opt, V, corpus, *, step, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "vocab_size": V,
        "compact_to_gpt2": corpus.compact_to_gpt2,
        "step": int(step),
        "metrics": dict(metrics),
    }, path)


def sample_demo(model, corpus, prompt, n_new, *, temperature, top_k, ban_oov,
                rep_window, sentence, min_new, clean):
    ids = corpus.encode(prompt).unsqueeze(0).to(model.device)
    ban = corpus.sample_ban_ids(clean=clean) if ban_oov or clean else None
    stop = corpus.sentence_end_ids() if sentence else None
    out = model.generate(ids, n_new, temperature=temperature, top_k=top_k,
                         ban_ids=ban, repetition_window=rep_window,
                         stop_ids=stop, min_new=min_new)
    show_ids = out[0] if sentence else out[0, ids.shape[1]:]
    text = corpus.decode(show_ids.cpu()).strip().replace("\n", "\\n")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="wikitext-2",
                   choices=["wikitext-2", "wikitext-103", "tiny-shakespeare"])
    p.add_argument("--data-root", default="./.data")
    p.add_argument("--vocab-cap", type=int, default=2048)
    p.add_argument("--max-train-tokens", type=int, default=None,
                   help="Truncate the returned training stream after building the vocab.")
    p.add_argument("--D", type=int, default=1024, help="Concept hypervector dim.")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--slots", type=int, default=256, help="Hopfield bank slots M.")
    p.add_argument("--top-k", type=int, default=15, help="Hopfield WTA width (odd).")
    p.add_argument("--no-bsr", dest="use_bsr", action="store_false", default=True,
                   help="disable BSR accumulator path for packed-only hardware profiles.")
    p.add_argument("--epi-window", type=int, default=None,
                   help="fixed episodic ring width W (None = chunk-bounded).")
    p.add_argument("--epi-read-k", type=int, default=1,
                   help="episodic top-k read width (1 = exact single-slot).")
    p.add_argument("--epi-chunk", type=int, default=64,
                   help="episodic chunk size C for the chunked ring search.")
    p.add_argument("--epi-bonus", type=float, default=0.0,
                   help="episodic shortlist-seeding weight at decode time (0 = off).")
    p.add_argument("--no-multiscale", dest="multiscale", action="store_false", default=True,
                   help="disable per-depth scaling of episodic windows and BSR decay.")
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
                   help="disable hierarchical position codes in the episodic address lane.")
    p.add_argument("--no-structured-codebook", dest="structured_codebook",
                   action="store_false", default=True,
                   help="use random token codes instead of the small-model BEF initializer.")
    p.add_argument("--bef-sweeps", type=int, default=30)
    p.add_argument("--sem-weight", type=float, default=0.5,
                   help="semantic rerank weight added to lexical decode logits.")
    p.add_argument("--boundary-nu", type=float, default=None,
                   help="BEP-style per-linear boundary eligibility gate; unset disables it.")
    p.add_argument("--highway-clip", type=int, default=15,
                   help="clipped integer concept-highway bound |u| ≤ S (int4/int5).")
    p.add_argument("--highway-carry", type=int, default=1,
                   help="λ identity-carry coefficient of the concept highway.")
    p.add_argument("--alpha-epi", type=int, default=4,
                   help="episodic branch scale (dominant so exact recall can override).")
    p.add_argument("--alpha-bsr", type=int, default=1)
    p.add_argument("--alpha-hop", type=int, default=1)
    p.add_argument("--alpha-ff", type=int, default=1)
    p.add_argument("--highway-nu", type=float, default=None,
                   help="highway boundary gate: only |u| ≤ ν·S bits propagate (None = off).")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--flip-dropout", type=float, default=0.0)
    p.add_argument("--sem-flip-scale", type=float, default=0.5,
                   help="semantic-bank flip-rate relative to transforms.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-max-batches", type=int, default=40,
                   help="validation batches per eval; <=0 evaluates the full validation split.")
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--prompt", default=None,
                   help="sampling prompt; defaults to 'The history of' for WikiText "
                        "and 'ROMEO:\\n' for TinyShakespeare.")
    p.add_argument("--sample-len", type=int, default=40)
    p.add_argument("--sample-min-len", type=int, default=12)
    p.add_argument("--sentence-sample", action="store_true",
                   help="include the prompt and stop samples at sentence punctuation when possible.")
    p.add_argument("--clean-sample", action="store_true",
                   help="ban OOV, control, non-ASCII, and continuation-fragment tokens while sampling.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--sample-top-k", type=int, default=20)
    p.add_argument("--rep-window", type=int, default=3)
    p.add_argument("--ckpt", default="./haemmr.pt")
    p.add_argument("--last-ckpt", default=None,
                   help="optional path for the final-step checkpoint; --ckpt stores the best validation model.")
    args = p.parse_args()

    device = auto_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    print(f"loading {args.dataset} …")
    if args.dataset == "tiny-shakespeare":
        corpus = tiny_shakespeare(data_root=args.data_root,
                                  max_train_tokens=args.max_train_tokens)
    else:
        print(f"  vocab cap {args.vocab_cap}")
        corpus = wikitext(name=args.dataset, data_root=args.data_root,
                          vocab_cap=args.vocab_cap, max_train_tokens=args.max_train_tokens)
    V = corpus.vocab_size
    Xtr, Ytr = make_lm_batches(corpus.train_ids, seq_len=args.seq_len, mask_oov=True,
                               seed=args.seed, shuffle=True)
    Xva, Yva = make_lm_batches(corpus.val_ids, seq_len=args.seq_len, mask_oov=True,
                               seed=args.seed, shuffle=False)
    train_oov = float((corpus.train_ids == OOV_ID).float().mean().item())
    val_oov = float((corpus.val_ids == OOV_ID).float().mean().item())
    train_valid = float((Ytr != IGNORE_INDEX).float().mean().item())
    val_valid = float((Yva != IGNORE_INDEX).float().mean().item())
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)
    print(f"  train tokens={corpus.train_ids.numel():,}  val tokens={corpus.val_ids.numel():,}"
          f"  vocab={V}  chunks: train={Xtr.shape[0]:,} val={Xva.shape[0]:,}")
    print(f"  OOV rate: train={train_oov:.1%} val={val_oov:.1%}"
          f"  |  valid targets: train={train_valid:.1%} val={val_valid:.1%}")
    print(f"  acc baselines (val valid targets):"
          f"  uniform={1.0/V:.4f}  unigram={unigram_baseline(Yva.cpu(), V):.4f}")
    if val_oov > 0.2:
        print("  warning: high OOV rate; this is a lossy frequent-token demo, not full WikiText LM training.")
    eval_scope = "full" if args.eval_max_batches <= 0 else f"first {args.eval_max_batches} batches"
    print(f"  eval scope: {eval_scope}")

    # Optional η annealing: explore more early, then let the integer accumulator
    # settle into smaller, steadier evidence updates late in training.
    eta_decay = args.eta_decay
    if args.eta_end is not None and args.steps > 0:
        eta_decay = (args.eta_end / args.eta) ** (1.0 / args.steps)

    cfg = HaemmrConfig(vocab_size=V, D=args.D, n_layers=args.layers, d_ff=args.d_ff,
                       n_slots=args.slots, top_k=args.top_k, seed=args.seed,
                       use_bsr=args.use_bsr,
                       epi_window=args.epi_window, epi_read_k=args.epi_read_k,
                       epi_chunk=args.epi_chunk, epi_bonus_weight=args.epi_bonus,
                       multiscale=args.multiscale,
                       highway_clip=args.highway_clip, highway_carry=args.highway_carry,
                       alpha_epi=args.alpha_epi, alpha_bsr=args.alpha_bsr,
                       alpha_hop=args.alpha_hop, alpha_ff=args.alpha_ff,
                       highway_nu=args.highway_nu,
                       use_position=args.use_position,
                       structured_codebook=args.structured_codebook,
                       bef_sweeps=args.bef_sweeps, sem_weight=args.sem_weight,
                       sem_flip_scale=args.sem_flip_scale,
                       boundary_nu=args.boundary_nu,
                       label_smoothing=args.label_smoothing,
                       flip_dropout=args.flip_dropout,
                       )
    model = HaemmrLM(cfg, device=device)
    opt = BoldOptimizer(model.parameters(),
                        BoldConfig(eta=args.eta, eta_decay=eta_decay,
                                   threshold=args.threshold, m_clip=args.m_clip))
    n_bits = model.num_bit_parameters()
    print(f"model: D={cfg.D} layers={cfg.n_layers} d_ff={cfg.d_ff} slots={cfg.n_slots}"
          f"  |  {n_bits:,} bit-params ≈ {n_bits/8/1e6:.2f} MB")
    init = evaluate(model, Xva, Yva, args.batch_size, max_batches=args.eval_max_batches)
    best = {"step": 0, **init}
    save_checkpoint(args.ckpt, model, opt, V, corpus, step=0, metrics=best)
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

        do_eval = (args.eval_every > 0 and step % args.eval_every == 0) or step == args.steps
        if do_eval:
            tr_loss = run_loss / max(run_n, 1)
            tr_acc = run_acc / max(run_n, 1)
            run_loss = run_acc = run_n = 0.0
            va = evaluate(model, Xva, Yva, bs, max_batches=args.eval_max_batches)
            improved = va["loss"] < best["loss"]
            if improved:
                best = {"step": step, **va}
                save_checkpoint(args.ckpt, model, opt, V, corpus, step=step, metrics=best)
            dt = time.time() - t0
            best_mark = "  *best*" if improved else ""
            print(f"step {step:5d}  train loss {tr_loss:.3f} (ema {ema:.3f}) acc {tr_acc:.4f}  |  "
                  f"val loss {va['loss']:.3f} ppl {va['ppl']:.1f} acc {va['acc']:.4f}  |  "
                  f"flip {st['flip_frac']*100:.3f}% η{st['eta']:.2f}  ({dt:.0f}s){best_mark}")
            if csv_f:
                csv_f.write(f"{step},{ema:.4f},{va['loss']:.4f},{va['ppl']:.2f},"
                            f"{va['acc']:.4f},{st['flip_frac']:.5f},{st['eta']:.4f}\n")
                csv_f.flush()

        if args.sample_every and step % args.sample_every == 0:
            prompt = args.prompt or (
                "ROMEO:\n" if args.dataset == "tiny-shakespeare" else "The history of")
            txt = sample_demo(model, corpus, prompt, args.sample_len,
                              temperature=args.temperature, top_k=args.sample_top_k,
                              ban_oov=True, rep_window=args.rep_window,
                              sentence=args.sentence_sample,
                              min_new=args.sample_min_len,
                              clean=args.clean_sample)
            print(f"  sample[{args.prompt!r}]: {txt}")

    if args.last_ckpt:
        final_metrics = va if args.steps > 0 and "va" in locals() else init
        save_checkpoint(args.last_ckpt, model, opt, V, corpus,
                        step=args.steps, metrics={"step": args.steps, **final_metrics})
        print(f"saved final checkpoint → {args.last_ckpt}")
    print(f"saved best checkpoint → {args.ckpt}"
          f"  (step {best['step']}, val loss {best['loss']:.3f}, acc {best['acc']:.4f})")
    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
