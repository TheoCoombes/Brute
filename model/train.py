"""Train a HÆMMR binary language model on WikiText with the GPT-2 tokenizer.

The hot path stays packed-bit: the codebook, binding masks, episodic address
projections, Hopfield slots, channel-mix weights, and residual gates all live
as ``brute.bit1`` tensors. Training uses BEP/BOLD-style integer hidden weights
(``bep.py``): each parameter stores an int16 ``H`` buffer, the visible weight is
``sign(H)``, and the backward path threads binary desired activations instead of
float gradients.

Quick start (a few minutes)::

    python train.py --steps 1500 --D 1024 --layers 2 --seq-len 64

Useful flags::

    --D            concept hypervector dimension
    --layers       number of Boolean blocks
    --codebook-mode {offline,structured,random}
                  offline GPT-2 SimHash by default; structured = inline BEF;
                  random = unstructured codebook
    --r            margin trigger fraction for lexical updates
    --bits         hidden-weight clamp width
    --gate-open    residual gate initial openness
    --no-position  disable hierarchical position codes in the episodic lane
    --sem-weight   semantic rerank weight added to lexical decode logits
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

from bep import BepConfig, BepOptimizer
from data import make_lm_batches, wikitext
from model import HaemmrConfig, HaemmrLM, IGNORE_INDEX


CODEBOOK_MODES = ("offline", "structured", "random")


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


def sample_demo(model, corpus, prompt, n_new, *, temperature, top_k, rep_window):
    ids = corpus.encode(prompt).unsqueeze(0).to(model.device)
    out = model.generate(ids, n_new, temperature=temperature, top_k=top_k,
                         repetition_window=rep_window)
    new_ids = out[0, ids.shape[1]:].cpu()
    text = corpus.decode(new_ids).replace("\n", "\\n")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    p.add_argument("--data-root", default="./.data")
    p.add_argument("--max-train-tokens", type=int, default=None,
                   help="Truncate the training stream (faster local demo).")
    p.add_argument("--D", type=int, default=1024, help="Concept hypervector dim.")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--slots", type=int, default=256, help="Hopfield bank slots M.")
    p.add_argument("--top-k", type=int, default=15, help="Hopfield WTA width (odd).")
    p.add_argument("--epi-slots", type=int, default=None,
                   help="episodic exact-recall window; default = full sequence during training.")
    p.add_argument("--epi-read-k", type=int, default=1,
                   help="episodic top-k read width (1 = exact single-slot).")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500, help="number of optimiser (flip) steps.")
    p.add_argument("--r", type=float, default=0.1,
                   help="BEP margin trigger: update fires when logit[tgt] − max_other < r·D.")
    p.add_argument("--p-r", type=float, default=0.0,
                   help="CP+R reinforcement probability (BEP §3.3).")
    p.add_argument("--bits", type=int, default=15, help="integer hidden-weight H bit-width.")
    p.add_argument("--log-csv", default=None, help="append the loss curve to this CSV file.")
    p.add_argument("--no-position", dest="use_position", action="store_false", default=True,
                   help="disable hierarchical position codes in the episodic address lane.")
    p.add_argument(
        "--codebook-mode",
        choices=CODEBOOK_MODES,
        default="offline",
        help="offline GPT-2 SimHash by default; structured uses inline BEF; "
             "random uses an unstructured codebook.",
    )
    p.add_argument("--bef-sweeps", type=int, default=30)
    p.add_argument("--sem-weight", type=float, default=0.5,
                   help="semantic rerank weight added to lexical decode logits.")
    p.add_argument("--boundary-nu", type=float, default=None,
                   help="BEP-style boundary eligibility gate; unset disables it.")
    p.add_argument("--flip-dropout", type=float, default=0.0)
    p.add_argument("--gate-open", type=float, default=0.05,
                   help="residual-gate init openness (higher ⇒ context flows sooner).")
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

    codebook_mode = args.codebook_mode
    structured_codebook = codebook_mode != "random"
    offline_codebook = codebook_mode == "offline"

    print(f"loading {args.dataset} with full GPT-2 tokenizer (codebook={codebook_mode}) …")
    corpus = wikitext(name=args.dataset, data_root=args.data_root,
                      max_train_tokens=args.max_train_tokens)
    V = corpus.vocab_size
    Xtr, Ytr = make_lm_batches(corpus.train_ids, seq_len=args.seq_len,
                               seed=args.seed, shuffle=True)
    Xva, Yva = make_lm_batches(corpus.val_ids, seq_len=args.seq_len,
                               seed=args.seed, shuffle=False)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)
    print(f"  train tokens={corpus.train_ids.numel():,}  val tokens={corpus.val_ids.numel():,}"
          f"  vocab={V}  chunks: train={Xtr.shape[0]:,} val={Xva.shape[0]:,}")
    print(f"  baselines (val):  uniform={1.0/V:.4f}  unigram={unigram_baseline(Yva.cpu(), V):.4f}")

    cfg = HaemmrConfig(vocab_size=V, D=args.D, n_layers=args.layers, d_ff=args.d_ff,
                       n_slots=args.slots, top_k=args.top_k, seed=args.seed,
                       epi_slots=args.epi_slots, epi_read_k=args.epi_read_k,
                       use_position=args.use_position, gate_open=args.gate_open,
                       structured_codebook=structured_codebook,
                       bef_sweeps=args.bef_sweeps, sem_weight=args.sem_weight,
                       boundary_nu=args.boundary_nu,
                       flip_dropout=args.flip_dropout,
                       r=args.r, p_r=args.p_r, bits=args.bits,
                       )
    codebook_init = None
    if offline_codebook:
        from codebook import from_corpus
        print(f"building offline GPT-2 SimHash codebook (D={args.D}) …")
        codebook_init = from_corpus(corpus, args.D, seed=args.seed,
                                    cache_dir=str(Path(args.data_root) / "codebook"))
    model = HaemmrLM(cfg, device=device, codebook_init=codebook_init)
    opt = BepOptimizer(model.parameters(),
                       BepConfig(r=args.r, p_r=args.p_r, bits=args.bits))
    n_bits = model.num_bit_parameters()
    print(f"model: D={cfg.D} layers={cfg.n_layers} d_ff={cfg.d_ff} slots={cfg.n_slots}"
          f"  |  {n_bits:,} bit-params ≈ {n_bits/8/1e6:.2f} MB")
    init = evaluate(model, Xva, Yva, args.batch_size)
    print(f"step 0    val loss {init['loss']:.3f}  ppl {init['ppl']:.1f}  acc {init['acc']:.4f}")

    csv_f = None
    if args.log_csv:
        csv_f = open(args.log_csv, "a")
        if csv_f.tell() == 0:
            csv_f.write("step,train_ema,val_loss,val_ppl,val_acc,flip_frac\n")

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
                  f"flip {st['flip_frac']*100:.3f}%  ({dt:.0f}s)")
            if csv_f:
                csv_f.write(f"{step},{ema:.4f},{va['loss']:.4f},{va['ppl']:.2f},"
                            f"{va['acc']:.4f},{st['flip_frac']:.5f}\n")
                csv_f.flush()

        if args.sample_every and step % args.sample_every == 0:
            txt = sample_demo(model, corpus, args.prompt, args.sample_len,
                              temperature=args.temperature, top_k=args.sample_top_k,
                              rep_window=args.rep_window)
            print(f"  sample[{args.prompt!r}]: {txt}")

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "vocab_size": V, "compact_to_gpt2": corpus.compact_to_gpt2},
               args.ckpt)
    print(f"saved checkpoint → {args.ckpt}")
    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
