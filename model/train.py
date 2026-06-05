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
    --codebook-mode {structured,random}
                  structured balanced hash by default; random = alternate balanced hash seed
    --r            margin trigger fraction for lexical updates
    --bits         hidden-weight clamp width
    --block-init-inertia
                  initial |H| for stacked blocks (default: 8 for stable depth)
    --block-update-clip
                  elementwise ΔH clamp for stacked blocks (default: 1)
    --readout-warmup-steps
                  early codebook-only steps before hidden BEP updates
    --margin-r-final / --margin-anneal-steps
                  anneal the BEP margin after readout warmup
    --max-trigger-rate
                  cap actual hidden BEP rows per batch
    --gate-open    residual gate initial openness
    --no-position  disable hierarchical position codes in the episodic lane
    --sem-weight   semantic rerank weight added to lexical decode logits
    --device       cpu | mps | cuda  (default: cpu)
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
from data import make_lm_batches, wikitext, tiny_shakespeare, tiny_shakespeare_char
from model import TransformerConfig, BinaryTransformerLM, IGNORE_INDEX


CODEBOOK_MODES = ("structured", "random")


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
    p.add_argument("--dataset", default="wikitext-2",
                   choices=["wikitext-2", "wikitext-103", "tiny-shakespeare", "tiny-shakespeare-char"])
    p.add_argument("--data-root", default="./.data")
    p.add_argument("--max-train-tokens", type=int, default=None,
                   help="Truncate the training stream (faster local demo).")
    p.add_argument("--D", type=int, default=512, help="Concept / model dimension.")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4, help="attention heads (D//heads %% 64 == 0).")
    p.add_argument("--d-ff", type=int, default=1024, help="binary GLU hidden width.")
    p.add_argument("--attn-mode", choices=["soft", "hardmax"], default="soft",
                   help="soft = int8 vote bundle; hardmax = packed argmax gather.")
    p.add_argument("--attn-band", type=int, default=1, help="soft attention integer margin band.")
    p.add_argument("--residual-mode", choices=["mux", "majority"], default="mux",
                   help="mux = branch replaces skip (copy); majority = maj3 (BOLD-faithful).")
    p.add_argument("--no-value-proj", dest="value_proj", action="store_false", default=True,
                   help="raw-concept-copy attention (no W_V/W_O) — cleaner retrieval transport.")
    p.add_argument("--causal-strict", action="store_true", default=False,
                   help="exclude self (j<i): ALiBi recency then selects the previous token.")
    p.add_argument("--no-alibi", dest="alibi", action="store_false", default=True)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500, help="number of optimiser (flip) steps.")
    p.add_argument("--r", type=float, default=0.1,
                   help="BEP margin trigger: update fires when logit[tgt] − max_other < r·D.")
    p.add_argument("--p-r", type=float, default=0.0,
                   help="CP+R reinforcement probability (BEP §3.3).")
    p.add_argument("--bits", type=int, default=15, help="integer hidden-weight H bit-width.")
    p.add_argument("--init-inertia", type=int, default=1,
                   help="initial |H| for non-codebook head/readout parameters.")
    p.add_argument("--update-clip", type=int, default=None,
                   help="optional elementwise ΔH clamp for non-codebook head/readout parameters.")
    p.add_argument("--block-init-inertia", type=int, default=8,
                   help="initial |H| for stacked blocks; higher damps first-batch rewrites.")
    p.add_argument("--block-update-clip", type=int, default=1,
                   help="optional elementwise ΔH clamp for stacked blocks.")
    p.add_argument("--readout-warmup-steps", type=int, default=25,
                   help="codebook-only steps before hidden/block BEP updates.")
    p.add_argument("--freeze-codebook-after-warmup", action="store_true", default=False,
                   help="freeze codebook after readout warmup: transformer trains against fixed char prototypes.")
    p.add_argument("--margin-r-final", type=float, default=0.0,
                   help="final BEP margin fraction after annealing.")
    p.add_argument("--margin-anneal-steps", type=int, default=100,
                   help="steps after readout warmup over which r decays.")
    p.add_argument("--max-trigger-rate", type=float, default=0.25,
                   help="cap actual hidden BEP update rows as a fraction of valid rows.")
    p.add_argument("--log-csv", default=None, help="append the loss curve to this CSV file.")
    p.add_argument("--gate-open", type=float, default=0.05,
                   help="residual admittance-gate init openness (→ identity init).")
    p.add_argument(
        "--codebook-mode",
        choices=CODEBOOK_MODES,
        default="structured",
        help="structured balanced hash by default; random uses an alternate balanced hash seed.",
    )
    p.add_argument("--bef-sweeps", type=int, default=30)
    p.add_argument("--sem-weight", type=float, default=0.5,
                   help="semantic rerank weight added to lexical decode logits.")
    p.add_argument("--boundary-nu", type=float, default=None,
                   help="BEP-style boundary eligibility gate; unset disables it.")
    p.add_argument("--flip-dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--prompt", default="ROMEO:")
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

    print(f"loading {args.dataset} with full GPT-2 tokenizer (codebook={codebook_mode}) …")
    if args.dataset == "tiny-shakespeare-char":
        corpus = tiny_shakespeare_char(data_root=args.data_root)
    elif args.dataset == "tiny-shakespeare":
        # convert token cap to char cap (≈4 chars/token) so we truncate before encoding
        max_chars = args.max_train_tokens * 5 if args.max_train_tokens is not None else None
        corpus = tiny_shakespeare(data_root=args.data_root, max_chars=max_chars)
    else:
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

    cfg = TransformerConfig(vocab_size=V, D=args.D, n_layers=args.layers,
                            n_heads=args.n_heads, d_ff=args.d_ff, seed=args.seed,
                            attn_mode=args.attn_mode, attn_band=args.attn_band,
                            residual_mode=args.residual_mode, value_proj=args.value_proj,
                            causal_strict=args.causal_strict, alibi=args.alibi,
                            gate_open=args.gate_open,
                            structured_codebook=structured_codebook,
                            bef_sweeps=args.bef_sweeps, sem_weight=args.sem_weight,
                            boundary_nu=args.boundary_nu,
                            init_inertia=args.init_inertia,
                            update_clip=args.update_clip,
                            block_init_inertia=args.block_init_inertia,
                            block_update_clip=args.block_update_clip,
                            readout_warmup_steps=args.readout_warmup_steps,
                            freeze_codebook_after_warmup=args.freeze_codebook_after_warmup,
                            margin_r_final=args.margin_r_final,
                            margin_anneal_steps=args.margin_anneal_steps,
                            max_trigger_rate=args.max_trigger_rate,
                            flip_dropout=args.flip_dropout,
                            r=args.r, p_r=args.p_r, bits=args.bits,
                            )
    model = BinaryTransformerLM(cfg, device=device)
    opt = BepOptimizer(model.parameters(),
                       BepConfig(r=args.r, p_r=args.p_r, bits=args.bits))
    n_bits = model.num_bit_parameters()
    print(f"model: D={cfg.D} layers={cfg.n_layers} heads={cfg.n_heads} d_ff={cfg.d_ff}"
          f" attn={cfg.attn_mode} resid={cfg.residual_mode} value_proj={cfg.value_proj}"
          f" block_H0={cfg.block_init_inertia or cfg.init_inertia}"
          f" warmup={cfg.readout_warmup_steps}"
          f" r={cfg.r}->{cfg.margin_r_final if cfg.margin_r_final is not None else cfg.r}"
          f" cap={cfg.max_trigger_rate}"
          f"  |  {n_bits:,} bit-params ≈ {n_bits/8/1e6:.2f} MB")
    init = evaluate(model, Xva, Yva, args.batch_size)
    print(f"step 0    val loss {init['loss']:.3f}  ppl {init['ppl']:.1f}  acc {init['acc']:.4f}")

    csv_f = None
    if args.log_csv:
        csv_f = open(args.log_csv, "a")
        if csv_f.tell() == 0:
            csv_f.write("step,train_ema,val_loss,val_ppl,val_acc,flip_frac,"
                        "trigger_rate,margin_trigger_rate,effective_r\n")

    bs = args.batch_size
    n_chunks = Xtr.shape[0]
    order = torch.randperm(n_chunks)
    ptr = 0
    run_loss = run_acc = run_n = run_trigger = run_margin_trigger = run_eff_r = 0.0
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
        run_trigger += info["trigger_rate"]
        run_margin_trigger += info.get("margin_trigger_rate", info["trigger_rate"])
        run_eff_r += info.get("effective_r", args.r)
        ema = b_loss if ema is None else 0.98 * ema + 0.02 * b_loss

        if step % args.eval_every == 0 or step == args.steps:
            tr_loss = run_loss / max(run_n, 1)
            tr_acc = run_acc / max(run_n, 1)
            denom = max(args.eval_every if step % args.eval_every == 0 else step % args.eval_every, 1)
            tr_trigger = run_trigger / denom
            tr_margin_trigger = run_margin_trigger / denom
            tr_eff_r = run_eff_r / denom
            run_loss = run_acc = run_n = run_trigger = run_margin_trigger = run_eff_r = 0.0
            va = evaluate(model, Xva, Yva, bs)
            dt = time.time() - t0
            print(f"step {step:5d}  train loss {tr_loss:.3f} (ema {ema:.3f}) acc {tr_acc:.4f}  |  "
                  f"val loss {va['loss']:.3f} ppl {va['ppl']:.1f} acc {va['acc']:.4f}  |  "
                  f"flip {st['flip_frac']*100:.3f}% trigger {tr_trigger:.3f}/{tr_margin_trigger:.3f}"
                  f" r {tr_eff_r:.3f}  ({dt:.0f}s)")
            if csv_f:
                csv_f.write(f"{step},{ema:.4f},{va['loss']:.4f},{va['ppl']:.2f},"
                            f"{va['acc']:.4f},{st['flip_frac']:.5f},{tr_trigger:.5f},"
                            f"{tr_margin_trigger:.5f},{tr_eff_r:.5f}\n")
                csv_f.flush()

        if args.sample_every and step % args.sample_every == 0:
            txt = sample_demo(model, corpus, args.prompt, args.sample_len,
                              temperature=args.temperature, top_k=args.sample_top_k,
                              rep_window=args.rep_window)
            print(f"  sample[{args.prompt!r}]: {txt}")

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "vocab_size": V},
               args.ckpt)
    print(f"saved checkpoint → {args.ckpt}")
    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
