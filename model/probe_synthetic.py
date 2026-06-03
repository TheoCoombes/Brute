"""Slow synthetic attention probes for HÆMMR.

These are not pytest tests. They are small training runs that expose what the
current binary model can and cannot do before spending time on WikiText.

Examples:
    python probe_synthetic.py --steps 180 --lengths 16 64
    python probe_synthetic.py --baseline --steps 180 --baseline-steps 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn

from bep import BepConfig, BepOptimizer
from model import HaemmrConfig, HaemmrLM, IGNORE_INDEX


TASKS = (
    "induction", "marker", "copy",
    "bsr_long", "hopfield_prior", "epi_distracted", "position_probe",
    "copy_distracted",
)


def _bsr_gap(seq_len: int) -> int:
    """Map benchmark lengths to the intended BSR gap sweep."""
    if seq_len >= 40:
        return 32
    if seq_len >= 24:
        return 16
    if seq_len >= 16:
        return 8
    return max(1, seq_len - 3)


def make_batch(task: str, batch_size: int, seq_len: int, *, device="cpu"):
    X = torch.randint(8, 16, (batch_size, seq_len), dtype=torch.long, device=device)
    Y = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    if task == "induction":
        a = torch.randint(0, 4, (batch_size,), device=device)
        b = a + 4
        X.fill_(9)
        X[:, 0] = a
        X[:, 1] = b
        X[:, -2] = a
        Y[:, -2] = b
    elif task == "marker":
        marker = torch.randint(0, 2, (batch_size,), device=device)
        X[:, 0] = marker
        X[:, -2] = 2
        Y[:, -2] = marker + 3
    elif task == "copy":
        value = torch.randint(0, 8, (batch_size,), device=device)
        X[:, 0] = value
        X[:, -2] = 2
        Y[:, -2] = value
    elif task == "bsr_long":
        gap = _bsr_gap(seq_len)
        if 1 + gap >= seq_len:
            raise ValueError(f"seq_len={seq_len} is too short for bsr_long gap={gap}")
        a = torch.randint(0, 4, (batch_size,), device=device)
        X[:, 0] = a
        X[:, 1:1 + gap] = torch.randint(
            8, 12, (batch_size, gap), dtype=torch.long, device=device)
        X[:, 1 + gap] = a
        Y[:, 1 + gap] = a
    elif task == "hopfield_prior":
        dominant = torch.randint(0, 4, (batch_size,), device=device)
        X[:, 0] = dominant
        if seq_len > 3:
            X[:, 1:-2] = torch.randint(
                4, 16, (batch_size, seq_len - 3), dtype=torch.long, device=device)
        X[:, -2] = 14
        Y[:, -2] = dominant
    elif task == "epi_distracted":
        n_distract = min(3, max(0, seq_len - 3))
        q_pos = n_distract + 1
        key = torch.randint(0, 4, (batch_size,), device=device)
        value = key + 4
        X[:, 0] = key
        if n_distract:
            X[:, 1:q_pos] = torch.randint(
                8, 16, (batch_size, n_distract), dtype=torch.long, device=device)
        X[:, q_pos] = key
        Y[:, q_pos] = value
    elif task == "position_probe":
        mid = seq_len // 2
        a = torch.randint(0, 4, (batch_size,), device=device)
        X.fill_(9)
        first_half = torch.arange(batch_size, device=device) < (batch_size // 2)
        X[first_half, 0] = a[first_half]
        Y[first_half, 0] = 5
        X[~first_half, mid] = a[~first_half]
        Y[~first_half, mid] = 6
    elif task == "copy_distracted":
        if seq_len < 6:
            raise ValueError("copy_distracted requires seq_len >= 6")
        value = torch.randint(0, 8, (batch_size,), device=device)
        q1 = seq_len // 2
        q2 = seq_len - 2
        X[:, 0] = value
        X[:, q1] = 2
        X[:, q2] = 2
        Y[:, q1] = value
        Y[:, q2] = value
    else:
        raise ValueError(f"unknown task {task!r}")
    return X, Y


def matched_slots(task: str, X: torch.Tensor) -> torch.Tensor:
    """Local slot targets for the v2 episodic address-margin objective."""
    B, n = X.shape
    matched = torch.full((B, n), -1, dtype=torch.long, device=X.device)
    if task in ("marker", "copy"):
        matched[:, -2] = 0
    elif task == "induction":
        matched[:, -2] = 1
    elif task == "bsr_long":
        gap = _bsr_gap(n)
        if 1 + gap < n:
            matched[:, 1 + gap] = 0
    elif task == "hopfield_prior":
        matched[:, -2] = 0
    elif task == "epi_distracted":
        n_distract = min(3, max(0, n - 3))
        matched[:, n_distract + 1] = 0
    elif task == "position_probe":
        pass
    elif task == "copy_distracted":
        matched[:, n // 2] = 0
        matched[:, -2] = 0
    else:
        raise ValueError(f"unknown task {task!r}")
    return matched


def eval_model(model, task: str, seq_len: int, *, device="cpu", batches=4, batch_size=256):
    tot_acc = tot_loss = tot = 0.0
    with torch.no_grad():
        for _ in range(batches):
            X, Y = make_batch(task, batch_size, seq_len, device=device)
            info = model.metrics(model.forward(X), Y)
            tot += info["n_valid"]
            tot_acc += info["acc"] * info["n_valid"]
            tot_loss += info["loss"] * info["n_valid"]
    return {"acc": tot_acc / max(tot, 1), "loss": tot_loss / max(tot, 1)}


def run_haemmr(task: str, seq_len: int, args):
    torch.manual_seed(args.seed)
    structured_codebook = args.codebook_mode != "random"
    cfg = HaemmrConfig(
        vocab_size=16, D=args.D, n_layers=args.layers, d_ff=args.d_ff,
        n_slots=args.slots, top_k=args.top_k, seed=args.seed,
        epi_slots=args.epi_slots, epi_read_k=args.epi_read_k,
        decay_shifts=args.decay_shifts,
        use_position=args.use_position, gate_open=args.gate_open, r=args.r,
        bits=args.bits, flip_dropout=args.flip_dropout,
        structured_codebook=structured_codebook,
        bef_sweeps=args.bef_sweeps, sem_weight=args.sem_weight,
        use_bsr=args.use_bsr, use_episodic=args.use_episodic,
        use_hopfield=args.use_hopfield,
    )
    model = HaemmrLM(cfg, device=args.device)
    opt = BepOptimizer(model.parameters(), BepConfig(r=args.r, bits=args.bits))
    for _ in range(args.steps):
        X, Y = make_batch(task, args.batch_size, seq_len, device=args.device)
        matched = matched_slots(task, X) if args.margin_supervision else None
        model.loss_and_backward(model.forward(X), Y, matched=matched)
        opt.step()
    return eval_model(model, task, seq_len, device=args.device)


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size=16, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(256, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        n = x.shape[1]
        h = self.emb(x) + self.pos[:n].unsqueeze(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        return self.head(self.encoder(h, mask=mask))


def run_transformer(task: str, seq_len: int, args):
    torch.manual_seed(args.seed)
    model = TinyCausalTransformer(d_model=args.baseline_dim, layers=args.baseline_layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.baseline_lr)
    for _ in range(args.baseline_steps):
        X, Y = make_batch(task, args.batch_size, seq_len, device=args.device)
        logits = model(X).reshape(-1, 16)
        loss = torch.nn.functional.cross_entropy(logits, Y.reshape(-1), ignore_index=IGNORE_INDEX)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        X, Y = make_batch(task, 1024, seq_len, device=args.device)
        logits = model(X).reshape(-1, 16)
        valid = Y.reshape(-1) != IGNORE_INDEX
        pred = logits.argmax(dim=1)
        acc = float((pred[valid] == Y.reshape(-1)[valid]).float().mean())
        loss = float(torch.nn.functional.cross_entropy(logits, Y.reshape(-1),
                                                       ignore_index=IGNORE_INDEX).item())
    return {"acc": acc, "loss": loss}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--lengths", nargs="+", type=int, default=[16, 64])
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--slots", type=int, default=32)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--epi-slots", type=int, default=None)
    p.add_argument("--epi-read-k", type=int, default=1)
    p.add_argument("--r", type=float, default=0.1, help="BEP margin trigger fraction.")
    p.add_argument("--gate-open", type=float, default=0.05)
    p.add_argument("--bits", type=int, default=15)
    p.add_argument("--flip-dropout", type=float, default=0.0)
    p.add_argument("--sem-weight", type=float, default=0.5)
    p.add_argument("--decay-shifts", type=lambda s: tuple(int(x) for x in s.split(",")),
                   default=(1, 2, 3, 4, 0),
                   help='comma-separated BSR decay shifts, e.g. "1,2,3,4,0"')
    p.add_argument("--codebook-mode", choices=("structured", "random"),
                   default="structured",
                   help="structured = inline BEF; random = unstructured codebook.")
    p.add_argument("--bef-sweeps", type=int, default=30)
    p.add_argument("--no-bsr", dest="use_bsr", action="store_false", default=True)
    p.add_argument("--no-episodic", dest="use_episodic", action="store_false", default=True)
    p.add_argument("--no-hopfield", dest="use_hopfield", action="store_false", default=True)
    p.add_argument("--no-position", dest="use_position", action="store_false", default=True,
                   help="disable hierarchical positions in the episodic address lane")
    p.add_argument("--no-margin-supervision", dest="margin_supervision",
                   action="store_false", default=True,
                   help="disable local matched-slot supervision for synthetic retrieval probes")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--baseline-steps", type=int, default=300)
    p.add_argument("--baseline-dim", type=int, default=64)
    p.add_argument("--baseline-layers", type=int, default=2)
    p.add_argument("--baseline-lr", type=float, default=3e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    rows = []
    t0 = time.time()
    for task in args.tasks:
        for length in args.lengths:
            h = run_haemmr(task, length, args)
            row = {"model": "haemmr", "task": task, "length": length, **h}
            rows.append(row)
            print(json.dumps(row))
            if args.baseline:
                b = run_transformer(task, length, args)
                row = {"model": "transformer", "task": task, "length": length, **b}
                rows.append(row)
                print(json.dumps(row))
    print(f"elapsed={time.time() - t0:.1f}s")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
