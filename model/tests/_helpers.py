"""Shared test helpers: synthetic attention tasks, a float transformer baseline,
and binary-model factories with the verified-learnable recipe.

The synthetic tasks isolate *what attention must do*:

* :func:`prev_token_batch` — ``target[i] = token[i-1]``.  Position-free concepts
  cannot select position ``i-1`` by content, so this is a pure **recency** task:
  ALiBi + strict-causal must select the previous token and the value path must
  transport it.  Genuinely non-local (the answer is never in ``x_i``).
* :func:`induction_batch` — in-context induction with a per-sequence random
  follower map (only solvable by recalling a token's earlier follower).
* :func:`copy_shift_batch` — fixed-content copy used for value-path checks.

The float :class:`TransformerBaseline` is a standard tiny decoder transformer
(position-free embeddings + integer-free ALiBi + causal attention) used to show
the binary model lands in the same accuracy regime on the same tasks, and that
its integer attention scores track real (float) ``q·k`` attention.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TransformerConfig, BinaryTransformerLM, IGNORE_INDEX
from bep import BepConfig, BepOptimizer


# ── synthetic tasks ─────────────────────────────────────────────────────────────

def prev_token_batch(B: int, n: int, V: int, gen: torch.Generator):
    """``target[i] = token[i-1]`` (position 0 ignored).  Returns (X, Y, matched)."""
    X = torch.randint(0, V, (B, n), generator=gen)
    Y = torch.full((B, n), IGNORE_INDEX, dtype=torch.long)
    Y[:, 1:] = X[:, :-1]
    matched = torch.full((B, n), -1, dtype=torch.long)
    matched[:, 1:] = torch.arange(n - 1).unsqueeze(0).expand(B, -1)
    return X, Y, matched


def induction_batch(B: int, n: int, V: int, gen: torch.Generator):
    """In-context induction: each sequence walks a private random follower map.

    ``target[i]`` is the follower of ``token[i]`` under that sequence's map; a
    position is *predictable* once its token has occurred before (induction can
    recall the follower).  Returns (X, Y, predictable_mask).
    """
    X = torch.zeros(B, n, dtype=torch.long)
    Y = torch.full((B, n), IGNORE_INDEX, dtype=torch.long)
    predictable = torch.zeros(B, n, dtype=torch.bool)
    for b in range(B):
        follower = torch.randint(0, V, (V,), generator=gen)
        cur = int(torch.randint(0, V, (1,), generator=gen))
        seen: set = set()
        for i in range(n):
            X[b, i] = cur
            Y[b, i] = int(follower[cur])
            if cur in seen:
                predictable[b, i] = True
            seen.add(cur)
            cur = int(follower[cur])
    return X, Y, predictable


def copy_shift_batch(B: int, n: int, V: int, gen: torch.Generator, *, shift: int = 1):
    """``target[i] = token[i-shift]`` (fixed offset).  Returns (X, Y, matched)."""
    X = torch.randint(0, V, (B, n), generator=gen)
    Y = torch.full((B, n), IGNORE_INDEX, dtype=torch.long)
    Y[:, shift:] = X[:, :-shift]
    matched = torch.full((B, n), -1, dtype=torch.long)
    matched[:, shift:] = torch.arange(n - shift).unsqueeze(0).expand(B, -1)
    return X, Y, matched


def masked_accuracy(pred: torch.Tensor, target: torch.Tensor,
                    extra_mask: Optional[torch.Tensor] = None) -> float:
    mask = target != IGNORE_INDEX
    if extra_mask is not None:
        mask = mask & extra_mask
    if not bool(mask.any()):
        return 0.0
    return float((pred[mask] == target[mask]).float().mean())


# ── binary-model factories (verified-learnable recipe) ──────────────────────────

def retrieval_config(V: int, *, D: int = 128, n_heads: int = 1, n_layers: int = 1,
                     attn_mode: str = "hardmax", seed: int = 0) -> TransformerConfig:
    """A config that learns recency copy cleanly end-to-end.

    Raw-concept-copy attention (``value_proj=False``) so the retrieved value is a
    clean codeword; strict-causal + a recency-dominant ALiBi slope so attention
    selects the previous token; the residual gate initialised mostly open (no
    cold-start transient on a clean transport); and gentle BEP updates so the
    lm_head/codebook co-adaptation converges instead of running away.
    """
    return TransformerConfig(
        vocab_size=V, D=D, n_layers=n_layers, n_heads=n_heads, d_ff=2 * D,
        attn_mode=attn_mode, residual_mode="mux", value_proj=False,
        causal_strict=True, alibi=True,
        alibi_slopes_override=tuple([2 * (D // n_heads) + 1] * n_heads),
        gate_open=0.9, r=0.04, max_trigger_rate=0.08, lm_head_warmup_steps=30,
        update_clip=3, block_init_inertia=8, block_update_clip=1,
        self_supervised_induction=False, margin_weight=0.0, seed=seed)


def train_prev_token(model: BinaryTransformerLM, *, steps: int = 600, B: int = 16,
                     n: int = 16, V: int = 16, seed: int = 1) -> float:
    """Train ``model`` on previous-token copy; return best masked accuracy."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = BepOptimizer(model.parameters(), BepConfig())
    best = 0.0
    for step in range(1, steps + 1):
        X, Y, _ = prev_token_batch(B, n, V, g)
        model.loss_and_backward(model.forward(X), Y)
        opt.step()
        if step % 100 == 0 or step == steps:
            Xe, Ye, _ = prev_token_batch(B, n, V, g)
            pred = model.forward(Xe).reshape(B, n, V).argmax(-1)
            best = max(best, masked_accuracy(pred, Ye))
    return best


# ── float transformer baseline ──────────────────────────────────────────────────

def _alibi_slopes_float(n_heads: int) -> torch.Tensor:
    # Standard geometric ALiBi slopes (2^{-8k/H}); recency strength per head.
    start = 2.0 ** (-8.0 / n_heads)
    return torch.tensor([start ** (k + 1) for k in range(n_heads)])


class TransformerBaseline(nn.Module):
    """A minimal float decoder-only transformer for comparison.

    Position-free token embeddings + ALiBi relative bias + causal mask + one or
    more standard multi-head attention / MLP blocks + a tied unembedding.  Mirrors
    the binary model's information flow (no absolute positions; recency via
    ALiBi), so accuracy on the synthetic tasks is an apples-to-apples reference.
    """

    def __init__(self, vocab: int, *, d_model: int = 64, n_heads: int = 2,
                 n_layers: int = 1, d_ff: int = 128, causal_strict: bool = False,
                 alibi_recency: bool = False):
        super().__init__()
        self.vocab, self.d_model, self.n_heads = vocab, d_model, n_heads
        self.d_h = d_model // n_heads
        self.causal_strict = causal_strict
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "qkv": nn.Linear(d_model, 3 * d_model, bias=False),
                "proj": nn.Linear(d_model, d_model, bias=False),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
                "fc1": nn.Linear(d_model, d_ff),
                "fc2": nn.Linear(d_ff, d_model),
            }) for _ in range(n_layers)
        ])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        slopes = _alibi_slopes_float(n_heads)
        if alibi_recency:                       # force a strong recency head
            slopes = slopes.clamp_min(0.0)
            slopes[-1] = float(d_model)         # last head ≫ content → attends nearest
        self.register_buffer("slopes", slopes)

    def _bias(self, n: int, device) -> torch.Tensor:
        i = torch.arange(n, device=device).unsqueeze(1)
        j = torch.arange(n, device=device).unsqueeze(0)
        dist = (i - j).float()                                  # (n,n)
        bias = -self.slopes.view(self.n_heads, 1, 1) * dist     # (H,n,n)
        keep = (j < i) if self.causal_strict else (j <= i)
        # Large *finite* negative (not -inf): -inf produces NaN gradients through
        # softmax for a fully-masked row (strict-causal row 0).  e^{-1e9} still
        # underflows to 0, so masked weights are ~0 and grads stay finite.
        bias = bias.masked_fill(~keep, -1e9)
        return bias

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, n = ids.shape
        x = self.emb(ids)
        bias = self._bias(n, ids.device)
        # Rows with no valid key (strict-causal row 0) attend to nothing — zero
        # their attention so they contribute no context (and don't softmax to a
        # spurious uniform distribution over masked positions).
        row_valid = (bias > -1e8).any(dim=-1, keepdim=True).float()   # (H,n,1)
        H, dh = self.n_heads, self.d_h
        for blk in self.blocks:
            qkv = blk["qkv"](blk["ln1"](x)).view(B, n, 3, H, dh)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B,n,H,dh)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) / (dh ** 0.5) + bias.unsqueeze(0)
            att = att.softmax(dim=-1) * row_valid.unsqueeze(0)   # zero fully-masked rows
            o = (att @ v).transpose(1, 2).reshape(B, n, H * dh)
            x = x + blk["proj"](o)
            x = x + blk["fc2"](F.gelu(blk["fc1"](blk["ln2"](x))))
        return self.head(self.lnf(x))


def train_baseline(model: TransformerBaseline, task_fn, *, steps: int = 400,
                   B: int = 16, n: int = 16, V: int = 16, lr: float = 3e-3,
                   seed: int = 1) -> float:
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    for step in range(1, steps + 1):
        out = task_fn(B, n, V, g)
        X, Y = out[0], out[1]
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1),
                               ignore_index=IGNORE_INDEX)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                Xe = task_fn(B, n, V, g)
                pred = model(Xe[0]).argmax(-1)
                best = max(best, masked_accuracy(pred, Xe[1]))
    return best
