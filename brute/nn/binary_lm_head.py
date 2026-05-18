"""LM head tied to a ``BinaryEmbedding`` plus a learned INT8 per-token bias.

For each predicting position ``i``:

    logits[v] = popcount(XNOR(E[v, :], h_i)) - d/2  +  b[v]

The bias is the only non-binary trainable parameter in the model — int8
with a small bounded-accumulator update rule (BOLD-style) that ticks
``b[v]`` by ±1 each time the accumulator crosses ±threshold.

The signed-integer output error signal (``signed_int_error``) is also
exposed here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import brute
from brute import nn
from brute.tensor import Tensor
from brute.nn.binary_embedding import BinaryEmbedding


class BinaryLMHead(nn.Module):
    """Tied LM head: shares ``embedding.weight`` with a per-token INT8 bias.

    Args:
      embedding: The ``BinaryEmbedding`` whose ``weight`` we tie to.
      bias_init: Initial value for every token bias (default 0).
      bias_clip: Range bound for the int8 bias (default ±127).
      device: Device.
    """

    def __init__(
        self,
        embedding: BinaryEmbedding,
        bias_init: int = 0,
        bias_clip: int = 127,
        device=None,
    ):
        super().__init__()
        self.embedding = embedding
        self.vocab_size = embedding.num_embeddings
        self.dim = embedding.dim
        self.bias_clip = int(bias_clip)
        bias = brute.full((self.vocab_size,), int(bias_init), dtype=brute.int8, device=device)
        accumulator = brute.zeros(self.vocab_size, dtype=brute.int32, device=device)
        self.register_buffer("bias", bias)
        self.register_buffer("bias_accumulator", accumulator)

    def forward(self, h: Tensor) -> Tensor:
        """Compute integer logits for a bit1 hidden state.

        Args:
          h: bit1 brute.Tensor of shape ``(..., dim)``.

        Returns:
          int32 brute.Tensor of shape ``(..., vocab_size)``.
        """
        leading = list(h.shape[:-1])
        h_flat = h.reshape(-1, self.dim)
        # bit1 matmul convention: h @ E.weight -> (M, V) int32.
        logits = h_flat @ self.embedding.weight
        logits = logits + self.bias.to(logits.dtype)
        return logits.reshape(*leading, self.vocab_size)


def signed_int_error(
    logits: Tensor,     # (M, V) int32
    targets: Tensor,    # (M,) long
    k_distractors: int = 8,
    err_clip: int = 7,
) -> Tensor:
    """Compute the BGPT-1 sparse signed-integer error signal.

    For each position, gather up to ``k_distractors`` non-target tokens
    whose integer logit ≥ logit[target]. Magnitude = ``min(err_clip,
    |logits[v] - logits[target]|)``.

    Sign convention:
      err[target] = +clip(max_margin, 1, err_clip)  if any distractor beats
        the target's logit, else 0.
      err[v ∈ distractors] = -clip(margin[v], 1, err_clip)
      err[everyone else]   = 0

    Returns int8 (M, V) tensor — written in place from two small scatter
    operations; no full (M, V) bool target mask is allocated.
    """
    m, v = logits.shape
    err = brute.zeros((m, v), dtype=brute.int8, device=logits.device)
    target_idx = targets.unsqueeze(1)                       # (M, 1) long
    target_logit = logits.gather(1, target_idx)             # (M, 1) int32
    margin = logits - target_logit                          # (M, V) int32

    # Suppress the target column from the distractor pool by writing a
    # large negative into it in-place — avoids a full (M, V) bool
    # target_mask. Negative-margin distractors stay negative; the post-
    # topk ``valid = topk_vals >= 0`` filters them.
    margin.scatter_(1, target_idx, -(1 << 30))

    k = min(k_distractors, v - 1)
    topk_vals, topk_idx = margin.topk(k, dim=1)
    valid = topk_vals >= 0                                   # (M, k) bool — small

    mag = topk_vals.clamp(min=1, max=err_clip).to(brute.int8)
    neg_err = (-mag).masked_fill(~valid, 0)
    err.scatter_(1, topk_idx, neg_err)

    any_beat = valid.any(dim=1)                              # (M,) bool
    max_margin = topk_vals.max(dim=1).values.clamp(min=1, max=err_clip).to(brute.int8)
    pos_err = max_margin.masked_fill(~any_beat, 0)
    err.scatter_(1, target_idx, pos_err.unsqueeze(1))
    return err


def lm_head_bias_step(
    bias: Tensor,        # (V,) int8 — modified in place
    accumulator: Tensor, # (V,) int32 — modified in place
    err: Tensor,         # (M, V) int8 — output of `signed_int_error`
    accum_threshold: int = 32,
    bias_clip: int = 127,
) -> int:
    """Bounded-accumulator update for the INT8 LM-head bias.

    ``err`` follows the ``signed_int_error`` convention:
      err[v] > 0  →  logit[v] is too low; pushing bias[v] up helps.
      err[v] < 0  →  logit[v] is too high; pushing bias[v] down helps.

    So accumulator > +threshold  →  bias[v] += 1, and accumulator <
    -threshold → bias[v] -= 1. Frequent targets accumulate positive err
    and slowly grow positive bias (a token-frequency prior).
    """
    if err.numel() == 0:
        return 0
    delta = err.to(brute.int32).sum(dim=0)
    accumulator.add_(delta)
    pos_mask = accumulator >=  accum_threshold
    neg_mask = accumulator <= -accum_threshold
    n_updates = int(pos_mask.sum().item()) + int(neg_mask.sum().item())
    bias_i32 = bias.to(brute.int32)
    bias_i32[pos_mask] = (bias_i32[pos_mask] + 1).clamp(-bias_clip, bias_clip)
    bias_i32[neg_mask] = (bias_i32[neg_mask] - 1).clamp(-bias_clip, bias_clip)
    bias.copy_(bias_i32.to(brute.int8))
    accumulator[pos_mask] -= accum_threshold
    accumulator[neg_mask] += accum_threshold
    return n_updates


__all__ = ["BinaryLMHead", "signed_int_error", "lm_head_bias_step"]
