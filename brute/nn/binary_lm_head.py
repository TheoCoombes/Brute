"""LM head tied to a ``BinaryEmbedding``. Pure binary — no FP/int8 bias.

For each predicting position ``i``:

    logits[v] = popcount(XNOR(E[v, :], h_i)) - d/2

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
    """Tied LM head: shares ``embedding.weight``. No bias.

    Args:
      embedding: The ``BinaryEmbedding`` whose ``weight`` we tie to.
      device: Device.
    """

    def __init__(
        self,
        embedding: BinaryEmbedding,
        device=None,
        **_legacy_kwargs,
    ):
        """``_legacy_kwargs`` swallows ``bias_init`` / ``bias_clip`` from older
        callers; both are ignored since the bias has been removed."""
        super().__init__()
        self.embedding = embedding
        self.vocab_size = embedding.num_embeddings
        self.dim = embedding.dim

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

    Sign convention (DESCENT direction, not gradient):
      err[target]          = +clip(max_margin, 1, err_clip)  if beaten, else 0
      err[v ∈ distractors] = -clip(margin[v], 1, err_clip)
      err[everyone else]   = 0

    With this convention, a positive ``err[v]`` means "push logit[v] up",
    which corresponds to making ``E[v]`` more agree-aligned with ``h``.
    Vote ``sum_b err_b · x_b`` then has sign matching the desired weight
    direction; descent flip is ``W ⊕ (sign(vote) ≠ sign(W))``.

    Returns int8 (M, V) tensor.
    """
    m, v = logits.shape
    err = brute.zeros((m, v), dtype=brute.int8, device=logits.device)
    target_idx = targets.unsqueeze(1)                       # (M, 1) long
    target_logit = logits.gather(1, target_idx)             # (M, 1) int32
    margin = logits - target_logit                          # (M, V) int32

    # Suppress the target column from the distractor pool.
    margin.scatter_(1, target_idx, -(1 << 30))

    k = min(k_distractors, v - 1)
    topk_vals, topk_idx = margin.topk(k, dim=1)
    valid = topk_vals >= 0                                   # (M, k) bool

    mag = topk_vals.clamp(min=1, max=err_clip).to(brute.int8)
    neg_err = (-mag).masked_fill(~valid, 0)
    err.scatter_(1, topk_idx, neg_err)

    any_beat = valid.any(dim=1)                              # (M,) bool
    max_margin = topk_vals.max(dim=1).values.clamp(min=1, max=err_clip).to(brute.int8)
    pos_err = max_margin.masked_fill(~any_beat, 0)
    err.scatter_(1, target_idx, pos_err.unsqueeze(1))
    return err


def lm_head_bias_step(*args, **kwargs) -> int:
    """No-op shim retained for backward-compat with older training loops.

    The INT8 LM-head bias has been removed (pure binary LM head). Callers
    can keep invoking this function; it simply returns 0.
    """
    return 0


__all__ = ["BinaryLMHead", "signed_int_error", "lm_head_bias_step"]
