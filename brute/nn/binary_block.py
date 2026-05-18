"""Binary transformer block with parallel residual and stochastic depth.

Per BGPT-1 §3.6 the block is:

    attn_out = attention(x)
    ffn_out  = ffn(x)
    summed   = x + attn_out + ffn_out        # ∈ {-3,-1,+1,+3}
    out      = sign(summed)

The three terms are independent ±1; their sum is odd-parity so the
``sign()`` rebinarisation is unambiguous (no ties).

LayerDrop: with probability ``p_drop`` (and only during training), both
sub-blocks are skipped and the block returns ``x`` unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import brute
from brute import nn
from brute.tensor import Tensor
from brute.nn.binary_attention import BinaryAttention
from brute.nn.binary_ffn import BinaryFFN
from brute.nn.binary_norm import bit_balance


def _majority3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """``sign(a + b + c)`` for ±1 inputs, computed via packed bit1 majority."""
    ab = brute.bitwise_and(a, b)
    ac = brute.bitwise_and(a, c)
    bc = brute.bitwise_and(b, c)
    return brute.bitwise_or(brute.bitwise_or(ab, ac), bc)


class BinaryTransformerBlock(nn.Module):
    """Single transformer block — parallel residual, bit1 throughout.

    Args:
      dim: Model dim.
      n_heads: Number of attention heads.
      max_context: Context-length cap.
      hidden_mult: FFN expansion factor (default 4).
      nu: Gate threshold for the inner layers' bit-balance.
      p_drop: LayerDrop probability for this block. 0 disables LayerDrop.
      device: Device for the parameter buffers.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_context: int,
        hidden_mult: int = 4,
        nu: float = 0.05,
        p_drop: float = 0.0,
        device=None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.p_drop = float(p_drop)
        self.attention = BinaryAttention(
            dim, n_heads, max_context, device=device,
        )
        self.ffn = BinaryFFN(
            dim, hidden_mult=hidden_mult, nu=nu, device=device,
        )

    def forward(
        self,
        x: Tensor,
        return_diag: bool = False,
        return_tape: bool = False,
        skip_layer: Optional[bool] = None,
    ) -> Tuple[Tensor, Optional[Dict], Optional[Dict]]:
        """Run the block.

        Args:
          x: bit1 brute.Tensor of shape ``(B, C, dim)``.
          return_diag: If True, return the attention diagnostics dict.
          return_tape: If True, return the FFN training tape (gates, pres).
          skip_layer: If supplied (e.g. by the caller's RNG), force-skip
            this block. If None, sample Bernoulli(p_drop) when training and
            never skip during eval.

        Returns:
          ``(out, diag, tape)``: out is bit1 (B, C, dim); diag and tape are
          None when not requested.
        """
        if skip_layer is None:
            do_skip = (self.training and self.p_drop > 0.0
                       and brute.rand((), device=x.device).item() < self.p_drop)
        else:
            do_skip = bool(skip_layer)
        if do_skip:
            return x, None, None

        attn_out, diag, _ = self.attention(x, return_diag=return_diag)
        ffn_out, tape = self.ffn(x, return_tape=return_tape)

        # Parallel residual: sign(x + attn + ffn) ≡ majority-of-3 on bit1.
        # Each term is ±1, sum ∈ {-3,..,+3} odd-parity, sign unambiguous.
        out = _majority3(x, attn_out, ffn_out)

        return out, diag, tape


__all__ = ["BinaryTransformerBlock"]
