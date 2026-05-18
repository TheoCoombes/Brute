"""Bit-balanced binarization with optional near-boundary gate flag.

`bit_balance` replaces a plain `sign()` at zero with `sign(x - m)` where
``m`` is the median over the feature axis. The result has exactly half +1
and half -1 elements per token (modulo one element for even ``d``),
recovering the ±1 balance that an ordinary LayerNorm preserves in
expectation. No learned parameters.

This module also produces a per-element bit1 "gate" flag — set when the
centered pre-activation magnitude is below ``nu * d``. The flag marks
neurons whose sign was decided on weak evidence; the BGPT-1 backward pass
masks error contributions through these neurons (BEP gating).
"""

from __future__ import annotations

from typing import Optional, Tuple

import brute
from brute import nn
from brute.tensor import Tensor


def bit_balance(
    pre: Tensor,
    nu: float = 0.0,
    dim: int = -1,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Binarize ``pre`` along ``dim`` after subtracting the per-token median.

    Args:
      pre: int / float ``brute.Tensor`` with shape ``(..., d)`` along ``dim``.
      nu: If > 0, also return a per-element gate mask where
        ``|pre - median| <= nu * d``. If 0, the second return value is None.
      dim: Feature axis to normalize along (default last).

    Returns:
      ``(out_bit1, gate_or_none)`` where ``out_bit1`` is a packed bit1
      ``brute.Tensor`` carrying ±1 values (True ↔ +1) and ``gate_or_none``
      is a packed bit1 tensor (True ↔ near-boundary) when ``nu > 0``,
      else ``None``.
    """
    m = brute.median(pre, dim=dim, keepdim=True).values
    centered = pre - m
    out_bool = centered > 0
    out_bit1 = brute.as_tensor(out_bool, dtype=brute.bit1, device=pre.device)
    if nu > 0.0:
        d_along = pre.shape[dim]
        thresh = int(nu * d_along)
        gate_bool = centered.abs() <= thresh
        gate_bit1 = brute.as_tensor(gate_bool, dtype=brute.bit1, device=pre.device)
        return out_bit1, gate_bit1
    return out_bit1, None


class BitBalancedNorm(nn.Module):
    """Module wrapper around :func:`bit_balance`. Stateless."""

    def __init__(self, nu: float = 0.05, dim: int = -1):
        super().__init__()
        self.nu = float(nu)
        self.dim = int(dim)

    def forward(
        self,
        pre: Tensor,
        return_gate: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        eff_nu = self.nu if return_gate else 0.0
        return bit_balance(pre, eff_nu, self.dim)

    def extra_repr(self) -> str:
        return f"nu={self.nu}, dim={self.dim}"


__all__ = ["BitBalancedNorm", "bit_balance"]
