"""Boolean threshold activation (BOLD).

Reference
---------
Van Minh Nguyen, Cristian Ocampo, Aymen Askri, Louis Leconte, Ba-Hien Tran.
"BOLD: Boolean Logic Deep Learning." NeurIPS 2024.
- §3.1, "Forward Activation"
- Appendix C.3 (backprop through the Boolean activation, ``tanh'`` STE)

The only natural Boolean activation in BOLD is the threshold:

    y = T  if  s ≥ τ      (else F)

Forward returns a float ``{0, 1}`` tensor (1.0 for T, 0.0 for F).
Backward is the straight-through estimator weighted by ``tanh'(α·Δ)`` with
``Δ = |s − τ|`` and ``α = π / (2·sqrt(3m))`` (Eq. 47), where ``m`` is the
input range of the pre-activation distribution — typically the receptive
field size, ``C_in · k_h · k_w`` for a conv layer or ``in_features`` for a
linear layer.

Choosing ``α`` per Eq. 47 makes the STE's effective slope distribution match
the pre-activation's spread, which (the paper shows in Figure 5) keeps the
gradient signal well-scaled across reasonable layer sizes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, autograd, nn


__all__ = ["BitActivation"]


class _BitActivationFunction(autograd.Function):
    @staticmethod
    def forward(ctx, X: Tensor, threshold: float, alpha: float) -> Tensor:
        ctx.save_for_backward(X)
        ctx.threshold = float(threshold)
        ctx.alpha = float(alpha)
        return (X >= threshold).to(X.dtype)

    @staticmethod
    def backward(ctx, Z: Tensor):
        (X,) = ctx.saved_tensors
        delta = X - ctx.threshold
        # tanh'(α·Δ) = 1 - tanh²(α·Δ). The further |Δ| is from 0, the smaller
        # the gradient — a smooth STE saturating in the "decided" region.
        g = 1.0 - torch.tanh(ctx.alpha * delta).pow(2)
        return Z * g, None, None


class BitActivation(nn.Module):
    """Threshold-step Boolean activation with a smooth STE backward.

    Args:
        threshold: ``τ`` — pre-activation threshold. Default ``0.0`` to
            match the 0-centred output of :class:`BitLinear` / :class:`BitConv2d`.
        m: pre-activation range used in the BOLD α formula
            ``α = π / (2·sqrt(3m))`` (Eq. 47). For a Boolean conv layer this
            is ``C_in · k_h · k_w``; for a Boolean linear layer it is
            ``in_features``. If ``None``, ``α`` defaults to ``1.0``
            (vanilla tanh' STE).
        alpha: directly override ``α``. Takes precedence over ``m``.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        m: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        self.threshold = float(threshold)
        if alpha is not None:
            self.alpha = float(alpha)
        elif m is not None:
            self.alpha = math.pi / (2.0 * math.sqrt(3.0 * m))
        else:
            self.alpha = 1.0

    def forward(self, X: Tensor) -> Tensor:
        return _BitActivationFunction.apply(X, self.threshold, self.alpha)

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, alpha={self.alpha:.4g}"
