"""Boolean optimizer (BOLD).

Reference
---------
Van Minh Nguyen, Cristian Ocampo, Aymen Askri, Louis Leconte, Ba-Hien Tran.
"BOLD: Boolean Logic Deep Learning." NeurIPS 2024.
- Algorithm 1 (FC training loop)
- Algorithm 8 (Python reference implementation of the Boolean optimizer)
- §3.3 Eqs. 9–11 (flip rule + β plasticity)

Design
------
A drop-in ``torch.optim.Optimizer`` for parameters that live in the Boolean
domain ``{0, 1}`` (typically the weight / bias of :class:`BitLinear`
or :class:`BitConv2d`). The update is BOLD's deterministic
single-bit flip rule with per-weight accumulation and a per-layer plasticity
factor ``β``.

Per parameter, we maintain:

  * ``accums``: real-valued accumulator ``m_{i,j}^{l,t}`` (Eq. 10).
  * ``ratios``: ``β_{l,t}`` from Eq. 11 — the fraction of weights NOT
    flipped at the previous step. Damps the accumulator of layers that have
    been thrashing.

Per step (Algorithm 8):

  1. ``m  := β · m + η · q``        (``q = param.grad.data`` — the
     aggregated Boolean variation produced by the layer's backward path).
  2. ``flip_mask := m · (2W − 1) ≥ 1``  (Eq. 9 in ``{0,1}`` form: equivalent
     to ``xnor(sign(m), W) = T`` with the |m| ≥ 1 confidence threshold).
  3. ``W := ¬W`` on ``flip_mask``; reset ``m`` to 0 at those entries.
  4. Update ``β := 1 − mean(flip_mask)``.

Use this optimizer for the *Boolean* parameters only. Full-precision
siblings (e.g. the first / last layer the BOLD paper keeps in FP) should
still be driven by a standard optimizer (Adam is the paper's choice).
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor


__all__ = ["BooleanOptimizer", "iter_boolean_parameters", "split_parameters"]


class BooleanOptimizer(torch.optim.Optimizer):
    """BOLD Boolean optimizer (Algorithm 8).

    Args:
        params: iterable of Boolean parameters (``{0, 1}``-valued floats).
            Typically obtained via :func:`iter_boolean_parameters` or
            :func:`split_parameters`.
        lr: learning rate ``η`` (Algorithm 1 / Eq. 10). The BOLD paper uses
            ``η = 150`` for VGG-SMALL without BN and ``η = 12`` with BN.

    Properties:
        nb_flips: number of bit flips since the last read, then auto-resets
            (matches the BOLD reference snippet). Useful for logging
            per-step flip rates.
    """

    def __init__(self, params: Iterable[Tensor], lr: float):
        if lr <= 0:
            raise ValueError(f"lr must be > 0; got {lr}")
        super().__init__(params, dict(lr=lr))
        for group in self.param_groups:
            group["accums"] = [torch.zeros_like(p.data) for p in group["params"]]
            group["ratios"] = [0.0 for _ in group["params"]]
        self._nb_flips = 0

    @property
    def nb_flips(self) -> int:
        """Number of bit flips since the last read. Auto-resets on read."""
        n = self._nb_flips
        self._nb_flips = 0
        return n

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                self._update(p, group, idx, lr)
        return loss

    def _update(self, param: Tensor, group: dict, idx: int, lr: float) -> None:
        accum = group["ratios"][idx] * group["accums"][idx] + lr * param.grad.data
        group["accums"][idx] = accum

        # Flip rule (Eq. 9, expressed in {0,1} embedding):
        #   xnor(q_logic, w_logic) = T  ⇔  accum · (2W − 1) ≥ 1
        # The strict ≥ 1 threshold is the deterministic confidence gate from
        # Algorithm 8; weaker accumulators stay pending for future steps.
        param_to_flip = accum * (2.0 * param.data - 1.0) >= 1.0
        if param_to_flip.any():
            param.data[param_to_flip] = 1.0 - param.data[param_to_flip]
            group["accums"][idx][param_to_flip] = 0.0
        # β_t+1 = N_unchanged / N_tot — per-layer plasticity factor (Eq. 11).
        group["ratios"][idx] = float(1.0 - param_to_flip.to(torch.float32).mean().item())
        self._nb_flips += int(param_to_flip.to(torch.int64).sum().item())


# Convenience: parameter splitting helpers
#
# Models built from :class:`BitLinear` / :class:`BitConv2d` carry a
# ``_bold_boolean`` attribute on their boolean Parameters. This makes it
# trivial to split a model's parameters into "boolean" and "real" groups
# without hard-coding layer names.

def iter_boolean_parameters(model: torch.nn.Module) -> Iterable[Tensor]:
    """Yield parameters tagged ``_bold_boolean = True``."""
    for p in model.parameters():
        if getattr(p, "_bold_boolean", False):
            yield p


def split_parameters(model: torch.nn.Module):
    """Return ``(boolean_params, real_params)`` lists for a model.

    Boolean parameters are the ones tagged by :class:`BitLinear`
    / :class:`BitConv2d`; real parameters are everything else
    (including BatchNorm scales / biases, FP first / last layers, etc.).
    """
    boolean, real = [], []
    for p in model.parameters():
        (boolean if getattr(p, "_bold_boolean", False) else real).append(p)
    return boolean, real
