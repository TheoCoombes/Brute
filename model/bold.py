"""BOLD optimizer — native Boolean training by bit-flipping (no FP latent weights).

Implements the learning machinery of *Boolean Logic Deep Learning* (Nguyen et
al., NeurIPS 2024) as summarised in HÆMMR §7:

    * Boolean variation  δ(a→b) ∈ {T, 0, F}, embedded e(T)=+1, e(0)=0, e(F)=-1.
    * The per-weight optimisation signal ``q`` is the ternary logic projection
      of aggregated Boolean variation: positive/T, zero/no variation, negative/F.
    * Flip rule (Eq. 9):     w  ←  ¬w   iff   xnor(q, w) = T
      i.e. flip the bit when the signal *agrees in sign* with the current
      bipolar weight value.  No gradient, no learning-rate-scaled subtraction —
      a logic test and a bit flip.
    * Accumulator + auto-regularising plasticity (Eqs. 10-11)::
          m ← β·m + η·q                         (per-weight accumulator)
          if xnor(m, w) = T:  flip w, reset m=0  (error feedback)
          β  =  (#unchanged weights) / (#weights)

The weight itself is **stored as ``brute.bit1``** (the canonical parameter).
``m`` is integer optimiser state and resets on every flip. ``q`` is a transient
integer signal buffer cleared after each optimizer step; it is not a latent
parameter and never drives a gradient-descent update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

import brute

from vsa import to_pm1


class BoldParam:
    """A single Boolean parameter trained by the BOLD flip rule.

    Parameters
    ----------
    bit : brute.Tensor (bit1)
        The packed 1-bit weight tensor — the canonical stored parameter.
    name : str
        Identifier (used for checkpointing / debugging).
    """

    __slots__ = (
        "name", "bit", "_pm1", "m", "q", "beta", "beta_num", "beta_den",
        "shape", "device", "flip_scale",
    )

    def __init__(self, bit: brute.Tensor, name: str = "", flip_scale: float = 1.0):
        assert getattr(bit, "_is_bit1", False), "BoldParam requires a bit1 tensor"
        self.name = name
        self.bit = bit
        self.shape = tuple(bit.shape)
        self.device = bit.device
        self._pm1: Optional[torch.Tensor] = None         # lazy ±1 cache
        # Integer per-weight accumulator (BOLD Eq. 10). int16 keeps the update
        # fixed-point and compact while leaving headroom for eta-scaled counts.
        self.m = torch.zeros(self.shape, dtype=torch.int16, device=bit.device)
        # Transient ternary Boolean-variation signal (BOLD Eq. 7 projected by
        # Def. A.1). It is cleared after every optimizer step.
        self.q = torch.zeros(self.shape, dtype=torch.int16, device=bit.device)
        self.beta = 1.0
        self.beta_num = 1
        self.beta_den = 1
        # Per-parameter flip-rate multiplier — the tied codebook is the fragile
        # representation, so it needs more accumulated evidence (higher effective
        # threshold) before it flips.
        self.flip_scale = flip_scale

    # ── views ────────────────────────────────────────────────────────────────
    @property
    def pm1(self) -> torch.Tensor:
        """Cached ±1 float view of the bit weight (recomputed after a flip)."""
        if self._pm1 is None:
            self._pm1 = to_pm1(self.bit)
        return self._pm1

    # ── signal accumulation (called by layer backward) ───────────────────────
    def add_signal(self, q_delta: torch.Tensor) -> None:
        """Accumulate a Boolean variation contribution ``q``.

        Inputs are projected to their logic value {-1, 0, +1}; the accumulator
        ``m`` carries persistence over training iterations.
        """
        if q_delta.shape != self.q.shape:
            q_delta = q_delta.reshape(self.q.shape)
        q_i = torch.sign(q_delta).to(self.q.dtype)
        self.q.add_(q_i)

    def zero_signal(self) -> None:
        self.q.zero_()

    # ── flip step (BOLD Eqs. 9-11, integer accumulator) ──────────────────────
    @torch.no_grad()
    def apply(self, eta: float, threshold: float = 1.0, m_clip: int = 127) -> int:
        """Apply one BOLD update step.  Returns the number of bits flipped.

        ``m ← β·m + η·q`` is performed as integer fixed-point arithmetic. A bit
        flips when the updated accumulator agrees with the current Boolean
        weight and crosses ``threshold`` (Eq. 9 / Algorithm 1), then that
        accumulator entry is reset to zero. β is the exact unchanged/total
        fraction from the previous step (Eq. 11).
        """
        eta_i = max(0, int(round(float(eta))))
        eff_threshold = max(1, int(round(float(threshold) / max(self.flip_scale, 1e-6))))
        bit, m, n_flip_t = brute.fast.bold_update_packed(
            self.bit, self.m, self.q, self.beta_num, self.beta_den,
            eta_i, eff_threshold, int(m_clip))
        n_flip = int(n_flip_t.item())
        self.bit = bit
        self.m = m
        self._pm1 = None
        n_tot = self.m.numel()
        self.beta_num = n_tot - n_flip
        self.beta_den = max(n_tot, 1)
        self.beta = self.beta_num / self.beta_den                  # Eq. 11
        self.q.zero_()
        return n_flip

    # ── checkpoint ────────────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "packed": self.bit._packed_buf.detach().cpu().clone(),
            "shape": self.shape,
            "m": self.m.detach().cpu().clone(),
            "beta": self.beta,
            "beta_num": self.beta_num,
            "beta_den": self.beta_den,
        }

    def load_state_dict(self, sd: dict) -> None:
        from brute.tensor import Tensor as _BT
        packed = sd["packed"].to(self.device)
        self.bit = _BT._make_bit1_from_packed(packed, list(sd["shape"]))
        self._pm1 = None
        self.m = sd["m"].to(self.device)
        self.beta = float(sd["beta"])
        self.beta_num = int(sd.get("beta_num", round(self.beta * self.m.numel())))
        self.beta_den = int(sd.get("beta_den", max(self.m.numel(), 1)))


@dataclass
class BoldConfig:
    eta: float = 3.0                # integer-rounded accumulation factor η
    eta_decay: float = 1.0          # multiplicative decay per optimiser step
    eta_min: float = 1.0
    threshold: float = 6.0          # integer evidence needed to flip a bit
    m_clip: int = 127               # accumulator saturation (±127 by default)


class BoldOptimizer:
    """Owns a collection of :class:`BoldParam` and applies the flip step to all."""

    def __init__(self, params: Iterable[BoldParam], config: Optional[BoldConfig] = None):
        self.params: List[BoldParam] = list(params)
        self.config = config or BoldConfig()
        self.eta = self.config.eta
        self._step = 0

    def zero_signals(self) -> None:
        for p in self.params:
            p.zero_signal()

    @torch.no_grad()
    def step(self) -> dict:
        n_flip = 0
        n_bits = 0
        for p in self.params:
            n_flip += p.apply(self.eta, self.config.threshold, self.config.m_clip)
            n_bits += p.m.numel()
        self._step += 1
        self.eta = max(self.config.eta_min, self.eta * self.config.eta_decay)
        return {"n_flip": n_flip, "n_bits": n_bits,
                "flip_frac": n_flip / max(n_bits, 1), "eta": self.eta}

    # ── checkpoint ────────────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {"eta": self.eta, "step": self._step,
                "params": {p.name: p.state_dict() for p in self.params}}

    def load_state_dict(self, sd: dict) -> None:
        self.eta = sd.get("eta", self.eta)
        self._step = sd.get("step", 0)
        byname = {p.name: p for p in self.params}
        for name, psd in sd["params"].items():
            if name in byname:
                byname[name].load_state_dict(psd)


# ── helpers ───────────────────────────────────────────────────────────────────

def random_bit_param(shape, name: str, *, generator: Optional[torch.Generator] = None,
                     device=None, p_true: float = 0.5) -> BoldParam:
    """A :class:`BoldParam` initialised with uniform-random bits (≈ ±1 balanced)."""
    # Draw on CPU (the seeded generator lives on CPU) then move to the device.
    bits = (torch.rand(shape, generator=generator) < p_true)
    bit = brute.as_tensor(bits, dtype=brute.bit1)
    if device is not None:
        bit = bit.to(device)
    return BoldParam(bit, name=name)


def signal_scale(fan_out: int) -> float:
    """Variance-matching backward scale ``sqrt(2 / fan_out)`` (BOLD §3.3)."""
    return (2.0 / max(int(fan_out), 1)) ** 0.5
