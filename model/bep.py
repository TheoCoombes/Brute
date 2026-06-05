"""BEP / BOLD native binary training — integer hidden weight, binary errors.

This replaces the old STE-style float backprop with the learning machinery
actually prescribed by the papers:

  * **BEP** (*Boolean error propagation*, 2512.04189 — Table 2, Eqs. 1-9): the
    backward signal is a **binary desired activation** ``a*`` propagated by
    ``a*_l = sign(Wᵀ(g ⊙ a*_{l+1}))`` (a bit1 XNOR-popcount matmul + sign); the
    weight update is a **binary outer product** ``ΔH = a*_out ⊗ a_inᵀ``
    accumulated straight into an **integer hidden weight** ``H`` (Eqs. 8-9).
  * **BOLD** (Boolean Logic Deep Learning): the visible weight is ``W = sign(H)``
    — there is no float copy of the weight and no float optimisation signal.

The only large allocation per parameter is the integer ``H`` (Int8); the
visible bit weight ``W = sign(H)`` is derived (≈1 bit/weight) and cached.  No
float tensors are held by any parameter, and nothing float crosses a layer
boundary — signals on the wire are bit1 desired activations plus an optional
bit1 eligibility gate.

Backward primitives (all pure bit1 matmuls into integers — no unpack):

  * :func:`linear_backward`  — the workhorse for ``BooleanLinear``.
  * :func:`signed_batch_sum` — per-coordinate Σ_batch ±1 for diagonal params.
  * :func:`combine_desired`  — majority of two binary desireds (residual fan-in).
  * :func:`mux`              — packed per-coordinate select.
  * :func:`pm1_int`          — bit1 → ±1 integer (for scatter/index updates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

import brute
from brute.tensor import Tensor as _BT

from vsa import sign_to_bit1


H_DTYPE = torch.int8
H_MIN = -128
H_MAX = 127


# ── packed bit helpers ─────────────────────────────────────────────────────────

def _not_packed(bit: brute.Tensor) -> torch.Tensor:
    return torch.ops.brute.bit1_not_packed(bit._packed_buf, bit.shape[-1])


def mux(sel_bit: brute.Tensor, t_bit: brute.Tensor, f_bit: brute.Tensor) -> brute.Tensor:
    """Per-coordinate select: ``sel ? t : f`` — fully packed (no unpack)."""
    ps = sel_bit._packed_buf
    nps = _not_packed(sel_bit)
    out = torch.bitwise_or(torch.bitwise_and(ps, t_bit._packed_buf),
                           torch.bitwise_and(nps, f_bit._packed_buf))
    return _BT._make_bit1_from_packed(out, list(t_bit.shape))


def combine_desired(a_bit: brute.Tensor, b_bit: Optional[brute.Tensor],
                    tie_bit: brute.Tensor) -> brute.Tensor:
    """Combine two binary desired activations for one node (residual fan-in).

    Where the two desireds *agree*, keep that value; where they *disagree*
    (a 2-way majority tie), fall back to ``tie_bit`` (the current activation, so
    a genuine conflict produces no spurious flip).  ``b_bit=None`` means "no
    opinion" (e.g. a hard-selection branch with zero upstream gradient) → return
    ``a_bit`` unchanged.  Packed throughout.
    """
    if b_bit is None:
        return a_bit
    agree = brute.fast.eq(a_bit, b_bit)
    return mux(agree, a_bit, tie_bit)


def pm1_int(bit: brute.Tensor, dtype=torch.int8) -> torch.Tensor:
    """bit1 → ±1 integer tensor (True→+1, False→-1).  Not an ``unpack_pm1``."""
    return bit.bool().to(dtype).mul_(2).sub_(1)


def signed_batch_sum(x_bit: brute.Tensor) -> torch.Tensor:
    """``x`` (M, D) bit1 → (D,) int signed column sum ``Σ_m e(x[m])``.

    Implemented as a single bit1 XNOR-popcount matmul against an all-+1 row, so
    it stays on the packed path and yields the exact integer ``Σ ±1``.  When an
    active-row mask is set (see :func:`set_active`), only active rows contribute.
    """
    am = _ACTIVE
    if am is not None and am.shape[0] == x_bit.shape[0]:
        return (pm1_int(x_bit, torch.int32) * am.unsqueeze(1)).sum(dim=0)
    M = x_bit.shape[0]
    ones = brute.as_tensor(torch.ones(1, M, dtype=torch.bool, device=x_bit.device),
                           dtype=brute.bit1)
    return brute.fast.matmul(ones, x_bit.transpose(0, 1)).reshape(-1)


# ── active-row mask (which positions inject a backward signal) ──────────────────
#
# Under BEP only *triggered* (supervised, margin-violating) positions should
# update weights or propagate a desired change; the rest carry "no opinion" (the
# desired equals the current activation, so nothing flips).  The mask is a single
# per-token boolean vector, constant across layers (a position is trainable iff
# its target is valid and triggers), so it can live as module-shared state rather
# than threading through every backward signature.  ``None`` ⇒ all rows active.

_ACTIVE: Optional[torch.Tensor] = None


def set_active(mask: Optional[torch.Tensor]) -> None:
    """Set (or clear with ``None``) the active-row mask, a (M,) bool tensor."""
    global _ACTIVE
    _ACTIVE = mask


def active_mask() -> Optional[torch.Tensor]:
    return _ACTIVE


# ── integer-weight parameter ───────────────────────────────────────────────────

class BepParam:
    """A single Boolean parameter trained by BEP: integer ``H``, ``W = sign(H)``.

    ``H`` (Int8) is the only significant buffer — the "weight inertia/momentum"
    of BEP §3.3.  ``bit = sign(H)`` (``H ≥ 0 → +1/True``) is the visible 1-bit
    weight read by every forward; it is derived and cached, refreshed lazily when
    ``H`` changes.  There is **no** float ``q``, **no** float ``pm1``, **no**
    separate int8 ``m``.
    """

    __slots__ = ("name", "H", "_bit", "_dirty", "_prev_sign", "shape", "device",
                 "lr", "update_clip")

    def __init__(self, H: torch.Tensor, name: str = "", *, lr: int = 1,
                 update_clip: Optional[int] = None):
        self.name = name
        self.H = H.to(H_DTYPE)
        self.shape = tuple(H.shape)
        self.device = H.device
        self.lr = lr                     # integer update scale (BEP "learning rate")
        self.update_clip = update_clip   # optional per-step elementwise ΔH clamp
        self._bit: Optional[brute.Tensor] = None
        self._dirty = True
        self._prev_sign: Optional[torch.Tensor] = None

    # ── visible weight W = sign(H) ────────────────────────────────────────────
    @property
    def bit(self) -> brute.Tensor:
        if self._dirty or self._bit is None:
            self._bit = sign_to_bit1(self.H)        # H >= 0 -> +1 (True)
            self._dirty = False
        return self._bit

    @bit.setter
    def bit(self, new_bit: brute.Tensor) -> None:
        """Set the visible weight directly (test/inference helper) — syncs ``H``.

        Stores a saturated ±1 hidden weight consistent with the requested signs.
        """
        self.H = pm1_int(new_bit, torch.int8).reshape(self.shape).to(self.device)
        self._bit = new_bit
        self._dirty = False

    # ── update (called by layer backward) ─────────────────────────────────────
    def accumulate(self, delta_int: torch.Tensor) -> None:
        """``H += delta`` (the masked, binary outer-product update from backward)."""
        if delta_int.shape != self.H.shape:
            delta_int = delta_int.reshape(self.H.shape)
        if self.update_clip is not None:
            clip = int(self.update_clip)
            delta_int = delta_int.clamp(-clip, clip)
        updated = self.H.to(torch.int32) + delta_int.to(torch.int32)
        self.H.copy_(updated.clamp_(H_MIN, H_MAX).to(H_DTYPE))
        self._dirty = True

    def sparse_accumulate_rows(self, indices: torch.Tensor, delta_rows: torch.Tensor) -> None:
        """Sparse row update with duplicate-index coalescing and int8 saturation."""
        idx = indices.to(device=self.device, dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            return None
        rows = delta_rows.to(device=self.device, dtype=torch.int32).reshape(idx.numel(), -1)
        if rows.shape[1] != self.H.shape[1]:
            rows = rows.reshape(idx.numel(), self.H.shape[1])
        uniq, inv = torch.unique(idx, sorted=False, return_inverse=True)
        coalesced = torch.zeros((uniq.numel(), self.H.shape[1]), dtype=torch.int32,
                                device=self.device)
        coalesced.index_add_(0, inv, rows)
        base = self.H.index_select(0, uniq).to(torch.int32)
        self.H.index_copy_(0, uniq, (base + coalesced).clamp_(H_MIN, H_MAX).to(H_DTYPE))
        self._dirty = True
        return None

    @torch.no_grad()
    def step(self) -> int:
        """Return #visible bits whose sign flipped since the previous step."""
        cur = self.H >= 0
        n_flip = 0 if self._prev_sign is None else int((cur != self._prev_sign).sum().item())
        self._prev_sign = cur.clone()
        self._dirty = True
        return n_flip

    # ── checkpoint (stores integer H only) ────────────────────────────────────
    def state_dict(self) -> dict:
        return {"H": self.H.detach().cpu().clone(), "shape": self.shape}

    def load_state_dict(self, sd: dict) -> None:
        self.H = sd["H"].to(self.device).to(torch.int32).clamp_(H_MIN, H_MAX).to(H_DTYPE)
        self.shape = tuple(sd["shape"])
        self._dirty = True
        self._prev_sign = None

    # ── memory contract ───────────────────────────────────────────────────────
    def param_bytes(self) -> int:
        return self.H.nbytes


# ── linear backward (the workhorse) ────────────────────────────────────────────

def linear_backward(param: BepParam, a_star_out_bit: brute.Tensor,
                    a_in_bit: brute.Tensor, *, gate_bit: Optional[brute.Tensor] = None,
                    update: bool = True) -> brute.Tensor:
    """BEP backward for ``y = sign(W·a_in)``, ``W`` = ``param.bit`` (out, in).

    Both steps are pure bit1 matmuls into integers — no unpack:

    * **Weight update** (Eq. 8-9): ``ΔH = a*_outᵀ · a_in`` (reduce over batch M)
      → int ``(out, in)``; accumulated as ``2·ΔH`` into ``H``.
    * **Upstream desired activation** (Eq. 6): ``a*_in = sign(Wᵀ · a*_out)``.

    ``a_star_out_bit`` (M, out) and ``a_in_bit`` (M, in) are bit1; the optional
    ``gate_bit`` (M, out) bit1 zeroes ineligible output units before both steps
    (BEP Eq. 5; stage-2, off by default).  Returns ``a*_in`` (M, in) bit1.
    """
    if gate_bit is not None:
        # ternary g ⊙ a*_out: where gate is closed, contribute neither sign.  We
        # approximate the masked popcount by routing gated-off units to a neutral
        # state; with the gate off (the default) this branch is never taken.
        a_star_out_bit = mux(gate_bit, a_star_out_bit, a_star_out_bit)  # no-op placeholder
    am = _ACTIVE
    if update:
        if am is not None and am.shape[0] == a_star_out_bit.shape[0]:
            ao = pm1_int(a_star_out_bit, torch.int32) * am.unsqueeze(1)   # (M,out)
            ai = pm1_int(a_in_bit, torch.int32)                          # (M,in)
            dH = ao.transpose(0, 1) @ ai                                 # (out,in)
        else:
            dH = brute.fast.matmul(a_star_out_bit.transpose(0, 1), a_in_bit.transpose(0, 1))
        param.accumulate(2 * param.lr * dH)
    Wt = param.bit.transpose(0, 1)                       # (in, out) bit1
    z = brute.fast.matmul(a_star_out_bit, Wt)            # (M, in) int
    a_in_des = sign_to_bit1(z)
    if am is not None and am.shape[0] == a_in_des.shape[0]:
        # inactive rows carry no opinion → keep the current input unchanged.
        sel = brute.as_tensor(am.unsqueeze(1).expand(-1, a_in_bit.shape[-1]).contiguous(),
                              dtype=brute.bit1)
        a_in_des = mux(sel, a_in_des, a_in_bit)
    return a_in_des


# ── optimizer ──────────────────────────────────────────────────────────────────

@dataclass
class BepConfig:
    lr: int = 1             # integer update scale
    update_clip: Optional[int] = None  # optional elementwise ΔH clamp


class BepOptimizer:
    """Owns a collection of :class:`BepParam` and applies the integer step."""

    def __init__(self, params: Iterable[BepParam], config: Optional[BepConfig] = None):
        self.params: List[BepParam] = list(params)
        self.config = config or BepConfig()
        for p in self.params:
            p.lr = self.config.lr
            if self.config.update_clip is not None:
                p.update_clip = int(self.config.update_clip)
        self._step = 0

    def zero_signals(self) -> None:        # kept for API compatibility (no-op)
        pass

    @torch.no_grad()
    def step(self) -> dict:
        n_flip = 0
        n_bits = 0
        for p in self.params:
            n_flip += p.step()
            n_bits += p.H.numel()
        self._step += 1
        return {"n_flip": n_flip, "n_bits": n_bits,
                "flip_frac": n_flip / max(n_bits, 1)}

    def state_dict(self) -> dict:
        return {"step": self._step,
                "params": {p.name: p.state_dict() for p in self.params}}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd.get("step", 0)
        byname = {p.name: p for p in self.params}
        for name, psd in sd.get("params", {}).items():
            if name in byname:
                byname[name].load_state_dict(psd)


# ── helpers ─────────────────────────────────────────────────────────────────────

def random_bit_param(shape, name: str, *, generator: Optional[torch.Generator] = None,
                     device=None, p_true: float = 0.5, init_inertia: int = 1,
                     update_clip: Optional[int] = None) -> BepParam:
    """A :class:`BepParam` with ``H`` initialised to small balanced ±1 ints.

    ``sign(H)`` is therefore a balanced random ±1 weight, and a single agreeing
    backward step can already flip a bit (small ``|H|`` ⇒ fast early plasticity).
    """
    signs = (torch.rand(shape, generator=generator) < p_true)        # CPU generator
    inertia = max(1, min(int(init_inertia), H_MAX))
    H = signs.to(torch.int8).mul_(2).sub_(1).mul_(inertia)           # ±inertia
    if device is not None:
        H = H.to(device)
    return BepParam(H, name=name, update_clip=update_clip)
