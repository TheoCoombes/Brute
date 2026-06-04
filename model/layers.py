"""Boolean transformer layers — bit1 forward + BEP binary-desired backward.

Forward is bitwise (packed ``brute.bit1``); the only integers are *transient*
signed matmul pre-activations and the int8 vote accumulator inside attention's
value combine (see :mod:`attention`).  Backward is **BEP** (Boolean error
propagation): every layer receives a *binary desired activation* ``a*`` (bit1),
accumulates an **integer** weight update ``ΔH`` into its :class:`bep.BepParam`s
(the binary outer product, Eqs. 8-9) and returns the upstream binary desired
``a*_in = sign(Wᵀ a*_out)`` (Eq. 6).  Nothing float crosses a layer boundary and
nothing is unpacked to ±1 on the hot path.

This module holds the trainer-agnostic building blocks shared by the binary
transformer block: :class:`BooleanLinear` (XNOR-matmul + sign), :class:`DiagBind`
(diagonal binding), :class:`TokenCodebook` (frozen input geometry + trainable
output decoder), the exact :class:`BinaryGLU` feed-forward (XNOR gating), and the
:class:`MajorityResidual` (binary residual = coordinate majority).  Token↔token
mixing lives in :mod:`attention`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

import brute
from brute.tensor import Tensor as _BT

import bep
from bep import BepParam, random_bit_param, combine_desired, mux, pm1_int, signed_batch_sum
from vsa import balanced_hash_frame_bool, bind, bundle3, sign_to_bit1, to_bit1


# The output codebook is trainable now.  Starting its existing H buffer at ±16
# keeps step-0 signs identical while preventing a few early mistakes from moving
# prototypes faster than the hidden path can learn.
_CODEBOOK_INIT_INERTIA = 16


# ── packed broadcast binding (c ⊗ mask) ───────────────────────────────────────

def bind_mask(c_bit: brute.Tensor, mask_bit: brute.Tensor) -> brute.Tensor:
    """XNOR-bind ``c`` (..., D) with a single mask vector ``mask`` (D,), packed."""
    return brute.fast.eq(c_bit, mask_bit)


def _broadcast_like(vec_bit: brute.Tensor, ref_bit: brute.Tensor) -> brute.Tensor:
    """Broadcast a (D,) bit1 vector to ref's logical shape on the packed buffer."""
    pv = vec_bit._packed_buf
    target = list(ref_bit.shape[:-1]) + [pv.shape[-1]]
    pv_b = pv.expand(target)
    return _BT._make_bit1_from_packed(pv_b, list(ref_bit.shape))


# ── Boolean linear (XNOR matmul + sign) ────────────────────────────────────────

class BooleanLinear:
    """``z = a · Wᵀ`` (XNOR/popcount), ``out = sign(z)``.  Weight ``W = sign(H)``."""

    def __init__(self, in_dim: int, out_dim: int, *, name: str,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None,
                 init_inertia: int = 1,
                 update_clip: Optional[int] = None):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.W = random_bit_param((out_dim, in_dim), f"{name}.W",
                                  generator=generator, device=device,
                                  init_inertia=init_inertia,
                                  update_clip=update_clip)
        self.boundary_nu = boundary_nu
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return [self.W]

    def forward(self, a_bit: brute.Tensor) -> Tuple[brute.Tensor, torch.Tensor]:
        z = brute.fast.matmul(a_bit, self.W.bit)     # (M, out) int32, signed ±1 dot
        self._cache = {"a_in": a_bit, "z": z}
        return sign_to_bit1(z), z

    def backward(self, a_star_out: brute.Tensor) -> brute.Tensor:
        """``a_star_out`` (M, out) bit1 desired. Returns (M, in) bit1 desired."""
        return bep.linear_backward(self.W, a_star_out, self._cache["a_in"])


# ── Diagonal binding (learned mask) ────────────────────────────────────────────

class DiagBind:
    """``out = c ⊗ m`` with a learned 1-bit mask ``m`` (D,) — a diagonal transform."""

    def __init__(self, D: int, *, name: str, generator: Optional[torch.Generator] = None,
                 device=None, init_inertia: int = 1,
                 update_clip: Optional[int] = None):
        self.D = D
        self.m = random_bit_param((D,), f"{name}.m", generator=generator, device=device,
                                  init_inertia=init_inertia,
                                  update_clip=update_clip)
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return [self.m]

    def forward(self, c_bit: brute.Tensor) -> brute.Tensor:
        out = bind_mask(c_bit, self.m.bit)
        self._cache = {"c_bit": c_bit}
        return out

    def backward(self, a_star_out: brute.Tensor) -> brute.Tensor:
        """``a*`` (..., D) bit1. Updates ``m.H``; returns desired ``c*`` (..., D)."""
        c_bit = self._cache["c_bit"]
        # m wants out = c ⊗ m  ⇒  desired m per-coord = majority over batch of c ⊗ a*.
        flat_c = c_bit.reshape(-1, self.D)
        flat_a = a_star_out.reshape(-1, self.D)
        self.m.accumulate(signed_batch_sum(bind(flat_c, flat_a)))
        # desired c* = a* ⊗ m  (binding is its own inverse)
        return bind_mask(a_star_out, self.m.bit)


# ── Binary GLU feed-forward (exact XNOR gating) ────────────────────────────────

class BinaryGLU:
    """Exact binary GLU/GEGLU FFN (BOLD §3.2: GEGLU/SwiGLU → XNOR).

    ``h = xnor(sign(W_g·x), sign(W_m·x))``  then  ``f = sign(W_2·h)``.  The
    element-wise gate is the binary collapse of GEGLU/SwiGLU: the gating
    nonlinearity becomes a sign, and the gate·value product becomes a bind
    (XNOR).  All three projections are :class:`BooleanLinear`; backward is the
    clean BEP chain ``W_2→h*``; ``g*=h*⊗m``, ``m*=h*⊗g``; ``W_g, W_m`` backward.
    """

    def __init__(self, D: int, d_ff: int, *, name: str,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None, init_inertia: int = 1,
                 update_clip: Optional[int] = None):
        self.D, self.d_ff = D, d_ff
        self.Wg = BooleanLinear(D, d_ff, name=f"{name}.Wg", generator=generator,
                                device=device, boundary_nu=boundary_nu,
                                init_inertia=init_inertia, update_clip=update_clip)
        self.Wm = BooleanLinear(D, d_ff, name=f"{name}.Wm", generator=generator,
                                device=device, boundary_nu=boundary_nu,
                                init_inertia=init_inertia, update_clip=update_clip)
        self.W2 = BooleanLinear(d_ff, D, name=f"{name}.W2", generator=generator,
                                device=device, boundary_nu=boundary_nu,
                                init_inertia=init_inertia, update_clip=update_clip)
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return self.Wg.params() + self.Wm.params() + self.W2.params()

    def forward(self, x_bit: brute.Tensor) -> brute.Tensor:
        g_bit, _ = self.Wg.forward(x_bit)            # (M, d_ff) bit1
        m_bit, _ = self.Wm.forward(x_bit)
        h_bit = bind(g_bit, m_bit)                   # XNOR gate
        f_bit, _ = self.W2.forward(h_bit)            # (M, D) bit1
        self._cache = {"x_bit": x_bit, "g_bit": g_bit, "m_bit": m_bit}
        return f_bit

    def backward(self, f_star: brute.Tensor) -> brute.Tensor:
        """``f_star`` (M, D) bit1 desired → upstream desired ``x*`` (M, D) bit1."""
        h_star = self.W2.backward(f_star)            # (M, d_ff)
        g_bit, m_bit = self._cache["g_bit"], self._cache["m_bit"]
        # h = g ⊗ m  ⇒  g* = h* ⊗ m,  m* = h* ⊗ g   (binding is its own inverse)
        g_star = bind(h_star, m_bit)
        m_star = bind(h_star, g_bit)
        x_g = self.Wg.backward(g_star)
        x_m = self.Wm.backward(m_star)
        return combine_desired(x_g, x_m, self._cache["x_bit"])


# ── Majority residual (binary skip connection = coordinate-wise majority) ───────

# Per-step integer push that opens a value-path residual gate at supervised
# (margin-triggered) positions — breaks the chicken-and-egg on a retrieval branch
# whose downstream transport is random early.  Kept gentle and *constant*
# (independent of how many rows trigger) so it bootstraps retrieval without
# saturating the gate on locally-solvable tasks, where the ordinary agreement
# signal still closes it.
_VALUE_BOOST: float = 1.0


class MajorityResidual:
    """Binary residual admitted by a learned per-coordinate gate ``g``.

    Two admittance modes for the open coordinates:

    * ``mode='majority'`` — the BOLD-faithful binary residual ``maj3(skip, branch,
      c)`` (BOLD §3.1, "majority residual = sign(x + a + c)"): a coordinate
      majority of the skip, the sublayer branch and a learned ±1 tiebreak ``c``.
      This *adds* the branch's agreement to the skip; it cannot fully overwrite
      the skip (binary votes have no magnitude), so it suits aggregation but not
      exact value replacement.
    * ``mode='mux'`` — the value-replacing residual ``branch`` (open) / ``skip``
      (closed): where the gate is open the branch overwrites the skip.  This is
      the binary analogue of an *additive* residual whose branch can dominate by
      magnitude, and is what exact copy / retrieval (induction, previous-token)
      requires — a single binary majority cannot move a copied value past the
      skip that disagrees with it.

    Either way the gate is initialised mostly **closed** so the block is ≈
    identity at init (an untrained branch otherwise injects ~25% bit noise into
    the skip and destroys the signal before the readout can learn); closed
    coordinates pass the skip unchanged.  Fully packed (AND/OR/MUX only).
    """

    def __init__(self, D: int, *, name: str, mode: str = "mux", p_open: float = 0.05,
                 c_p_true: float = 0.5, value_path: bool = False,
                 generator: Optional[torch.Generator] = None, device=None,
                 init_inertia: int = 1, update_clip: Optional[int] = None):
        if mode not in ("mux", "majority"):
            raise ValueError("mode must be 'mux' or 'majority'")
        self.D = D
        self.mode = mode
        self.value_path = value_path
        self.g = random_bit_param((D,), f"{name}.g", generator=generator, device=device,
                                  p_true=p_open, init_inertia=init_inertia,
                                  update_clip=update_clip)
        self.c = random_bit_param((D,), f"{name}.c", generator=generator, device=device,
                                  p_true=c_p_true, init_inertia=init_inertia,
                                  update_clip=update_clip) if mode == "majority" else None
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return [self.g] if self.c is None else [self.g, self.c]

    def forward(self, skip_bit: brute.Tensor, branch_bit: brute.Tensor) -> brute.Tensor:
        if self.mode == "majority":
            admitted = bundle3(skip_bit, branch_bit, _broadcast_like(self.c.bit, skip_bit))
        else:
            admitted = branch_bit
        out = mux(_broadcast_like(self.g.bit, skip_bit), admitted, skip_bit)
        self._cache = {"skip_bit": skip_bit, "branch_bit": branch_bit, "admitted": admitted}
        return out

    def backward(self, out_star: brute.Tensor) -> Tuple[brute.Tensor, brute.Tensor]:
        """Return ``(skip*, branch*)``; update the admittance gate ``g`` (and, in
        majority mode, the tiebreak ``c``).

        The desired routes to both the skip and the branch (so the branch keeps
        learning even while its gate is closed); the gate opens where the admitted
        value predicts the desired better than the skip does.
        """
        skip = self._cache["skip_bit"]
        branch = self._cache["branch_bit"]
        admitted = self._cache["admitted"]
        flat_o = out_star.reshape(-1, self.D)
        flat_s = skip.reshape(-1, self.D)
        flat_a = admitted.reshape(-1, self.D)

        # gate update: open where the admitted value predicts the desired better than skip.
        dG = signed_batch_sum(bind(flat_a, flat_o)) - signed_batch_sum(bind(flat_s, flat_o))
        self.g.accumulate(dG)

        # tiebreak update (majority mode): push c toward out* where skip≠branch.
        if self.c is not None:
            disagree = brute.fast.bitwise_xor(flat_s, branch.reshape(-1, self.D))
            da = brute.fast.bitwise_and(disagree, flat_o)
            db = brute.fast.bitwise_and(disagree, brute.fast.bitwise_not(flat_o))
            dc = (signed_batch_sum(da) - signed_batch_sum(db)).div(2, rounding_mode="trunc")
            self.c.accumulate(dc)

        # Value-path open bootstrap (retrieval branches): push the gate open at the
        # supervised/triggered positions, sized per active sequence-position so the
        # threshold is batch- and length-invariant.  Safe because a closed
        # coordinate is the exact identity.
        am = bep.active_mask()
        if (self.value_path and _VALUE_BOOST and am is not None
                and am.shape[0] == flat_s.shape[0] and skip.dim() == 3):
            B, n, _ = skip.shape
            n_act_pos = int(am.reshape(B, n).any(dim=0).sum().item())
            if n_act_pos > 0:
                boost = int(int(am.sum().item()) / n_act_pos * _VALUE_BOOST)
                if boost:
                    self.g.accumulate(torch.full((self.D,), boost, dtype=torch.int32,
                                                 device=self.g.device))
        return out_star, out_star


# ── Token codebook (frozen input rows + trainable output decoder) ──────────────

class TokenCodebook:
    """Binary token codes with a frozen input geometry and trainable output ``E``.

    Input rows are regenerated from token ids and ``seed`` (never stored as a
    second V×D tensor), keeping input/output untied without extra per-weight state.
    """

    def __init__(self, vocab_size: int, D: int, *, name: str = "E",
                 structured: bool = False, bef_alpha: float = 1.0, bef_sweeps: int = 30,
                 generator: Optional[torch.Generator] = None, device=None,
                 seed: int = 0):
        self.V, self.D = vocab_size, D
        self._input_seed = int(seed) if structured else int(seed) ^ 0x5EED_5EED
        self._embed_cache: dict = {}

        dev = device
        H = torch.empty(vocab_size, D, dtype=torch.int16, device=dev)
        chunk = max(1, min(vocab_size, max(1, 4_194_304 // max(D, 1))))
        for start in range(0, vocab_size, chunk):
            end = min(start + chunk, vocab_size)
            ids = torch.arange(start, end, dtype=torch.long, device=dev)
            rows = balanced_hash_frame_bool(ids, D, seed=self._input_seed, device=dev)
            H[start:end] = rows.to(torch.int16).mul_(2).sub_(1).mul_(_CODEBOOK_INIT_INERTIA)
        self.E = BepParam(H, name=name)

    def params(self) -> List[BepParam]:
        return [self.E]

    # input side -------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> brute.Tensor:
        B, n = ids.shape
        flat = ids.reshape(-1).long()
        uniq, inv = torch.unique(flat, sorted=False, return_inverse=True)
        row_bits = balanced_hash_frame_bool(uniq, self.D, seed=self._input_seed,
                                            device=flat.device)
        rows = brute.as_tensor(row_bits[inv], dtype=brute.bit1)
        self._embed_cache = {"ids": flat, "shape": (B, n)}
        return rows.reshape(B, n, self.D) if rows.dim() == 2 else rows

    def backward_embed(self, a_star_rows: brute.Tensor) -> None:
        """Input codebook is frozen (fixed prototypes) — no update."""
        return None

    # output side ------------------------------------------------------------
    def decode(self, chat_bit: brute.Tensor) -> torch.Tensor:
        """``chat`` (M, D) bit1 → logits (M, V) int32 = ``<chat, E(t)>`` (stateless)."""
        return brute.fast.matmul(chat_bit, self.E.bit)     # (M, V) int32

    def prototype(self, ids: torch.Tensor) -> brute.Tensor:
        """Current output prototype rows ``E[ids]``."""
        return self.E.bit[ids.reshape(-1).long()]

    def backward_decode(self, ell_bit: brute.Tensor, target_ids: torch.Tensor,
                        wrong_ids: torch.Tensor, active_mask: torch.Tensor) -> None:
        """Sparse multiclass-perceptron update for the trainable output codebook.

        Triggered rows vote ``E[target] += ell`` and ``E[wrong] -= ell`` directly
        into the existing int16 ``H`` buffer with row ``index_add_``.  No dense
        ``V x D`` delta tensor and no additional per-weight state are allocated.
        """
        active = active_mask.to(device=self.E.device, dtype=torch.bool).reshape(-1)
        tgt = target_ids.to(device=self.E.device, dtype=torch.long).reshape(-1)
        wrong = wrong_ids.to(device=self.E.device, dtype=torch.long).reshape(-1)
        active = active & (tgt >= 0) & (tgt < self.V) & (wrong >= 0) & (wrong < self.V) & (tgt != wrong)
        if int(active.sum().item()) == 0:
            return None
        rows = pm1_int(ell_bit, dtype=torch.int16)[active]
        if self.E.lr != 1:
            rows = (rows.to(torch.int32) * int(self.E.lr)).to(torch.int16)
        self.E.H.index_add_(0, tgt[active], rows)
        self.E.H.index_add_(0, wrong[active], -rows)
        self.E._dirty = True
        return None
