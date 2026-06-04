"""HÆMMR Boolean layers — bit1 forward + BEP binary-desired-activation backward.

Forward is bitwise (packed ``brute.bit1``); the only integers are signed matmul
pre-activations and the BSR accumulator.  Backward is **BEP** (Boolean error
propagation): every layer receives a *binary desired activation* ``a*`` (bit1),
accumulates an **integer** weight update ``ΔH`` into its :class:`bep.BepParam`s
(the binary outer product, Eqs. 8-9) and returns the upstream binary desired
``a*_in = sign(Wᵀ a*_out)`` (Eq. 6).  Nothing float crosses a layer boundary and
nothing is unpacked to ±1 on the hot path — desired activations and gates are
bit1, accumulators are int.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

import brute
from brute.tensor import Tensor as _BT

import bep
from bep import BepParam, random_bit_param, combine_desired, mux, pm1_int, signed_batch_sum
from vsa import balanced_hash_frame_bool, bind, sign_to_bit1, to_bit1


# Per-step integer push that opens a value-path :class:`ResidualMerge` gate at
# triggered positions (see :meth:`ResidualMerge.backward`).  Kept gentle and
# *constant* (independent of how many rows trigger) so it bootstraps retrieval
# tasks without saturating the gate on tasks that are solvable from local
# context — there the ordinary agreement signal still closes the gate.
_VALUE_BOOST: float = 1.0

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


# ── Binary residual merge (per-coordinate MUX between skip and transform) ───────

class ResidualMerge:
    """Gated binary residual: ``out_d = trans_d if g_d (open) else skip_d``."""

    def __init__(self, D: int, *, name: str, p_open: float = 0.05,
                 value_path: bool = False,
                 generator: Optional[torch.Generator] = None, device=None,
                 init_inertia: int = 1,
                 update_clip: Optional[int] = None):
        self.D = D
        self.g = random_bit_param((D,), f"{name}.g", generator=generator,
                                  device=device, p_true=p_open,
                                  init_inertia=init_inertia,
                                  update_clip=update_clip)
        # ``value_path`` merges carry retrieved content that must reach the head at
        # supervised (triggered) positions — see :meth:`backward`'s open bootstrap.
        self.value_path = value_path
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return [self.g]

    def forward(self, skip_bit: brute.Tensor, trans_bit: brute.Tensor) -> brute.Tensor:
        out = mux(_broadcast_like(self.g.bit, skip_bit), trans_bit, skip_bit)
        self._cache = {"skip_bit": skip_bit, "trans_bit": trans_bit}
        return out

    def backward(self, a_star_out: brute.Tensor) -> Tuple[brute.Tensor, brute.Tensor]:
        """Route the binary desired to the selected branch; update the gate.

        Returns ``(a*_skip, a*_trans)``.  Open coords: ``a*`` goes to the
        transform, the skip keeps its current value (no opinion); closed coords:
        ``a*`` goes to the skip, the transform keeps its current value.
        """
        skip_bit = self._cache["skip_bit"]
        trans_bit = self._cache["trans_bit"]
        # Route the desired to *both* branches so the transform keeps learning even
        # while its gate is closed (avoids the dead-residual cold-start); the gate
        # still decides what the forward pass actually uses.
        a_skip = a_star_out
        a_trans = a_star_out
        # gate update: open where the transform agrees with a* more than the skip does.
        flat_a = a_star_out.reshape(-1, self.D)
        flat_s = skip_bit.reshape(-1, self.D)
        flat_t = trans_bit.reshape(-1, self.D)
        dH = signed_batch_sum(bind(flat_t, flat_a)) - signed_batch_sum(bind(flat_s, flat_a))
        self.g.accumulate(dH)

        # Value-path open bootstrap (breaks the chicken-and-egg on a retrieval merge):
        # the downstream lex transport is random early, so the chain desired ``a*`` is
        # noise and the gate gets no clean open signal — yet the transform branch (the
        # episodic read) is precisely what must reach the head at the *triggered* (LM-
        # supervised, margin-violating) positions.  Push the global per-coordinate gate
        # open there, scaled by the number of active rows.  This is a binary value-error
        # signal from the LM loss (the active mask = the margin trigger), complementing
        # the address-lane margin supervision; it is safe because the episodic read is
        # the identity at unmatched positions, so an open coordinate carries the skip
        # value wherever no slot was retrieved.
        am = bep.active_mask()
        if (self.value_path and _VALUE_BOOST and am is not None
                and am.shape[0] == flat_s.shape[0] and skip_bit.dim() == 3):
            # Open pressure for the gate, sized as the *mean number of triggered rows
            # per active sequence-position* (≈ the batch size).  On a true retrieval
            # task only the recall position triggers, so the skip can never predict
            # the target: its agreement signal is near-zero noise of magnitude ~B, and
            # this push (~B) opens the gate.  On a locally-solvable task many positions
            # trigger consistently, so the genuine close signal scales as B·#positions
            # and dominates the same ~B push — the gate settles closed.  Using rows
            # *per position* (not the raw count) makes the threshold batch-size- and
            # sequence-length-invariant.
            B, n, _ = skip_bit.shape
            am_bn = am.reshape(B, n)
            n_act_pos = int(am_bn.any(dim=0).sum().item())
            if n_act_pos > 0:
                boost = int(int(am.sum().item()) / n_act_pos * _VALUE_BOOST)
                if boost:
                    self.g.accumulate(torch.full((self.D,), boost, dtype=torch.int32,
                                                 device=self.g.device))
        return a_skip, a_trans


# ── Token codebook (frozen input rows + trainable output decoder) ──────────────

class TokenCodebook:
    """Binary token codes with a frozen input geometry and trainable output ``E``.

    The output codebook is the existing :class:`BepParam` named ``E`` so the
    locked decode matmul is unchanged.  For default balanced-hash initialisation,
    input rows are regenerated from token ids and ``seed`` instead of stored in
    a second ``V x D`` tensor; this unties input/output without adding another
    per-weight accumulator.  Arbitrary offline initialisers cannot be regenerated
    under that memory constraint, so they remain tied opt-in ablations.
    """

    def __init__(self, vocab_size: int, D: int, *, name: str = "E",
                 structured: bool = False, bef_alpha: float = 1.0, bef_sweeps: int = 30,
                 init_pm1: Optional[torch.Tensor] = None,
                 generator: Optional[torch.Generator] = None, device=None,
                 seed: int = 0):
        self.V, self.D = vocab_size, D
        self._input_seed: Optional[int] = None
        self._embed_cache: dict = {}
        if init_pm1 is not None:
            # Precomputed prototypes (e.g. an offline GPT-2 SimHash codebook).
            if tuple(init_pm1.shape) != (vocab_size, D):
                raise ValueError(
                    f"init_pm1 shape {tuple(init_pm1.shape)} != ({vocab_size}, {D})")
            frame = init_pm1
            bit = brute.as_tensor(frame > 0, dtype=brute.bit1)
            if device is not None:
                bit = bit.to(device)
            self.E = BepParam(pm1_int(bit) * _CODEBOOK_INIT_INERTIA, name=name)
            return
        elif structured:
            self._input_seed = int(seed)
        else:
            self._input_seed = int(seed) ^ 0x5EED_5EED

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
        if self._input_seed is None:
            rows = self.E.bit[flat]                        # offline ablation: tied
        else:
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


# ── Latent Hopfield Bank (binary associative memory / latent attention) ────────

class HopfieldBank:
    """M learned slots — key ``P`` (M,D), payload ``U`` (M,D).  Top-k WTA read."""

    def __init__(self, D: int, n_slots: int, top_k: int, *, name: str = "hop",
                 generator: Optional[torch.Generator] = None, device=None,
                 init_inertia: int = 1,
                 update_clip: Optional[int] = None):
        self.D, self.M, self.k = D, n_slots, top_k
        self.P = random_bit_param((n_slots, D), f"{name}.P", generator=generator, device=device,
                                  init_inertia=init_inertia,
                                  update_clip=update_clip)
        self.U = random_bit_param((n_slots, D), f"{name}.U", generator=generator, device=device,
                                  init_inertia=init_inertia,
                                  update_clip=update_clip)
        self._cache: dict = {}

    def params(self) -> List[BepParam]:
        return [self.P, self.U]

    def forward(self, q_bit: brute.Tensor) -> brute.Tensor:
        """``q`` (M_tok, D) bit1 → read (M_tok, D) bit1 = ``sign(Σ_topk U_m)``."""
        sim = brute.fast.matmul(q_bit, self.P.bit)         # (M_tok, M) int32
        kk = min(self.k, self.M)
        _, idx = torch.topk(sim, k=kk, dim=1)              # (M_tok, k)
        U_int = pm1_int(self.U.bit)                        # (M, D)
        sel = U_int[idx]                                   # (M_tok, k, D)
        read = sel.sum(dim=1)                              # (M_tok, D) integer vote
        self._cache = {"idx": idx, "q_bit": q_bit}
        return sign_to_bit1(read)

    def backward(self, a_star_read: brute.Tensor) -> None:
        """Scatter the binary desired payload into selected ``U`` slots; Hebbian
        key update into ``P``.  The hard top-k selection is piecewise-constant in
        the query ⇒ no upstream signal (returns ``None``)."""
        idx = self._cache["idx"]                           # (M_tok, k)
        q_bit = self._cache["q_bit"]                       # (M_tok, D)
        Mtok, kk = idx.shape
        flat_idx = idx.reshape(-1)                         # (M_tok*k,)
        r_int = pm1_int(a_star_read)                       # (M_tok, D)
        am = bep.active_mask()
        if am is not None and am.shape[0] == Mtok:
            r_int = r_int * am.unsqueeze(1).to(r_int.dtype)
        dU = torch.zeros_like(self.U.H)
        dU.index_add_(0, flat_idx,
                      r_int.unsqueeze(1).expand(Mtok, kk, self.D).reshape(-1, self.D).to(dU.dtype))
        self.U.accumulate(dU)
        q_int = pm1_int(q_bit)
        if am is not None and am.shape[0] == Mtok:
            q_int = q_int * am.unsqueeze(1).to(q_int.dtype)
        dP = torch.zeros_like(self.P.H)
        dP.index_add_(0, flat_idx,
                      q_int.unsqueeze(1).expand(Mtok, kk, self.D).reshape(-1, self.D).to(dP.dtype))
        self.P.accumulate(dP)
        return None


# ── Binary Episodic Slot Memory (exact in-window recall) ───────────────────────

class EpisodicSlotMemory:
    """Addressable binary slots for *exact* context-local recall (HÆMMR v2 §B1).

    Score ``score[t,m] = ⟨q^c_t, k^c_m⟩ + ⟨q^p_t, POS_m⟩`` for ``m < t``
    (read-before-write), computed by integer bit1 matmuls; forward read is hard
    top-1 (exact).  Backward scatters the binary desired read into the selected
    *source* slot's concept (the value path); the address lane (Kc/Qc/Qp) is
    trained by :meth:`margin_loss` only (BEP discrete hinge → binary desired
    activations + integer ``ΔH``).
    """

    def __init__(self, D: int, *, name: str = "epi", read_k: int = 1,
                 n_slots: Optional[int] = None, attn_inv_temp: Optional[float] = None,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None,
                 sink_threshold: Optional[int] = None,
                 init_inertia: int = 1,
                 update_clip: Optional[int] = None):
        self.D = D
        self.k = read_k
        self.N = n_slots
        self.sink_threshold = 0 if sink_threshold is None else int(sink_threshold)
        self.Kc = BooleanLinear(D, D, name=f"{name}.Kc", generator=generator, device=device,
                                boundary_nu=boundary_nu, init_inertia=init_inertia,
                                update_clip=update_clip)
        self.Qc = BooleanLinear(D, D, name=f"{name}.Qc", generator=generator, device=device,
                                boundary_nu=boundary_nu, init_inertia=init_inertia,
                                update_clip=update_clip)
        self.Qp = BooleanLinear(D, D, name=f"{name}.Qp", generator=generator, device=device,
                                boundary_nu=boundary_nu, init_inertia=init_inertia,
                                update_clip=update_clip)
        self._cache: dict = {}
        self._s_kc = self._s_pos = self._s_pay = None
        self._s_ptr = 0
        self._s_cnt = 0

    def params(self) -> List[BepParam]:
        return self.Kc.params() + self.Qc.params() + self.Qp.params()

    _NEG = -(1 << 30)

    def forward(self, c_bit: brute.Tensor, pos_bit: brute.Tensor | None) -> brute.Tensor:
        B, n, D = c_bit.shape
        c_flat = c_bit.reshape(B * n, D)
        kc_bit, _ = self.Kc.forward(c_flat)
        qc_bit, _ = self.Qc.forward(c_flat)
        qp_bit, _ = self.Qp.forward(c_flat)
        kc_bit = kc_bit.reshape(B, n, D)
        qc_bit = qc_bit.reshape(B, n, D)
        qp_bit = qp_bit.reshape(B, n, D)
        if pos_bit is None:
            pos_bit = to_bit1(torch.ones(n, D, dtype=torch.bool, device=c_bit.device))

        t = torch.arange(n, device=c_bit.device).unsqueeze(1)
        m = torch.arange(n, device=c_bit.device).unsqueeze(0)
        causal = m < t
        if self.N is not None:
            causal = causal & ((t - m) <= self.N)
        cmask = (~causal).unsqueeze(0)                     # (1,n,n)

        qc_int = pm1_int(qc_bit, dtype=torch.int32)
        kc_int = pm1_int(kc_bit, dtype=torch.int32)
        qp_int = pm1_int(qp_bit, dtype=torch.int32)
        pos_int = pm1_int(pos_bit, dtype=torch.int32)
        content = torch.bmm(qc_int, kc_int.transpose(1, 2))         # (B,n,n)
        position = torch.matmul(qp_int, pos_int.transpose(0, 1))    # (B,n,n)
        score = content + position
        score = score.masked_fill(cmask, self._NEG)

        kk = min(self.k, n, self.N or n)
        if kk == 1:
            topv, topi = score.max(dim=2, keepdim=True)
        else:
            topv, topi = torch.topk(score, k=kk, dim=2)
        valid_top = (topv > (self._NEG // 2)) & (topv >= self.sink_threshold)
        no_slot = ~valid_top.any(dim=2)                    # (B,n)

        c_int = pm1_int(c_bit, dtype=torch.int32)                    # (B,n,D)
        batch_idx = torch.arange(B, device=c_bit.device).view(B, 1, 1).expand(B, n, kk)
        sel = c_int[batch_idx, topi]                                 # (B,n,kk,D)
        read = (sel * valid_top.unsqueeze(-1)).sum(dim=2)             # (B,n,D)
        read = torch.where(no_slot.unsqueeze(-1), c_int, read)
        read_bit = sign_to_bit1(read)

        self._cache = {
            "shape": (B, n, D), "c_bit": c_bit,
            "kc_bit": kc_bit, "qc_bit": qc_bit, "qp_bit": qp_bit, "pos_bit": pos_bit,
            "score": score, "topi": topi, "valid_top": valid_top,
            "no_slot": no_slot, "causal": causal, "c_flat": c_flat,
        }
        return read_bit

    def backward(self, a_star_read: brute.Tensor) -> brute.Tensor:
        """Value path: route the binary desired read back to the selected source.

        Returns the desired concept ``c*`` (B,n,D); the address lane is trained
        by :meth:`margin_loss`, not here.
        """
        B, n, D = self._cache["shape"]
        topi = self._cache["topi"]                         # (B,n,kk)
        valid_top = self._cache["valid_top"]
        no_slot = self._cache["no_slot"]
        c_bit = self._cache["c_bit"]
        kk = topi.shape[2]
        r_int = pm1_int(a_star_read, dtype=torch.int32)    # (B,n,D)
        am = bep.active_mask()
        if am is not None and am.shape[0] == B * n:
            r_int = r_int * am.reshape(B, n, 1).to(r_int.dtype)

        flat_acc = torch.zeros(B * n, D, dtype=torch.int32, device=a_star_read.device)
        batch_base = (torch.arange(B, device=a_star_read.device) * n).view(B, 1, 1)
        flat_idx = (batch_base + topi).reshape(-1)
        src = (r_int.unsqueeze(2) * valid_top.unsqueeze(-1)).reshape(-1, D)
        flat_acc.index_add_(0, flat_idx, src)
        acc = flat_acc.reshape(B, n, D)
        acc += no_slot.unsqueeze(-1) * r_int                 # no-slot rows: identity
        cur = c_bit.bool()
        out = torch.where(acc > 0, torch.ones_like(cur),
                          torch.where(acc < 0, torch.zeros_like(cur), cur))
        return to_bit1(out)

    # ── §F local Hamming-margin objective (binary desired activations) ─────────
    def margin_loss(self, matched: torch.Tensor, *, theta_pos: float = 0.5,
                    theta_neg: float = 0.0, weight: float = 1.0,
                    active_rows: Optional[torch.Tensor] = None) -> float:
        """Push the matched query/key pair above ``θ⁺`` (and the best distractor
        below ``θ⁻``) by emitting binary desired activations for the address
        projections and accumulating their integer ``ΔH`` (HÆMMR v2 §F).
        """
        B, n, D = self._cache["shape"]
        kc_bit, qc_bit, qp_bit = self._cache["kc_bit"], self._cache["qc_bit"], self._cache["qp_bit"]
        pos_bit, score = self._cache["pos_bit"], self._cache["score"]
        tp, tn = theta_pos * D, theta_neg * D
        causal = self._cache["causal"].unsqueeze(0).expand(B, n, n)
        in_range = (matched >= 0) & (matched < n)
        safe_idx = matched.clamp(0, max(n - 1, 0))
        valid = in_range & torch.gather(causal, 2, safe_idx.unsqueeze(-1)).squeeze(-1)
        if active_rows is not None:
            valid = valid & active_rows.to(device=valid.device, dtype=torch.bool).reshape(B, n)
        if int(valid.sum()) == 0:
            return 0.0
        pos_score = torch.gather(score, 2, safe_idx.unsqueeze(-1)).squeeze(-1)
        distract = score.clone()
        distract.scatter_(2, safe_idx.unsqueeze(-1), self._NEG)
        neg_score, neg_idx = distract.max(dim=2)

        pos_viol = (pos_score < tp) & valid                # (B,n)
        neg_viol = (neg_score > tn) & valid
        loss = (((tp - pos_score).clamp_min(0) + (neg_score - tn).clamp_min(0)) * valid).sum()
        n_valid = int(valid.sum())
        loss_val = float(loss.item()) / max(n_valid, 1)

        kc_b = kc_bit.bool(); qc_b = qc_bit.bool(); qp_b = qp_bit.bool()
        pos_b = pos_bit.bool().unsqueeze(0).expand(B, n, D)
        idx_m = safe_idx.unsqueeze(-1).expand(B, n, D)
        idx_d = neg_idx.unsqueeze(-1).expand(B, n, D)
        k_match = torch.gather(kc_b, 1, idx_m)             # (B,n,D)
        k_neg = torch.gather(kc_b, 1, idx_d)
        pos_match = torch.gather(pos_b, 1, idx_m)

        # desired query (content): matched → equal k_match; distractor → anti k_neg.
        pv = pos_viol.unsqueeze(-1); nv = neg_viol.unsqueeze(-1)
        qc_des = torch.where(nv, ~k_neg, qc_b)
        qc_des = torch.where(pv, k_match, qc_des)
        qp_des = torch.where(pv, pos_match, qp_b)

        # desired key at the matched slot: equal qc_t (majority over queries).
        dk = torch.zeros(B, n, D, dtype=torch.int32, device=score.device)
        qc_pm1 = pm1_int(qc_bit, dtype=torch.int32)
        contrib = qc_pm1 * pos_viol.unsqueeze(-1)
        dk.scatter_add_(1, idx_m, contrib)
        kc_des = torch.where(dk > 0, torch.ones_like(kc_b),
                             torch.where(dk < 0, torch.zeros_like(kc_b), kc_b))

        prev_active = bep.active_mask()
        bep.set_active((pos_viol | neg_viol).reshape(-1))
        self.Qc.backward(to_bit1(qc_des).reshape(B * n, D))
        self.Qp.backward(to_bit1(qp_des).reshape(B * n, D))
        self.Kc.backward(to_bit1(kc_des).reshape(B * n, D))
        bep.set_active(prev_active)
        return loss_val

    # ── streaming (inference) ring-buffer form ────────────────────────────────
    @torch.no_grad()
    def reset_stream(self, batch_size: int, n_slots: int, *, device=None) -> None:
        dev = device if device is not None else self.Kc.W.device
        self._s_kc = torch.zeros(batch_size, n_slots, self.D, dtype=torch.int16, device=dev)
        self._s_pos = torch.zeros(batch_size, n_slots, self.D, dtype=torch.int16, device=dev)
        self._s_pay = torch.zeros(batch_size, n_slots, self.D, dtype=torch.int16, device=dev)
        self._s_ptr = 0
        self._s_cnt = 0

    @torch.no_grad()
    def step(self, c_bit: brute.Tensor, pos_bit: brute.Tensor, n_slots: int) -> brute.Tensor:
        if c_bit.dim() == 1:
            c_bit = c_bit.reshape(1, self.D)
        B, D = c_bit.shape
        if self._s_kc is None or self._s_kc.shape[0] != B or self._s_kc.shape[1] != n_slots:
            self.reset_stream(B, n_slots, device=c_bit.device)
        qc = pm1_int(self.Qc.forward(c_bit)[0])
        qp = pm1_int(self.Qp.forward(c_bit)[0])
        c_int = pm1_int(c_bit)
        if self._s_cnt == 0:
            read = c_int
        else:
            cnt = self._s_cnt
            content = torch.einsum("bd,bsd->bs", qc.int(), self._s_kc[:, :cnt].int())
            position = torch.einsum("bd,bsd->bs", qp.int(), self._s_pos[:, :cnt].int())
            score = content + position
            sel = score.argmax(dim=1)
            best = score.gather(1, sel.view(B, 1)).squeeze(1)
            picked = torch.gather(self._s_pay[:, :cnt], 1,
                                  sel.view(B, 1, 1).expand(B, 1, D)).squeeze(1)
            read = torch.where((best >= self.sink_threshold).view(B, 1), picked, c_int)
        kc = pm1_int(self.Kc.forward(c_bit)[0])
        pos_int = pm1_int(pos_bit) if getattr(pos_bit, "_is_bit1", False) else pos_bit.to(torch.int16)
        p = self._s_ptr
        self._s_kc[:, p] = kc
        self._s_pos[:, p] = pos_int.reshape(1, D).expand(B, D)
        self._s_pay[:, p] = c_int
        self._s_ptr = (p + 1) % n_slots
        self._s_cnt = min(self._s_cnt + 1, n_slots)
        return sign_to_bit1(read)


# ── Bundling State Recurrence (BSR) — linear-time binary sequence mixer ─────────

def pow2_decay_shifts(D: int, shifts=(1, 2, 3, 4, 0), device=None) -> torch.Tensor:
    """Per-coordinate right-shift amount from a small palette (v2 §B2).

    Decay is the exact integer recurrence ``A ← A − (A >> s)`` (multiplier
    ``1 − 2^{−s}``); ``s = 0`` means *permanent* (no decay).  Returns the (D,)
    int shift tensor (channel groups, RetNet-style multi-timescale retention).
    """
    groups = len(shifts)
    idx = (torch.arange(D, device=device) * groups) // D
    out = torch.empty(D, dtype=torch.int32, device=device)
    for g, s in enumerate(shifts):
        out[idx == g] = int(s)
    return out


def pow2_decay_palette(D: int, shifts=(1, 2, 3, 4, 0), device=None) -> torch.Tensor:
    """Float multiplier view of the shift palette (kept for reference/tests)."""
    s = pow2_decay_shifts(D, shifts, device=device)
    mult = torch.where(s == 0, torch.ones_like(s, dtype=torch.float32),
                       1.0 - 2.0 ** (-s.to(torch.float32)))
    return mult


def _decay_int(A: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Integer decay ``A − (A >> s)`` per coordinate; ``s = 0`` ⇒ permanent."""
    shifted = torch.bitwise_right_shift(A, shifts)
    return torch.where(shifts == 0, A, A - shifted)


class BSR:
    """Binary delta-corrected linear-recurrence context mixer (HÆMMR v2 §B2).

    Per position the key/value/query are dense 1-bit projections of the concept;
    an **integer** accumulator ``A`` holds the bundled associations under the
    erase-before-write delta rule::

        S_i = sign(A_i),  r_i = q_i ⊗ S_i
        g_i = 1[ k_i⊗S_i ≠ v_i ]                 # disagreement gate (XOR)
        A_{i+1} = decay(A_i) + g_i ⊙ e(k_i ⊗ v_i)

    Forward is an O(n) integer scan; backward is the O(n) reverse-scan BEP
    adjoint with a **binary** desired-state accumulator (sign / hard gate treated
    as pass-through, decay permanent).
    """

    def __init__(self, D: int, *, name: str = "bsr",
                 decay_shifts=(1, 2, 3, 4, 0),
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None,
                 init_inertia: int = 1,
                 update_clip: Optional[int] = None):
        self.D = D
        self.K = BooleanLinear(D, D, name=f"{name}.K", generator=generator, device=device,
                               boundary_nu=boundary_nu, init_inertia=init_inertia,
                               update_clip=update_clip)
        self.V = BooleanLinear(D, D, name=f"{name}.V", generator=generator, device=device,
                               boundary_nu=boundary_nu, init_inertia=init_inertia,
                               update_clip=update_clip)
        self.Q = BooleanLinear(D, D, name=f"{name}.Q", generator=generator, device=device,
                               boundary_nu=boundary_nu, init_inertia=init_inertia,
                               update_clip=update_clip)
        self.shifts = pow2_decay_shifts(D, decay_shifts, device=device)        # (D,)
        self._cache: dict = {}
        self._stream_A: Optional[torch.Tensor] = None

    def params(self) -> List[BepParam]:
        return self.K.params() + self.V.params() + self.Q.params()

    def forward(self, c_bit: brute.Tensor) -> brute.Tensor:
        B, n, D = c_bit.shape
        c_flat = c_bit.reshape(B * n, D)
        k_bit, _ = self.K.forward(c_flat)
        v_bit, _ = self.V.forward(c_flat)
        q_bit, _ = self.Q.forward(c_flat)
        k_bit = k_bit.reshape(B, n, D)
        v_bit = v_bit.reshape(B, n, D)
        q_bit = q_bit.reshape(B, n, D)
        k_int = pm1_int(k_bit, dtype=torch.int32)
        v_int = pm1_int(v_bit, dtype=torch.int32)
        assoc_int = k_int * v_int                            # (B,n,D) ±1

        shifts = self.shifts
        A = torch.zeros(B, D, dtype=torch.int32, device=c_bit.device)
        S_state = torch.empty(B, n, D, dtype=torch.bool, device=c_bit.device)
        gate = torch.empty(B, n, D, dtype=torch.bool, device=c_bit.device)
        for i in range(n):
            Sge = A >= 0                                     # bool, +1 where True
            S_state[:, i, :] = Sge
            Spm1 = torch.where(Sge, 1, -1).to(torch.int32)
            pred = k_int[:, i, :] * Spm1
            g_i = pred != v_int[:, i, :]                     # disagreement
            gate[:, i, :] = g_i
            A = _decay_int(A, shifts) + g_i.to(torch.int32) * assoc_int[:, i, :]
        S_state_bit = to_bit1(S_state)
        self._cache = {
            "k_bit": k_bit, "v_bit": v_bit, "q_bit": q_bit,
            "S_state_bit": S_state_bit, "gate": gate, "shape": (B, n, D),
            "c_bit": c_bit,
        }
        return bind(q_bit, S_state_bit)

    def backward(self, a_star_r: brute.Tensor) -> brute.Tensor:
        """``a*_r`` (B,n,D) bit1 desired read. Returns desired ``c*`` (B,n,D)."""
        B, n, D = self._cache["shape"]
        k_bit, v_bit, q_bit = self._cache["k_bit"], self._cache["v_bit"], self._cache["q_bit"]
        S_state_bit = self._cache["S_state_bit"]
        gate = self._cache["gate"]
        c_bit = self._cache["c_bit"]

        # r_i = q_i ⊗ S_i  ⇒  desired q_i* = a*_r ⊗ S_i,  desired S_i* = a*_r ⊗ q_i
        q_des = bind(a_star_r, S_state_bit)                 # (B,n,D)
        s_des = bind(a_star_r, q_bit)

        # reverse-scan binary adjoint: a desired future state Abar flows back; at
        # each step it contributes the desired association (gated), and combines
        # with the local desired state (decay permanent ⇒ pass-through).
        assoc_des = torch.empty(B, n, D, dtype=torch.bool, device=a_star_r.device)
        gate_b = gate
        Abar: Optional[brute.Tensor] = None
        S_b = S_state_bit.bool()
        for i in range(n - 1, -1, -1):
            if Abar is None:
                ad_i = c_bit[:, i, :].bool()                # no future desire yet → no-op
                assoc_des[:, i, :] = ad_i
                Abar = to_bit1(s_des[:, i, :].bool())
            else:
                abar_b = Abar.bool()
                assoc_des[:, i, :] = abar_b
                Abar = combine_desired(to_bit1(s_des[:, i, :].bool()), Abar,
                                       to_bit1(S_b[:, i, :]))
        # where the gate was closed the association did not write → reinforce
        # current assoc (no harmful flip): assoc_des = gate ? Abar : (k⊗v current)
        k_b = k_bit.bool(); v_b = v_bit.bool()
        assoc_cur = ~(k_b ^ v_b)                            # k⊗v as bool (xnor)
        assoc_des = torch.where(gate_b, assoc_des, assoc_cur)
        assoc_des_bit = to_bit1(assoc_des)

        # assoc = k ⊗ v  ⇒  desired k = assoc_des ⊗ v,  desired v = assoc_des ⊗ k
        k_des = bind(assoc_des_bit, v_bit)
        v_des = bind(assoc_des_bit, k_bit)

        g_ck = self.K.backward(k_des.reshape(B * n, D)).reshape(B, n, D)
        g_cv = self.V.backward(v_des.reshape(B * n, D)).reshape(B, n, D)
        g_cq = self.Q.backward(q_des.reshape(B * n, D)).reshape(B, n, D)
        g_c = combine_desired(g_ck, g_cv, c_bit)
        g_c = combine_desired(g_c, g_cq, c_bit)
        return g_c

    @torch.no_grad()
    def reset_stream(self, batch_size: int, *, device=None) -> None:
        dev = device if device is not None else self.shifts.device
        self._stream_A = torch.zeros(batch_size, self.D, dtype=torch.int32, device=dev)

    @torch.no_grad()
    def step(self, c_bit: brute.Tensor) -> brute.Tensor:
        if c_bit.dim() == 1:
            c_bit = c_bit.reshape(1, self.D)
        B, D = c_bit.shape
        if self._stream_A is None or self._stream_A.shape[0] != B:
            self.reset_stream(B, device=c_bit.device)
        k_bit, _ = self.K.forward(c_bit)
        v_bit, _ = self.V.forward(c_bit)
        q_bit, _ = self.Q.forward(c_bit)
        k_int = pm1_int(k_bit, dtype=torch.int32)
        v_int = pm1_int(v_bit, dtype=torch.int32)
        assoc_int = k_int * v_int
        Sge = self._stream_A >= 0
        r = bind(q_bit, to_bit1(Sge))
        Spm1 = torch.where(Sge, 1, -1).to(torch.int32)
        pred = k_int * Spm1
        g_i = (pred != v_int).to(torch.int32)
        self._stream_A = _decay_int(self._stream_A, self.shifts) + g_i * assoc_int
        return r
