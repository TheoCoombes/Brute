"""HÆMMR Boolean layers — bit1 forward + BOLD-signal backward.

Each layer caches the bipolar activations it needs during ``forward`` and, in
``backward``, (a) accumulates the flip-signal ``q`` onto every :class:`BoldParam`
it owns (BOLD Eqs. 5/7) and (b) returns the upstream signal ``g`` (Eqs. 6/8)
for the layer below.  Signals are real-valued — their *sign* is the Boolean
variation and their *magnitude* is the confidence — exactly as BOLD prescribes;
only the forward pass is bitwise.

Convention: ``S`` denotes δLoss/δactivation arriving from downstream (a positive
sign means "loss increases if this activation increases").  ``sign``/threshold
activations are treated as pass-through (BOLD's threshold variation), and matmul
backward signals are variance-scaled by ``sqrt(2/fan_out)`` (BOLD §3.3).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

import brute
from brute.tensor import Tensor as _BT

from bold import BoldParam, random_bit_param, signal_scale
from vsa import bind, sign_to_bit1, to_pm1


class _LazyPM1Cache(dict):
    """Forward cache that delays packed bit materialisation until backward."""

    def __init__(self, *args, pm1_sources=None, float01_sources=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pm1_sources = pm1_sources or {}
        self._float01_sources = float01_sources or {}

    def __getitem__(self, key):
        if not super().__contains__(key) and key in self._pm1_sources:
            bit_key, shape = self._pm1_sources[key]
            value = to_pm1(super().__getitem__(bit_key))
            if shape is not None:
                value = value.reshape(*shape)
            super().__setitem__(key, value)
            return value
        if not super().__contains__(key) and key in self._float01_sources:
            bit_key, shape = self._float01_sources[key]
            value = super().__getitem__(bit_key).bool().as_subclass(torch.Tensor).to(torch.float32)
            if shape is not None:
                value = value.reshape(*shape)
            super().__setitem__(key, value)
            return value
        return super().__getitem__(key)

    def __contains__(self, key):
        return (super().__contains__(key)
                or key in self._pm1_sources
                or key in self._float01_sources)

    def get(self, key, default=None):
        return self[key] if key in self else default


# ── packed broadcast binding (c ⊗ mask) ───────────────────────────────────────

def bind_mask(c_bit: brute.Tensor, mask_bit: brute.Tensor) -> brute.Tensor:
    """XNOR-bind ``c`` (..., D) with a single mask vector ``mask`` (D,), packed.

    Broadcasts on the packed buffers — no unpack to bool.
    """
    return brute.fast.eq(c_bit, mask_bit)


# ── Boolean linear (XNOR matmul + sign) ────────────────────────────────────────

class BooleanLinear:
    """``z = a · Wᵀ`` (XNOR/popcount), ``out = sign(z)``.  Weight ``W`` is (out, in).

    ``boundary_nu`` (BEP Eq. 5) optionally gates the backward
    signal through the *eligibility mask* ``|z| ≤ ν·in_dim`` — only units whose
    pre-activation is near the decision boundary (i.e. a weight flip could
    actually change their sign) propagate a signal.  This focuses flips where
    they matter and is a partial remedy for BOLD's error floor.  ``None`` (the
    default) leaves the signal ungated.
    """

    def __init__(self, in_dim: int, out_dim: int, *, name: str,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.W = random_bit_param((out_dim, in_dim), f"{name}.W",
                                  generator=generator, device=device)
        self.boundary_nu = boundary_nu
        self._cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.W]

    def forward(self, a_bit: brute.Tensor) -> Tuple[brute.Tensor, torch.Tensor]:
        z = brute.fast.matmul(a_bit, self.W.bit)     # (M, out) int32, signed ±1 dot
        self._cache = _LazyPM1Cache(
            {"a_bit": a_bit, "z": z},
            pm1_sources={"a_pm1": ("a_bit", tuple(a_bit.shape))},
        )
        return sign_to_bit1(z), z

    def backward(self, S: torch.Tensor) -> torch.Tensor:
        """``S``: (M, out) signal at the output activation. Returns (M, in)."""
        a_pm1 = self._cache["a_pm1"]
        if self.boundary_nu is not None:
            z = self._cache["z"].reshape(S.shape)
            elig = (z.abs() <= self.boundary_nu * self.in_dim).to(S.dtype)   # BEP Eq. 5
            S = S * elig
        # Eq. 7 — weight flip-signal q_W = Sᵀ · a_pm1   (out, in)
        self.W.add_signal(S.transpose(0, 1) @ a_pm1)
        # Eq. 8 — upstream signal g = S · W_pm1, variance-scaled.
        g = (S @ self.W.pm1) * signal_scale(self.out_dim)
        return g


# ── Diagonal binding (learned mask) ────────────────────────────────────────────

class DiagBind:
    """``out = c ⊗ m`` with a learned 1-bit mask ``m`` (D,) — a diagonal transform."""

    def __init__(self, D: int, *, name: str, generator: Optional[torch.Generator] = None,
                 device=None):
        self.D = D
        self.m = random_bit_param((D,), f"{name}.m", generator=generator, device=device)
        self._cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.m]

    def forward(self, c_bit: brute.Tensor) -> brute.Tensor:
        out = bind_mask(c_bit, self.m.bit)
        self._cache = _LazyPM1Cache(
            {"c_bit": c_bit},
            pm1_sources={"c_pm1": ("c_bit", tuple(c_bit.shape))},
        )
        return out

    def backward(self, S: torch.Tensor) -> torch.Tensor:
        """``S``: (..., D). Returns signal to ``c`` (..., D)."""
        c_pm1 = self._cache["c_pm1"]
        m_pm1 = self.m.pm1                                  # (D,)
        flat_S = S.reshape(-1, self.D)
        flat_c = c_pm1.reshape(-1, self.D)
        self.m.add_signal((flat_S * flat_c).sum(dim=0))     # q_m (D,)
        return S * m_pm1                                    # g_c (broadcast)


# ── Binary residual merge (majority of skip ⊕ transform ⊕ learned tiebreaker) ──

class ResidualMerge:
    """Gated binary residual: per-coordinate MUX between ``skip`` and ``transform``.

    A learned 1-bit gate ``g`` (D,) selects, per coordinate::

        out_d = transform_d   if g_d (open)   else   skip_d   (closed)
              = (skip & ¬g) | (transform & g)                 (fully packed)

    Initialised mostly *closed* (``p_open`` small) so a fresh block is ≈ identity
    — the codebook can still decode and the loss-signal flows straight through —
    then BOLD learns to open the gates where the transform actually helps.  This
    is the binary analogue of zero-initialised residual branches.
    """

    def __init__(self, D: int, *, name: str, p_open: float = 0.05,
                 generator: Optional[torch.Generator] = None, device=None):
        self.D = D
        self.g = random_bit_param((D,), f"{name}.g", generator=generator,
                                  device=device, p_true=p_open)
        self._cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.g]

    def forward(self, skip_bit: brute.Tensor, trans_bit: brute.Tensor) -> brute.Tensor:
        pg = self.g.bit._packed_buf
        not_pg = torch.ops.brute.bit1_not_packed(pg, self.D)
        closed = torch.bitwise_and(skip_bit._packed_buf, not_pg)
        open_ = torch.bitwise_and(trans_bit._packed_buf, pg)
        out = _BT._make_bit1_from_packed(
            torch.bitwise_or(closed, open_),
            list(skip_bit.shape),
        )
        self._cache = _LazyPM1Cache(
            {"skip_bit": skip_bit, "trans_bit": trans_bit},
            pm1_sources={
                "skip_pm1": ("skip_bit", tuple(skip_bit.shape)),
                "trans_pm1": ("trans_bit", tuple(trans_bit.shape)),
            },
        )
        return out

    def backward(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip_pm1 = self._cache["skip_pm1"]
        trans_pm1 = self._cache["trans_pm1"]
        g_pm1 = self.g.pm1                                  # (D,) +1 open / -1 closed
        open_ = g_pm1 > 0
        # Pass the signal to whichever source the gate currently selects.
        g_skip = S * (~open_)
        g_trans = S * open_
        # Gate flip-signal:  out_pm1 = (skip+trans)/2 + g_pm1·(trans-skip)/2,
        # so q_g = Σ S·(trans-skip)/2  (nonzero only where the sources disagree).
        q_g = (S * (trans_pm1 - skip_pm1) * 0.5).reshape(-1, self.D).sum(dim=0)
        self.g.add_signal(q_g)
        return g_skip, g_trans


def _broadcast_like(vec_bit: brute.Tensor, ref_bit: brute.Tensor) -> brute.Tensor:
    """Broadcast a (D,) bit1 vector to ref's logical shape on the packed buffer."""
    pv = vec_bit._packed_buf
    target = list(ref_bit.shape[:-1]) + [pv.shape[-1]]
    pv_b = pv.expand(target)
    return _BT._make_bit1_from_packed(pv_b, list(ref_bit.shape))


def _batched_similarity(q_bit: brute.Tensor, key_bit: brute.Tensor) -> torch.Tensor:
    """Batched ``<q, key>`` using packed XNOR-popcount per batch row."""
    B = int(q_bit.shape[0])
    return torch.stack(
        [brute.fast.matmul(q_bit[b], key_bit[b]) for b in range(B)],
        dim=0,
    )


# ── Token codebook (shared input embedding + output decoder) ───────────────────

class TokenCodebook:
    """One learned binary codebook ``E`` (V, D): row lookup *and* min-Hamming decode.

    ``structured`` initialises ``E`` from a Binary Equiangular Frame (BEP App. C)
    instead of random codes — maximally and uniformly separated rows, which
    removes the anomalously-close pairs that drive decode collisions.
    The decode is *stateless* (the caller passes ``chat_pm1`` back into
    :meth:`decode_backward`) so the head can decode several projections of the
    concept (lexical + semantic) against the one shared codebook.
    """

    def __init__(self, vocab_size: int, D: int, *, name: str = "E", flip_scale: float = 0.3,
                 structured: bool = False, bef_alpha: float = 1.0, bef_sweeps: int = 30,
                 generator: Optional[torch.Generator] = None, device=None):
        self.V, self.D = vocab_size, D
        if structured:
            from vsa import binary_equiangular_frame
            frame = binary_equiangular_frame(vocab_size, D, alpha=bef_alpha,
                                             n_sweeps=bef_sweeps, generator=generator)
            bit = brute.as_tensor(frame > 0, dtype=brute.bit1)
            if device is not None:
                bit = bit.to(device)
            self.E = BoldParam(bit, name=name)
        else:
            self.E = random_bit_param((vocab_size, D), name, generator=generator, device=device)
        self.E.flip_scale = flip_scale  # the tied codebook is fragile → flip it slowly
        self._embed_cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.E]

    # input side -------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> brute.Tensor:
        """``ids`` (B, n) long → (B, n, D) bit1 via packed row gather."""
        B, n = ids.shape
        flat = ids.reshape(-1).long().to(self.E.bit._packed_buf.device)
        packed = self.E.bit._packed_buf.index_select(0, flat).reshape(B, n, -1)
        self._embed_cache = {"ids": flat, "shape": (B, n)}
        return _BT._make_bit1_from_packed(packed.contiguous(), [B, n, self.D])

    def backward_embed(self, S_rows: torch.Tensor) -> None:
        """Scatter the signal at the embedding rows into ``E``'s flip-signal."""
        flat_ids = self._embed_cache["ids"]
        S_flat = S_rows.reshape(-1, self.D)
        qE = torch.zeros_like(self.E.q)
        qE.index_add_(0, flat_ids, S_flat)
        self.E.add_signal(qE)

    # output side ------------------------------------------------------------
    def decode(self, chat_bit: brute.Tensor) -> torch.Tensor:
        """``chat`` (M, D) bit1 → logits (M, V) int32 = ``<chat, E(t)>`` (stateless)."""
        return brute.fast.matmul(chat_bit, self.E.bit)     # (M, V) int32

    def decode_backward(self, S_logits: torch.Tensor, chat_pm1: torch.Tensor) -> torch.Tensor:
        """``S_logits`` (M, V), ``chat_pm1`` (M, D). Returns signal at ``chat``; updates ``E``."""
        self.E.add_signal(S_logits.transpose(0, 1) @ chat_pm1)
        return (S_logits @ self.E.pm1) * signal_scale(self.V)


# ── Latent Hopfield Bank (binary associative memory / latent attention) ────────

class HopfieldBank:
    """M learned slots — key ``P`` (M,D), payload ``U`` (M,D).  Top-k WTA read."""

    def __init__(self, D: int, n_slots: int, top_k: int, *, name: str = "hop",
                 generator: Optional[torch.Generator] = None, device=None):
        self.D, self.M, self.k = D, n_slots, top_k
        self.P = random_bit_param((n_slots, D), f"{name}.P", generator=generator, device=device)
        self.U = random_bit_param((n_slots, D), f"{name}.U", generator=generator, device=device)
        self._cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.P, self.U]

    def forward(self, q_bit: brute.Tensor) -> brute.Tensor:
        """``q`` (M_tok, D) bit1 → read (M_tok, D) bit1 = ``sign(Σ_topk U_m)``."""
        sim = brute.fast.matmul(q_bit, self.P.bit)         # (M_tok, M) int32
        kk = min(self.k, self.M)
        if kk == 1:
            idx = sim.argmax(dim=1, keepdim=True)          # (M_tok, 1)
            packed = self.U.bit._packed_buf.index_select(0, idx.squeeze(1))
            read_bit = _BT._make_bit1_from_packed(packed.contiguous(), [q_bit.shape[0], self.D])
        else:
            _, idx = torch.topk(sim, k=kk, dim=1)          # (M_tok, k)
            U_pm1 = self.U.pm1                             # (M, D)
            sel = U_pm1[idx]                               # (M_tok, k, D)
            read = sel.sum(dim=1)                          # (M_tok, D) integer vote
            read_bit = sign_to_bit1(read)
        self._cache = _LazyPM1Cache(
            {"idx": idx, "q_bit": q_bit},
            pm1_sources={"q_pm1": ("q_bit", tuple(q_bit.shape))},
        )
        return read_bit

    def backward(self, S_read: torch.Tensor) -> torch.Tensor:
        """Pass-through to selected payloads; Hebbian key update; no query signal.

        The top-k selection is piecewise-constant in the query, so the query
        receives no gradient (legitimately zero a.e.); the bank is trained by
        (a) loss-driven payload flips and (b) a Hebbian key-specialisation rule.
        """
        idx = self._cache["idx"]                           # (M_tok, k)
        q_pm1 = self._cache["q_pm1"]                       # (M_tok, D)
        Mtok, kk = idx.shape
        flat_idx = idx.reshape(-1)                         # (M_tok*k,)
        # payload flip-signal (loss-driven, pass-through through sign+sum)
        qU = torch.zeros_like(self.U.q)
        qU.index_add_(0, flat_idx,
                      S_read.unsqueeze(1).expand(Mtok, kk, self.D).reshape(-1, self.D))
        self.U.add_signal(qU)
        # Hebbian key specialisation in BOLD's minimise-loss convention:
        # a negative query signal flips only bits that disagree with the query.
        qP = torch.zeros_like(self.P.q)
        qP.index_add_(0, flat_idx,
                      -q_pm1.unsqueeze(1).expand(Mtok, kk, self.D).reshape(-1, self.D))
        self.P.add_signal(qP)
        return torch.zeros_like(S_read)


# ── Binary Episodic Slot Memory (exact in-window recall) ───────────────────────

class EpisodicSlotMemory:
    """Addressable binary slots for exact context-local recall.

    Every token writes its own slot and a later query retrieves a specific one
    by Hamming address, so copy-within-window is a lookup rather than a bundle
    decode.

    Addressing is two-component (content + position), the binary analogue of
    content + positional attention:

        score[t,m] = ⟨q^c_t, k^c_m⟩ + ⟨q^p_t, POS_m⟩,   for m < t  (read-before-write)

    where ``k^c = Kc(c)``, ``q^c = Qc(c)``, ``q^p = Qp(c)`` are dense Boolean
    projections (the address lane — never decoded) and ``POS_m`` are the absolute
    position codes.  Content addressing solves induction/marker; the positional
    query lets a trigger retrieve "the token at position p" (copy).  The payload
    is the source concept verbatim: position-free, already in the clean decode
    frame, and requiring no decode-time positional unbind.

    Forward read is **hard top-1** (exact).  The backward pass is a
    straight-through *soft-attention* surrogate so the address projections get a
    real LM-loss gradient.  :meth:`margin_loss` adds a local Hamming-margin
    objective for retrieval curricula.
    """

    def __init__(self, D: int, *, name: str = "epi", read_k: int = 1,
                 n_slots: Optional[int] = None, attn_inv_temp: Optional[float] = None,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None):
        self.D = D
        self.k = read_k
        self.N = n_slots                       # ring-buffer cap (None ⇒ one slot / token)
        self.inv_temp = attn_inv_temp if attn_inv_temp is not None else D ** -0.5
        self.Kc = BooleanLinear(D, D, name=f"{name}.Kc", generator=generator, device=device, boundary_nu=boundary_nu)
        self.Qc = BooleanLinear(D, D, name=f"{name}.Qc", generator=generator, device=device, boundary_nu=boundary_nu)
        self.Qp = BooleanLinear(D, D, name=f"{name}.Qp", generator=generator, device=device, boundary_nu=boundary_nu)
        self._cache: dict = {}
        # streaming (inference) ring-buffer state, stored as packed bit buffers
        self._s_kc = self._s_pos = self._s_pay = None
        self._s_ptr = 0
        self._s_cnt = 0

    def params(self) -> List[BoldParam]:
        return self.Kc.params() + self.Qc.params() + self.Qp.params()

    # ── batched (parallel-training) form ──────────────────────────────────────
    def forward(self, c_bit: brute.Tensor, pos_bit: brute.Tensor | None) -> brute.Tensor:
        """``c`` (B,n,D) bit1, ``pos`` (n,D) bit1 position codes → read (B,n,D) bit1."""
        B, n, D = c_bit.shape
        c_flat = c_bit.reshape(B * n, D)
        kc_bit, _ = self.Kc.forward(c_flat)
        qc_bit, _ = self.Qc.forward(c_flat)
        qp_bit, _ = self.Qp.forward(c_flat)
        kc_bit = kc_bit.reshape(B, n, D)
        qc_bit = qc_bit.reshape(B, n, D)
        qp_bit = qp_bit.reshape(B, n, D)
        if pos_bit is None:
            pos_bit = brute.ones(n, D, dtype=brute.bit1, device=c_bit.device)

        # two-component score (content + position), causal read-before-write
        content = _batched_similarity(qc_bit, kc_bit)          # (B,n,n)
        position = brute.fast.matmul(qp_bit.reshape(B * n, D), pos_bit).reshape(B, n, n)
        score = (content + position).to(torch.float32)
        t = torch.arange(n, device=c_bit.device).unsqueeze(1)
        m = torch.arange(n, device=c_bit.device).unsqueeze(0)
        causal = m < t
        if self.N is not None:
            causal = causal & ((t - m) <= self.N)
        neg = torch.finfo(score.dtype).min / 4
        score = score.masked_fill(~causal.unsqueeze(0), neg)

        kk = min(self.k, n, self.N or n)
        if kk == 1:
            topv, topi = score.max(dim=2, keepdim=True)       # argmax tie-breaks like streaming
        else:
            topv, topi = torch.topk(score, k=kk, dim=2)        # (B,n,kk)
        valid_top = topv > (neg / 2)
        no_slot = ~valid_top.any(dim=2)                        # (B,n)

        cache_data = {
            "shape": (B, n, D), "kc_bit": kc_bit, "qc_bit": qc_bit,
            "qp_bit": qp_bit, "pos_bit": pos_bit, "c_bit": c_bit,
            "score": score, "topi": topi, "no_slot": no_slot, "causal": causal,
        }
        pm1_sources = {
            "kc": ("kc_bit", (B, n, D)),
            "qc": ("qc_bit", (B, n, D)),
            "qp": ("qp_bit", (B, n, D)),
            "pos": ("pos_bit", (n, D)),
            "c_pm1": ("c_bit", (B, n, D)),
        }

        if kk == 1:
            # Exact single-slot read can gather whole packed rows directly.
            packed = c_bit._packed_buf
            Kp = int(packed.shape[-1])
            idx = topi.squeeze(-1).clamp_min(0)
            gathered = torch.gather(
                packed, 1, idx.unsqueeze(-1).expand(B, n, Kp))
            out_packed = torch.where(no_slot.unsqueeze(-1), packed, gathered)
            read_bit = _BT._make_bit1_from_packed(out_packed.contiguous(), [B, n, D])
        else:
            # Multi-slot reads still need an integer vote over payload signs.
            c_pm1 = to_pm1(c_bit)
            cache_data["c_pm1"] = c_pm1
            pm1_sources.pop("c_pm1")
            sel_pay = torch.gather(
                c_pm1.unsqueeze(1).expand(B, n, n, D), 2,
                topi.unsqueeze(-1).expand(B, n, kk, D))        # (B,n,kk,D)
            sel_pay = sel_pay * valid_top.unsqueeze(-1)
            read = sel_pay.sum(dim=2)                          # integer vote
            read = torch.where(no_slot.unsqueeze(-1), c_pm1, read)
            read_bit = sign_to_bit1(read)

        self._cache = _LazyPM1Cache(cache_data, pm1_sources=pm1_sources)
        return read_bit

    def _attn(self):
        """Softmax attention weights over the causal score (for the ST backward)."""
        score = self._cache["score"]
        return torch.softmax(score * self.inv_temp, dim=2)     # (B,n,n)

    def backward(self, S_read: torch.Tensor) -> torch.Tensor:
        """``S_read`` (B,n,D) signal at the read. Returns signal at ``c`` (B,n,D).

        Forward selection is hard (argmax); the backward uses the soft-attention
        surrogate ``a = softmax(score)`` (straight-through) so the address
        projections receive a gradient.  Three signal paths feed back into ``c``:
        the payload (value) path, and the query/key (address) paths.
        """
        B, n, D = self._cache["shape"]
        kc, qc, qp = self._cache["kc"], self._cache["qc"], self._cache["qp"]
        c_pm1, pos = self._cache["c_pm1"], self._cache["pos"]
        no_slot = self._cache["no_slot"]
        a = self._attn()                                       # (B,n,n)

        # value path: read_t = Σ_m a[t,m]·c_m  ⇒  dL/dc_m = Σ_t a[t,m]·S_read[t]
        g_val = torch.einsum("btm,btd->bmd", a, S_read)        # (B,n,D)

        # score path through softmax: dL/da[t,m] = ⟨S_read[t], c_m⟩
        dA = torch.einsum("btd,bmd->btm", S_read, c_pm1)       # (B,n,n)
        dscore = a * (dA - (a * dA).sum(dim=2, keepdim=True))  # softmax jacobian
        dscore = dscore * self.inv_temp
        # score = ⟨qc_t,kc_m⟩ + ⟨qp_t,pos_m⟩
        g_qc = torch.einsum("btm,bmd->btd", dscore, kc)
        g_kc = torch.einsum("btm,btd->bmd", dscore, qc)
        g_qp = torch.einsum("btm,md->btd", dscore, pos)

        # where a row had no valid slot the read was the identity c_t
        g_val = g_val + no_slot.unsqueeze(-1) * S_read

        g_c = g_val
        g_c = g_c + self.Kc.backward(g_kc.reshape(B * n, D)).reshape(B, n, D)
        g_c = g_c + self.Qc.backward(g_qc.reshape(B * n, D)).reshape(B, n, D)
        g_c = g_c + self.Qp.backward(g_qp.reshape(B * n, D)).reshape(B, n, D)
        return g_c

    # ── local Hamming-margin objective for the address lane ───────────────────
    def margin_loss(self, matched: torch.Tensor, *, theta_pos: float = 0.5,
                    theta_neg: float = 0.0, weight: float = 1.0) -> float:
        """Push the matched query/key pair above ``θ⁺`` and the best distractor
        below ``θ⁻`` (margins as a fraction of ``D``), and accumulate the
        resulting flip-signal onto the address projections.

        ``matched`` is ``(B,n)`` long: for each query position ``t`` the slot id
        ``m`` it *should* retrieve, or ``-1`` if there is no supervised target.
        Must be called after :meth:`forward` (uses its cache); returns the scalar
        margin loss for logging.  Trains the addresses with a local discrete
        target.
        """
        B, n, D = self._cache["shape"]
        kc, qc, qp = self._cache["kc"], self._cache["qc"], self._cache["qp"]
        pos, score = self._cache["pos"], self._cache["score"]
        tp, tn = theta_pos * D, theta_neg * D
        causal = self._cache["causal"].unsqueeze(0).expand(B, n, n)
        in_range = (matched >= 0) & (matched < n)
        safe_idx = matched.clamp(0, max(n - 1, 0))
        valid = in_range & torch.gather(causal, 2, safe_idx.unsqueeze(-1)).squeeze(-1)
        if int(valid.sum()) == 0:
            return 0.0
        m_idx = safe_idx                                      # (B,n)
        pos_score = torch.gather(score, 2, m_idx.unsqueeze(-1)).squeeze(-1)  # (B,n)
        # hardest distractor = best non-matched valid slot
        distract = score.clone()
        distract.scatter_(2, m_idx.unsqueeze(-1), torch.finfo(score.dtype).min / 4)
        neg_score, neg_idx = distract.max(dim=2)              # (B,n)

        # hinge: want pos_score ≥ tp and neg_score ≤ tn
        pos_viol = (pos_score < tp) & valid                   # push matched up
        neg_viol = (neg_score > tn) & valid                   # push distractor down
        loss = (((tp - pos_score).clamp_min(0) + (neg_score - tn).clamp_min(0)) * valid).sum()
        n_valid = int(valid.sum())
        loss_val = float(loss.item()) / max(n_valid, 1)

        # flip-signal: raise ⟨q,k⟩ for matched (−1 on score gradient sense), lower for distractor
        g_qc = torch.zeros(B, n, D, device=qc.device)
        g_kc = torch.zeros(B, n, D, device=kc.device)
        g_qp = torch.zeros(B, n, D, device=qp.device)
        scale = weight / max(n_valid, 1)
        # raise matched score: dScore>0 ⇒ signal pushes ⟨q,k⟩ up  (use −kc as the
        # "increase similarity" direction in BOLD's minimise-loss convention)
        pm = (pos_viol.float() * scale).unsqueeze(-1)         # (B,n,1)
        k_match = torch.gather(kc, 1, m_idx.unsqueeze(-1).expand(B, n, D))   # (B,n,D)
        pos_match = torch.gather(pos.unsqueeze(0).expand(B, n, D), 1,
                                 m_idx.unsqueeze(-1).expand(B, n, D))
        g_qc -= pm * k_match
        g_qp -= pm * pos_match
        g_kc.scatter_add_(1, m_idx.unsqueeze(-1).expand(B, n, D), -pm * qc)
        # lower distractor score
        nm = (neg_viol.float() * scale).unsqueeze(-1)
        k_neg = torch.gather(kc, 1, neg_idx.unsqueeze(-1).expand(B, n, D))
        pos_neg = torch.gather(pos.unsqueeze(0).expand(B, n, D), 1,
                               neg_idx.unsqueeze(-1).expand(B, n, D))
        g_qc += nm * k_neg
        g_qp += nm * pos_neg
        g_kc.scatter_add_(1, neg_idx.unsqueeze(-1).expand(B, n, D), nm * qc)

        self.Qc.backward(g_qc.reshape(B * n, D))
        self.Kc.backward(g_kc.reshape(B * n, D))
        self.Qp.backward(g_qp.reshape(B * n, D))
        return loss_val

    # ── streaming (inference) ring-buffer form ────────────────────────────────
    @torch.no_grad()
    def reset_stream(self, batch_size: int, n_slots: int, *, device=None) -> None:
        dev = device if device is not None else self.Kc.W.device
        Kp = (self.D + 63) // 64
        self._s_kc = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_pos = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_pay = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_ptr = 0
        self._s_cnt = 0

    @torch.no_grad()
    def step(self, c_bit: brute.Tensor, pos_bit: brute.Tensor, n_slots: int) -> brute.Tensor:
        """One causal step (ring buffer of ``n_slots``). ``c`` (B,D), ``pos`` (D,)."""
        if c_bit.dim() == 1:
            c_bit = c_bit.reshape(1, self.D)
        B, D = c_bit.shape
        if self._s_kc is None or self._s_kc.shape[0] != B or self._s_kc.shape[1] != n_slots:
            self.reset_stream(B, n_slots, device=c_bit.device)
        qc_bit = self.Qc.forward(c_bit)[0]
        qp_bit = self.Qp.forward(c_bit)[0]
        if self._s_cnt == 0:
            read_bit = c_bit                                   # nothing to retrieve yet
        else:
            cnt = self._s_cnt
            rows = []
            for b in range(B):
                kc = _BT._make_bit1_from_packed(
                    self._s_kc[b, :cnt].contiguous(), [cnt, D])
                pc = _BT._make_bit1_from_packed(
                    self._s_pos[b, :cnt].contiguous(), [cnt, D])
                content = brute.fast.matmul(qc_bit[b:b + 1], kc).squeeze(0)
                position = brute.fast.matmul(qp_bit[b:b + 1], pc).squeeze(0)
                rows.append(content + position)
            score = torch.stack(rows, dim=0)                  # (B,cnt)
            sel = score.argmax(dim=1)                         # (B,)
            gathered = self._s_pay[torch.arange(B, device=c_bit.device), sel]
            read_bit = _BT._make_bit1_from_packed(gathered.contiguous(), [B, D])
        # write current token into the ring buffer (after the read)
        kc_bit = self.Kc.forward(c_bit)[0]
        pos_store = pos_bit
        if not getattr(pos_store, "_is_bit1", False):
            pos_store = sign_to_bit1(pos_store.reshape(1, D)).reshape(D)
        p = self._s_ptr
        self._s_kc[:, p] = kc_bit._packed_buf
        self._s_pos[:, p] = pos_store._packed_buf.reshape(1, -1).expand(B, -1)
        self._s_pay[:, p] = c_bit._packed_buf
        self._s_ptr = (p + 1) % n_slots
        self._s_cnt = min(self._s_cnt + 1, n_slots)
        return read_bit


# ── Bundling State Recurrence (BSR) — linear-time binary sequence mixer ─────────

def pow2_decay_palette(D: int, shifts=(1, 2, 3, 4, 0), device=None) -> torch.Tensor:
    """Per-coordinate decay multiplier from a small power-of-two palette.

    Each coordinate is assigned a shift ``s`` from ``shifts`` (channel groups,
    RetNet-style multi-timescale retention); its decay multiplier is
    ``1 - 2^-s``. In the packed BSR scan this is implemented as an integer
    shift update, ``A <- A - (A >> s)``. ``s = 0`` means permanent memory
    (no decay, multiplier 1).
    """
    groups = len(shifts)
    idx = (torch.arange(D, device=device) * groups) // D            # (D,) group id
    mult = torch.empty(D, dtype=torch.float32, device=device)
    for g, s in enumerate(shifts):
        m = 1.0 if s == 0 else (1.0 - 2.0 ** (-s))
        mult[idx == g] = m
    return mult


class BSR:
    """Binary delta-corrected linear-recurrence context mixer.

    Per position: key/value/query are dense 1-bit Boolean projections of the
    *position-free* concept. BSR is the compressed discourse mixer; positional
    and exact recall are handled by the episodic slot memory.  The
    query reads the prior causal state, then the current association is written
    by an **erase-before-write delta rule**::

        S_i = sign(A_i),   r_i = q_i ⊗ S_i
        pred_i = k_i ⊗ sign(A_i)               # what the bundle already predicts for k_i
        g_i = 1[ pred_i ≠ v_i ]                 # disagreement gate (pure XOR)
        A_{i+1} = decay ⊙ A_i + g_i ⊙ (k_i ⊗ v_i)

    Writing only the error, where the bundle already disagrees, prevents
    over-counting and saturation.  ``decay`` is the power-of-two palette
    (:func:`pow2_decay_palette`). Forward is an O(n) packed scan; backward is
    the O(n) reverse-scan adjoint with sign and the hard gate treated as
    pass-through.
    """

    def __init__(self, D: int, *, name: str = "bsr",
                 decay_shifts=(1, 2, 3, 4, 0),
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None):
        self.D = D
        # K/V/Q must be genuine Boolean projections, not diagonal bindings:
        # (c⊗W_K)⊗(c⊗W_V) cancels c exactly. Dense bitwise projections preserve
        # content in the association while staying in the packed XNOR path.
        self.K = BooleanLinear(D, D, name=f"{name}.K", generator=generator, device=device, boundary_nu=boundary_nu)
        self.V = BooleanLinear(D, D, name=f"{name}.V", generator=generator, device=device, boundary_nu=boundary_nu)
        self.Q = BooleanLinear(D, D, name=f"{name}.Q", generator=generator, device=device, boundary_nu=boundary_nu)
        self.decay = pow2_decay_palette(D, decay_shifts, device=device)        # (D,)
        self.decay_shifts = torch.tensor(tuple(decay_shifts), dtype=torch.int32, device=device)
        self.decay_shift_values = tuple(sorted(set(int(s) for s in decay_shifts)))
        groups = len(decay_shifts)
        idx = (torch.arange(D, device=device) * groups) // D
        self.decay_shift_by_dim = torch.empty(D, dtype=torch.int32, device=device)
        for g, s in enumerate(decay_shifts):
            self.decay_shift_by_dim[idx == g] = int(s)
        self._cache: dict = {}
        self._stream_A: Optional[torch.Tensor] = None

    def params(self) -> List[BoldParam]:
        return self.K.params() + self.V.params() + self.Q.params()

    def forward(self, c_bit: brute.Tensor) -> brute.Tensor:
        """``c`` (B, n, D) bit1 → read ``r`` (B, n, D) bit1."""
        B, n, D = c_bit.shape
        c_flat = c_bit.reshape(B * n, D)
        k_bit, _ = self.K.forward(c_flat)
        v_bit, _ = self.V.forward(c_flat)
        q_bit, _ = self.Q.forward(c_flat)
        k_bit = k_bit.reshape(B, n, D)
        v_bit = v_bit.reshape(B, n, D)
        q_bit = q_bit.reshape(B, n, D)
        assoc_bit = bind(k_bit, v_bit)
        read_bit, S_state_bit, gate_bit = brute.fast.bsr_scan(
            q_bit, assoc_bit, self.decay_shifts.to(c_bit.device))
        self._cache = _LazyPM1Cache(
            {
                "k_bit": k_bit, "v_bit": v_bit, "q_bit": q_bit,
                "S_state_bit": S_state_bit, "gate_bit": gate_bit,
                "shape": (B, n, D),
            },
            pm1_sources={
                "k_pm1": ("k_bit", (B, n, D)),
                "v_pm1": ("v_bit", (B, n, D)),
                "q_pm1": ("q_bit", (B, n, D)),
                "S_state_pm1": ("S_state_bit", (B, n, D)),
            },
            float01_sources={"gate": ("gate_bit", (B, n, D))},
        )
        return read_bit

    def backward(self, S_r: torch.Tensor) -> torch.Tensor:
        """``S_r`` (B, n, D) signal at the read. Returns signal at ``c`` (B, n, D)."""
        B, n, D = self._cache["shape"]
        k_pm1, v_pm1, q_pm1 = self._cache["k_pm1"], self._cache["v_pm1"], self._cache["q_pm1"]
        S_state_pm1 = self._cache["S_state_pm1"]
        gate = self._cache["gate"]
        decay = self.decay

        # r_i = q_i ⊗ S_i  →  signal to q_i and to the state S_i
        g_q = S_r * S_state_pm1                              # (B,n,D)
        gS = S_r * q_pm1                                     # signal entering sign(A_i)

        # reverse-scan adjoint of the delta recurrence:
        #   S_i = sign(A_i),  A_{i+1} = decay·A_i + gate_i ⊙ assoc_i
        g_assoc = torch.empty_like(gS)
        Abar = torch.zeros(B, D, dtype=torch.float32, device=S_r.device)
        for i in range(n - 1, -1, -1):
            g_assoc[:, i, :] = gate[:, i, :] * Abar          # dL/dassoc_i (gate is pass-through)
            Abar = gS[:, i, :] + decay * Abar                # dL/dA_i

        # assoc_i = k_i ⊗ v_i
        g_k = g_assoc * v_pm1
        g_v = g_assoc * k_pm1

        # Dense Boolean projections handle both param flip-signals and upstream
        # concept signals through their existing BOLD backward path.
        g_c = self.K.backward(g_k.reshape(B * n, D))
        g_c += self.V.backward(g_v.reshape(B * n, D))
        g_c += self.Q.backward(g_q.reshape(B * n, D))
        return g_c.reshape(B, n, D)

    @torch.no_grad()
    def reset_stream(self, batch_size: int, *, device=None) -> None:
        """Reset the recurrent inference state for streaming BSR reads."""
        dev = device if device is not None else self.decay.device
        self._stream_A = torch.zeros(batch_size, self.D, dtype=torch.int32, device=dev)

    @torch.no_grad()
    def step(self, c_bit: brute.Tensor) -> brute.Tensor:
        """One causal streaming step. ``c_bit`` is ``(B,D)`` and returns ``(B,D)``."""
        if c_bit.dim() == 1:
            c_bit = c_bit.reshape(1, self.D)
        B, D = c_bit.shape
        if self._stream_A is None or self._stream_A.shape[0] != B:
            self.reset_stream(B, device=c_bit.device)
        k_bit, _ = self.K.forward(c_bit)
        v_bit, _ = self.V.forward(c_bit)
        q_bit, _ = self.Q.forward(c_bit)
        assoc_bit = bind(k_bit, v_bit)
        state_bool = self._stream_A >= 0
        r = bind(q_bit, sign_to_bit1(self._stream_A))
        assoc_bool = assoc_bit.bool().as_subclass(torch.Tensor)
        gate = state_bool != assoc_bool
        shifts = self.decay_shift_by_dim.to(c_bit.device)
        decayed = self._stream_A.clone()
        for s in self.decay_shift_values:
            if s > 0:
                mask = shifts == s
                decayed[:, mask] = self._stream_A[:, mask] - (self._stream_A[:, mask] >> s)
        update = torch.where(gate, torch.where(assoc_bool, 1, -1), 0).to(torch.int32)
        self._stream_A = decayed + update
        return r
