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

    def forward(self, a_bit: brute.Tensor) -> Tuple[brute.Tensor, Optional[torch.Tensor]]:
        """Returns ``(out_bit, z_or_None)``.

        When ``boundary_nu`` is ``None`` (the common packed path) the fused
        ``xnor_popcount_matmul_sign`` kernel is used and ``z`` is never
        materialised — the returned second element is ``None``.  When
        ``boundary_nu`` is set the int32 pre-activation ``z`` is required for
        the eligibility mask, so the standard ``matmul + sign`` path is kept.
        """
        if self.boundary_nu is None:
            out_bit = brute.fast.matmul_sign(a_bit, self.W.bit, self.in_dim)
            self._cache = _LazyPM1Cache(
                {"a_bit": a_bit},
                pm1_sources={"a_pm1": ("a_bit", tuple(a_bit.shape))},
            )
            return out_bit, None
        else:
            z = brute.fast.matmul(a_bit, self.W.bit)   # (M, out) int32
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
            _, idx = torch.topk(sim, k=kk, dim=1)         # (M_tok, k)
            # Bit-sliced majority vote — no int32 tally, no unpack.
            # Gather k packed payload rows → (M_tok, k, Kp), then majority.
            sel_packed = self.U.bit._packed_buf[idx.reshape(-1)].reshape(
                q_bit.shape[0], kk, -1)
            rows_bit = _BT._make_bit1_from_packed(
                sel_packed.contiguous(), [q_bit.shape[0], kk, self.D])
            read_bit = brute.fast.majority(rows_bit, kk, self.D)
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


# ── Binary Episodic Slot Memory (two-tier chunked, linear-time) ────────────────

class EpisodicSlotMemory:
    """Two-tier causal memory for exact local recall and bounded global recall.

    **Tier 1 — exact local recall (O(n·C)):**
    Each token in chunk ``g`` attends causally over the 2C-token window spanning
    chunks ``g-1`` and ``g``.  This is chunk-parallel (no O(n²) dense matrix).
    When ``n ≤ C`` the sequence is a single chunk and degenerates exactly to the
    previous full intra-sequence causal top-1 recall behaviour.

    **Tier 2 — bounded global register cache (O(n·S), optional):**
    A fixed bank of ``S`` register slots per batch element carries salient tokens
    from chunks older than Tier-1's 2C span, enabling beyond-window induction.
    Registers are written by LSH last-writer-wins after each completed chunk.
    Tier-2 read is a content-only Hamming search over the S slots; the higher
    score of the two candidates (Tier-1 vs Tier-2) wins.  When ``S=0`` the bank
    is disabled and the behaviour is pure Tier-1.

    **Backward:**
    Tier-1 uses a soft-attention (softmax surrogate) backward windowed to the
    2C chunk size.  Tier-2 has no key gradient (hard piecewise-constant
    selection); the value path routes signal to the source token's concept.
    Address projections Kc/Qc/Qp get gradient from Tier-1 and margin_loss.
    """

    def __init__(self, D: int, *, name: str = "epi", read_k: int = 1,
                 n_slots: Optional[int] = None, attn_inv_temp: Optional[float] = None,
                 epi_chunk: int = 64, epi_registers: int = 64,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None):
        self.D = D
        self.k = read_k
        self.N = n_slots                        # streaming Tier-1 window cap
        self.C = epi_chunk                      # chunk size
        self.S = epi_registers                  # Tier-2 register slots (0 = disabled)
        self.inv_temp = attn_inv_temp if attn_inv_temp is not None else D ** -0.5
        self.Kc = BooleanLinear(D, D, name=f"{name}.Kc", generator=generator,
                                device=device, boundary_nu=boundary_nu)
        self.Qc = BooleanLinear(D, D, name=f"{name}.Qc", generator=generator,
                                device=device, boundary_nu=boundary_nu)
        self.Qp = BooleanLinear(D, D, name=f"{name}.Qp", generator=generator,
                                device=device, boundary_nu=boundary_nu)
        self._cache: dict = {}
        # streaming state (packed ring buffers)
        self._s_kc = self._s_pos = self._s_pay = None
        self._s_ptr = 0
        self._s_cnt = 0
        # streaming Tier-2 registers
        self._r_kc = self._r_pay = self._r_valid = None

    def params(self) -> List[BoldParam]:
        return self.Kc.params() + self.Qc.params() + self.Qp.params()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _banded_sim(q_bit: brute.Tensor, k_bit: brute.Tensor) -> torch.Tensor:
        """(B, q_cnt, D) × (B, k_cnt, D) → (B, q_cnt, k_cnt) int32."""
        B, q_cnt, D = q_bit.shape
        k_cnt = int(k_bit.shape[1])
        return torch.stack([
            brute.fast.matmul(q_bit[b], k_bit[b])
            for b in range(B)
        ], dim=0)

    @staticmethod
    def _reg_sim(q_bit: brute.Tensor, reg_packed: torch.Tensor, D: int) -> torch.Tensor:
        """(B, q_cnt, D) bit1 × (B, S, Kp) int64 → (B, q_cnt, S) int32."""
        B, q_cnt, _ = q_bit.shape
        S = reg_packed.shape[1]
        return torch.stack([
            brute.fast.matmul(
                q_bit[b],
                _BT._make_bit1_from_packed(reg_packed[b].contiguous(), [S, D]))
            for b in range(B)
        ], dim=0)

    # ── batched (parallel-training) form ──────────────────────────────────────

    def forward(self, c_bit: brute.Tensor, pos_bit: Optional[brute.Tensor]) -> brute.Tensor:
        """``c`` (B,n,D) bit1, ``pos`` (n,D) bit1 → read (B,n,D) bit1.

        Processes the sequence in chunks of size C.  When n ≤ C this is a
        single chunk and the result is identical to the old full-sequence causal
        read (Tier-2 inactive for the first chunk).
        """
        B, n, D = c_bit.shape
        C = self.C
        S = self.S
        G = (n + C - 1) // C
        Kp = int(c_bit._packed_buf.shape[-1])
        dev = c_bit.device
        neg = torch.finfo(torch.float32).min / 4

        if pos_bit is None:
            pos_bit = brute.ones(n, D, dtype=brute.bit1, device=dev)

        # Project all tokens in one batched call.
        c_flat = c_bit.reshape(B * n, D)
        kc_bit, _ = self.Kc.forward(c_flat)
        qc_bit, _ = self.Qc.forward(c_flat)
        qp_bit, _ = self.Qp.forward(c_flat)
        kc_bit = kc_bit.reshape(B, n, D)
        qc_bit = qc_bit.reshape(B, n, D)
        qp_bit = qp_bit.reshape(B, n, D)

        pay_packed = c_bit._packed_buf              # (B, n, Kp)
        out_packed = torch.zeros(B, n, Kp, dtype=torch.int64, device=dev)

        # Per-chunk bookkeeping for the backward.
        chunk_scores: List[torch.Tensor] = []       # (B, q_cnt, w_cnt) per chunk
        chunk_ranges: List[Tuple[int, int, int, int]] = []
        chunk_no_slot: List[torch.Tensor] = []      # (B, q_cnt) bool per chunk

        # Tier-2 register state (only allocated when S > 0).
        reg_kc:     Optional[torch.Tensor] = None   # (B, S, Kp) int64
        reg_pay:    Optional[torch.Tensor] = None   # (B, S, Kp) int64
        reg_src:    Optional[torch.Tensor] = None   # (B, S) int64 source position
        reg_valid:  Optional[torch.Tensor] = None   # (B, S) bool
        if S > 0:
            reg_kc    = torch.zeros(B, S, Kp, dtype=torch.int64, device=dev)
            reg_pay   = torch.zeros(B, S, Kp, dtype=torch.int64, device=dev)
            reg_src   = torch.full((B, S), -1, dtype=torch.int64, device=dev)
            reg_valid = torch.zeros(B, S, dtype=torch.bool, device=dev)

        # Full-sequence Tier-2 bookkeeping for the backward value path.
        t2_used = torch.zeros(B, n, dtype=torch.bool, device=dev)
        t2_src  = torch.full((B, n), -1, dtype=torch.int64, device=dev)

        for g in range(G):
            q_start = g * C
            q_end   = min((g + 1) * C, n)
            q_cnt   = q_end - q_start
            k_start = max(0, (g - 1) * C)
            k_end   = q_end              # keys up to end of current chunk
            w_cnt   = k_end - k_start

            qc_chunk = qc_bit[:, q_start:q_end, :]   # (B, q_cnt, D) bit1
            qp_chunk = qp_bit[:, q_start:q_end, :]
            kc_win   = kc_bit[:, k_start:k_end, :]   # (B, w_cnt, D) bit1
            pos_win  = pos_bit[k_start:k_end, :]      # (w_cnt, D) bit1

            # Tier-1 scores (B, q_cnt, w_cnt)
            content  = self._banded_sim(qc_chunk, kc_win)
            position = brute.fast.matmul(
                qp_chunk.reshape(B * q_cnt, D), pos_win).reshape(B, q_cnt, w_cnt)
            score = (content + position).to(torch.float32)

            # Causal mask: key at global position k_start+j is visible to query
            # at global position q_start+i only if k_start+j < q_start+i.
            q_glob = torch.arange(q_start, q_end, device=dev)  # (q_cnt,)
            k_glob = torch.arange(k_start, k_end, device=dev)  # (w_cnt,)
            causal  = k_glob.unsqueeze(0) < q_glob.unsqueeze(1)  # (q_cnt, w_cnt)
            if self.N is not None:
                in_win = (q_glob.unsqueeze(1) - k_glob.unsqueeze(0)) <= self.N
                causal = causal & in_win
            score = score.masked_fill(~causal.unsqueeze(0), neg)

            topv, topi = score.max(dim=2)             # (B, q_cnt)
            valid_t1 = topv > (neg / 2)
            no_slot_t1 = ~valid_t1                    # (B, q_cnt)

            # Gather Tier-1 payload.
            t1_pay = torch.gather(
                pay_packed[:, k_start:k_end, :], 1,
                topi.clamp_min(0).unsqueeze(-1).expand(B, q_cnt, Kp))  # (B, q_cnt, Kp)

            # Tier-2 read — active from chunk g=1 onwards (registers populated
            # after chunk 0 is folded in at the end of g=0's iteration).
            use_t2 = torch.zeros(B, q_cnt, dtype=torch.bool, device=dev)
            if S > 0 and g >= 1 and reg_valid.any():
                gscore = self._reg_sim(qc_chunk, reg_kc, D).float()  # (B, q_cnt, S)
                gscore.masked_fill_(~reg_valid.unsqueeze(1).expand(B, q_cnt, S), neg)
                gt2v, gt2i = gscore.max(dim=2)                        # (B, q_cnt)
                valid_t2 = gt2v > (neg / 2)
                use_t2 = valid_t2 & (gt2v > topv)                     # prefer higher score

                # Gather Tier-2 payload.
                gt2i_cl = gt2i.clamp_min(0)
                t2_pay = torch.gather(
                    reg_pay, 1,
                    gt2i_cl.unsqueeze(-1).expand(B, q_cnt, Kp))       # (B, q_cnt, Kp)

                # Record source positions for the backward value path.
                t2_src_chunk = torch.gather(reg_src, 1, gt2i_cl)      # (B, q_cnt)
                t2_src_chunk = torch.where(use_t2, t2_src_chunk,
                                           torch.full_like(t2_src_chunk, -1))

                out_chunk = torch.where(
                    use_t2.unsqueeze(-1),
                    t2_pay,
                    torch.where(no_slot_t1.unsqueeze(-1),
                                pay_packed[:, q_start:q_end, :], t1_pay))

                t2_used[:, q_start:q_end] = use_t2
                t2_src[:, q_start:q_end]  = t2_src_chunk
            else:
                out_chunk = torch.where(
                    no_slot_t1.unsqueeze(-1),
                    pay_packed[:, q_start:q_end, :], t1_pay)

            out_packed[:, q_start:q_end, :] = out_chunk

            chunk_scores.append(score)
            chunk_ranges.append((q_start, q_end, k_start, k_end))
            chunk_no_slot.append(no_slot_t1)

            # Fold chunk g-1 tokens into Tier-2 registers so they are available
            # as global context for chunk g+1 and beyond.
            if S > 0 and g >= 1:
                fold_s = (g - 1) * C
                fold_e = min(g * C, n)
                fold_n = fold_e - fold_s

                n_bk   = max(1, (S - 1).bit_length())  # ceil(log2(S))
                bk_mask = (1 << n_bk) - 1

                kc_fold_w0 = kc_bit._packed_buf[:, fold_s:fold_e, 0]  # (B, fold_n) int64
                buckets    = (kc_fold_w0 & bk_mask).long() % S         # (B, fold_n)
                src_idx    = torch.arange(fold_s, fold_e, device=dev
                                         ).unsqueeze(0).expand(B, -1)  # (B, fold_n)

                bk_kp = buckets.unsqueeze(-1).expand(B, fold_n, Kp)
                reg_kc.scatter_(1, bk_kp, kc_bit._packed_buf[:, fold_s:fold_e, :])
                reg_pay.scatter_(1, bk_kp, pay_packed[:, fold_s:fold_e, :])
                reg_src.scatter_(1, buckets, src_idx)
                reg_valid.scatter_(
                    1, buckets,
                    torch.ones(B, fold_n, dtype=torch.bool, device=dev))

            # On the very last chunk, also fold chunk G-1 (= current chunk g)
            # into registers so a hypothetical next chunk could use them.
            # (Does not affect this forward pass but keeps register state correct
            # for streaming continuations; only relevant when G >= 1.)
            if S > 0 and g == G - 1 and G > 0:
                fold_s = g * C
                fold_e = n
                fold_n = fold_e - fold_s
                if fold_n > 0:
                    n_bk   = max(1, (S - 1).bit_length())
                    bk_mask = (1 << n_bk) - 1
                    kc_fold_w0 = kc_bit._packed_buf[:, fold_s:fold_e, 0]
                    buckets    = (kc_fold_w0 & bk_mask).long() % S
                    src_idx    = torch.arange(fold_s, fold_e, device=dev
                                             ).unsqueeze(0).expand(B, -1)
                    bk_kp = buckets.unsqueeze(-1).expand(B, fold_n, Kp)
                    reg_kc.scatter_(1, bk_kp, kc_bit._packed_buf[:, fold_s:fold_e, :])
                    reg_pay.scatter_(1, bk_kp, pay_packed[:, fold_s:fold_e, :])
                    reg_src.scatter_(1, buckets, src_idx)
                    reg_valid.scatter_(
                        1, buckets,
                        torch.ones(B, fold_n, dtype=torch.bool, device=dev))

        # Tier-1 no_slot: positions where Tier-1 found nothing.
        no_slot_t1_full = torch.cat(chunk_no_slot, dim=1)  # (B, n)
        # Final no_slot: Tier-1 found nothing AND Tier-2 not used.
        no_slot_full = no_slot_t1_full & ~t2_used

        # Assemble cache for backward.
        cache_data = {
            "shape": (B, n, D), "kc_bit": kc_bit, "qc_bit": qc_bit,
            "qp_bit": qp_bit, "pos_bit": pos_bit, "c_bit": c_bit,
            "chunk_scores": chunk_scores, "chunk_ranges": chunk_ranges,
            "no_slot": no_slot_full, "t2_used": t2_used, "t2_src": t2_src,
        }
        pm1_sources = {
            "kc":    ("kc_bit", (B, n, D)),
            "qc":    ("qc_bit", (B, n, D)),
            "qp":    ("qp_bit", (B, n, D)),
            "pos":   ("pos_bit", (n, D)),
            "c_pm1": ("c_bit",  (B, n, D)),
        }
        self._cache = _LazyPM1Cache(cache_data, pm1_sources=pm1_sources)
        return _BT._make_bit1_from_packed(out_packed.contiguous(), [B, n, D])

    def backward(self, S_read: torch.Tensor) -> torch.Tensor:
        """``S_read`` (B,n,D) → signal at ``c`` (B,n,D).

        Tier-1: windowed soft-attention (softmax surrogate) per chunk.
        Tier-2: hard value pass-through to source token; no key gradient.
        """
        B, n, D = self._cache["shape"]
        kc, qc, qp = self._cache["kc"], self._cache["qc"], self._cache["qp"]
        c_pm1  = self._cache["c_pm1"]
        pos    = self._cache["pos"]
        no_slot = self._cache["no_slot"]
        chunk_scores = self._cache["chunk_scores"]
        chunk_ranges = self._cache["chunk_ranges"]
        t2_used = self._cache["t2_used"]
        t2_src  = self._cache["t2_src"]

        g_val = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)
        g_qc  = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)
        g_kc  = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)
        g_qp  = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)

        for score_g, (q_start, q_end, k_start, k_end) in zip(chunk_scores, chunk_ranges):
            q_cnt = q_end - q_start
            a_g = torch.softmax(score_g * self.inv_temp, dim=2)  # (B, q_cnt, w_cnt)

            S_g       = S_read[:, q_start:q_end, :]              # (B, q_cnt, D)
            c_win     = c_pm1[:, k_start:k_end, :]               # (B, w_cnt, D)
            kc_win    = kc[:, k_start:k_end, :]
            pos_win   = pos[k_start:k_end, :]
            qc_chunk  = qc[:, q_start:q_end, :]
            qp_chunk  = qp[:, q_start:q_end, :]

            # Value path: dL/dc_m = Σ_t a[t,m]·S[t]
            g_val[:, k_start:k_end, :] += torch.einsum("bqw,bqd->bwd", a_g, S_g)

            # Score path (softmax Jacobian).
            dA      = torch.einsum("bqd,bwd->bqw", S_g, c_win)
            dscore  = a_g * (dA - (a_g * dA).sum(dim=2, keepdim=True))
            dscore  = dscore * self.inv_temp

            g_qc[:, q_start:q_end, :] += torch.einsum("bqw,bwd->bqd", dscore, kc_win)
            g_kc[:, k_start:k_end, :] += torch.einsum("bqw,bqd->bwd", dscore, qc_chunk)
            g_qp[:, q_start:q_end, :] += torch.einsum("bqw,wd->bqd",  dscore, pos_win)

        # Identity path: no valid slot → read was c_t itself.
        g_val = g_val + no_slot.unsqueeze(-1) * S_read

        # Tier-2 value path: route signal from Tier-2 queries to their source.
        if t2_used.any():
            t2_sig = S_read * t2_used.unsqueeze(-1).to(S_read.dtype)  # (B, n, D)
            src    = t2_src.clamp_min(0).unsqueeze(-1).expand(B, n, D)
            g_val.scatter_add_(1, src, t2_sig)

        g_c = g_val
        g_c = g_c + self.Kc.backward(g_kc.reshape(B * n, D)).reshape(B, n, D)
        g_c = g_c + self.Qc.backward(g_qc.reshape(B * n, D)).reshape(B, n, D)
        g_c = g_c + self.Qp.backward(g_qp.reshape(B * n, D)).reshape(B, n, D)
        return g_c

    # ── local Hamming-margin objective (windowed Tier-1 only) ─────────────────

    def margin_loss(self, matched: torch.Tensor, *, theta_pos: float = 0.5,
                    theta_neg: float = 0.0, weight: float = 1.0) -> float:
        """Hinge margin over the Tier-1 window — trains address projections.

        ``matched`` (B,n) long: for each query position ``t`` the global token
        index it *should* retrieve, or ``-1`` if unsupervised.  A matched slot
        outside the Tier-1 window is automatically skipped (the Tier-2 path has
        no margin signal; its value gradient comes from the backward).
        """
        B, n, D = self._cache["shape"]
        kc = self._cache["kc"]
        qc = self._cache["qc"]
        qp = self._cache["qp"]
        pos = self._cache["pos"]

        tp, tn = theta_pos * D, theta_neg * D
        total_loss = 0.0
        n_valid_total = 0
        scale = weight

        g_qc = torch.zeros(B, n, D, device=qc.device)
        g_kc = torch.zeros(B, n, D, device=kc.device)
        g_qp = torch.zeros(B, n, D, device=qp.device)

        for score_g, (q_start, q_end, k_start, k_end) in zip(
                self._cache["chunk_scores"], self._cache["chunk_ranges"]):
            q_cnt = q_end - q_start
            w_cnt = k_end - k_start

            matched_g = matched[:, q_start:q_end]  # (B, q_cnt) global slot idx
            # Adjust to window-relative index.
            rel_g     = matched_g - k_start          # (B, q_cnt) offset into window
            in_range  = (rel_g >= 0) & (rel_g < w_cnt) & (matched_g >= 0)
            if not in_range.any():
                continue

            safe_rel  = rel_g.clamp(0, max(w_cnt - 1, 0))
            # Must also be causally reachable: key < query.
            q_glob    = torch.arange(q_start, q_end, device=matched.device)
            k_at_safe = k_start + safe_rel  # (B, q_cnt) global key position
            causal_ok = k_at_safe < q_glob.unsqueeze(0)
            valid     = in_range & causal_ok

            if not valid.any():
                continue

            n_valid = int(valid.sum())
            n_valid_total += n_valid
            s = scale / max(n_valid, 1)

            pos_score = torch.gather(score_g, 2, safe_rel.unsqueeze(-1)).squeeze(-1)  # (B, q_cnt)
            distract  = score_g.clone()
            distract.scatter_(2, safe_rel.unsqueeze(-1),
                              torch.finfo(score_g.dtype).min / 4)
            neg_score, neg_idx = distract.max(dim=2)

            pos_viol = (pos_score < tp) & valid
            neg_viol = (neg_score > tn) & valid
            chunk_loss = (
                ((tp - pos_score).clamp_min(0) + (neg_score - tn).clamp_min(0))
                * valid.float()
            ).sum()
            total_loss += float(chunk_loss.item())

            kc_win  = kc[:, k_start:k_end, :]
            qc_ch   = qc[:, q_start:q_end, :]
            pos_win = pos[k_start:k_end, :]

            pm = (pos_viol.float() * s).unsqueeze(-1)
            k_match   = torch.gather(kc_win, 1, safe_rel.unsqueeze(-1).expand(B, q_cnt, D))
            pos_match = torch.gather(
                pos_win.unsqueeze(0).expand(B, w_cnt, D), 1,
                safe_rel.unsqueeze(-1).expand(B, q_cnt, D))
            g_qc[:, q_start:q_end, :] -= pm * k_match
            g_qp[:, q_start:q_end, :] -= pm * pos_match
            g_kc[:, k_start:k_end, :].scatter_add_(
                1, safe_rel.unsqueeze(-1).expand(B, q_cnt, D), -pm * qc_ch)

            nm = (neg_viol.float() * s).unsqueeze(-1)
            k_neg   = torch.gather(kc_win, 1, neg_idx.unsqueeze(-1).expand(B, q_cnt, D))
            pos_neg = torch.gather(
                pos_win.unsqueeze(0).expand(B, w_cnt, D), 1,
                neg_idx.unsqueeze(-1).expand(B, q_cnt, D))
            g_qc[:, q_start:q_end, :] += nm * k_neg
            g_qp[:, q_start:q_end, :] += nm * pos_neg
            g_kc[:, k_start:k_end, :].scatter_add_(
                1, neg_idx.unsqueeze(-1).expand(B, q_cnt, D), nm * qc_ch)

        self.Qc.backward(g_qc.reshape(B * n, D))
        self.Kc.backward(g_kc.reshape(B * n, D))
        self.Qp.backward(g_qp.reshape(B * n, D))
        return total_loss / max(n_valid_total, 1)

    # ── streaming (inference) ring-buffer form ────────────────────────────────

    @torch.no_grad()
    def reset_stream(self, batch_size: int, n_slots: int, *, device=None) -> None:
        """Reset Tier-1 ring buffer and Tier-2 registers."""
        dev = device if device is not None else self.Kc.W.device
        Kp = (self.D + 63) // 64
        self._s_kc  = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_pos = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_pay = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_ptr = 0
        self._s_cnt = 0
        self._s_step = 0    # total steps processed; gates Tier-2 reads at step >= C
        if self.S > 0:
            self._r_kc    = torch.zeros(batch_size, self.S, Kp, dtype=torch.int64, device=dev)
            self._r_pay   = torch.zeros(batch_size, self.S, Kp, dtype=torch.int64, device=dev)
            self._r_valid = torch.zeros(batch_size, self.S, dtype=torch.bool, device=dev)

    @torch.no_grad()
    def step(self, c_bit: brute.Tensor, pos_bit: brute.Tensor, n_slots: int) -> brute.Tensor:
        """One causal streaming step.  ``c`` (B,D), ``pos`` (D,) → read (B,D) bit1.

        ``n_slots`` caps the Tier-1 ring buffer.  Use ``2*C`` for full
        streaming parity with the chunked batched forward.

        Tier-2 reads are gated at ``step_count >= C`` (mirroring the batched
        forward where Tier-2 is first active at chunk g=1).
        """
        if c_bit.dim() == 1:
            c_bit = c_bit.reshape(1, self.D)
        B, D = c_bit.shape
        if (self._s_kc is None or self._s_kc.shape[0] != B
                or self._s_kc.shape[1] != n_slots):
            self.reset_stream(B, n_slots, device=c_bit.device)

        Kp = (D + 63) // 64
        qc_bit = self.Qc.forward(c_bit)[0]
        qp_bit = self.Qp.forward(c_bit)[0]

        if self._s_cnt == 0:
            read_bit = c_bit        # nothing to retrieve yet
        else:
            cnt_buf = torch.full((B,), self._s_cnt, dtype=torch.int32, device=c_bit.device)
            read_t1, _, score_t1 = brute.fast.episodic_causal_search(
                qc_bit, self._s_kc, qp_bit, self._s_pos, self._s_pay, cnt_buf, D)

            # Tier-2 read: gated at step >= C to match batched-forward chunk logic.
            use_t2_read = (self.S > 0
                           and getattr(self, "_s_step", 0) >= self.C
                           and self._r_kc is not None
                           and self._r_valid.any())
            if use_t2_read:
                neg_v = float(-(D * 2 + 2))
                best_t2_scores = torch.full((B,), neg_v, device=c_bit.device)
                best_t2_pay   = torch.zeros(B, Kp, dtype=torch.int64, device=c_bit.device)
                for b in range(B):
                    if not self._r_valid[b].any():
                        continue
                    reg_b = _BT._make_bit1_from_packed(
                        self._r_kc[b].contiguous(), [self.S, D])
                    sims = brute.fast.matmul(qc_bit[b:b + 1], reg_b).squeeze(0).float()
                    sims.masked_fill_(~self._r_valid[b], neg_v)
                    best_idx = int(sims.argmax().item())
                    if float(sims[best_idx].item()) > neg_v / 2:
                        best_t2_scores[b] = sims[best_idx]
                        best_t2_pay[b]    = self._r_pay[b, best_idx]

                use_t2 = best_t2_scores > score_t1.float()   # (B,) bool
                out_pk = torch.where(
                    use_t2.unsqueeze(-1),
                    best_t2_pay, read_t1._packed_buf)
                read_bit = _BT._make_bit1_from_packed(out_pk.contiguous(), [B, D])
            else:
                read_bit = read_t1

        # Write into Tier-1 ring buffer (read-before-write).
        kc_bit = self.Kc.forward(c_bit)[0]
        pos_store = pos_bit
        if not getattr(pos_store, "_is_bit1", False):
            pos_store = sign_to_bit1(pos_store.reshape(1, D)).reshape(D)
        p = self._s_ptr
        self._s_kc[:, p]  = kc_bit._packed_buf
        self._s_pos[:, p] = pos_store._packed_buf.reshape(1, -1).expand(B, -1)
        self._s_pay[:, p] = c_bit._packed_buf
        self._s_ptr = (p + 1) % n_slots
        self._s_cnt = min(self._s_cnt + 1, n_slots)

        # Write into Tier-2 registers (LSH last-writer-wins).
        if self.S > 0 and self._r_kc is not None:
            n_bk    = max(1, (self.S - 1).bit_length())
            bk_mask = (1 << n_bk) - 1
            kc_w0   = kc_bit._packed_buf[:, 0]
            buckets = (kc_w0 & bk_mask).long() % self.S
            for b in range(B):
                bkt = int(buckets[b])
                self._r_kc[b, bkt]    = kc_bit._packed_buf[b]
                self._r_pay[b, bkt]   = c_bit._packed_buf[b]
                self._r_valid[b, bkt] = True

        self._s_step = getattr(self, "_s_step", 0) + 1
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
