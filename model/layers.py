"""HÆMMR v3 Boolean layers — bit1 forward + BOLD-signal backward.

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

v3 notes
--------
* The four block branches (delta-BSR, episodic ring, Hopfield priors, channel
  mix) each read the *same* block-input concept and emit a binary proposal.  The
  block does **not** majority-merge them; the model's clipped integer concept
  highway (see ``model.Block``) accumulates the proposals.  There is therefore
  no ``ResidualMerge`` in v3 — the majority-vote residual merge is explicitly
  forbidden as the main inter-layer transport.
* :class:`EpisodicSlotMemory` is a single canonical fixed-width causal ring with
  read-before-write semantics.  Batched training is a chunked implementation of
  that *same* ring search (no dense ``(B, n, n)`` score matrix, no Tier-2 LSH
  register cache — exact recall is window-bounded, as the spec requires).
* :class:`BSR` is a decoupled erase/write delta editor (separate erase and write
  strengths over channel groups) over a power-of-two decay palette.
* :class:`TokenCodebook` holds a **fixed** lexical frame ``E_lex`` (used for both
  the input embedding and the min-Hamming lexical decode) plus an optional
  **learned** semantic prototype bank ``E_sem`` used only for the rerank.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

import brute
from brute.tensor import Tensor as _BT

from bold import BoldParam, random_bit_param, signal_scale
from vsa import bind, sign_to_bit1, to_i8_pm1, to_pm1


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


def _logic_signal(S: torch.Tensor) -> torch.Tensor:
    """Project a numeric downstream signal to BOLD's ternary logic value."""
    if S.is_floating_point():
        return torch.sign(S)
    return S


def _decode_logic_signal(S: torch.Tensor) -> torch.Tensor:
    """Sparse ternary signal for decoder/classification logits.

    CE-style rows contain one negative target component and many tiny positive
    non-target components. Native Boolean variation should not turn all of
    those non-targets into equal votes, so each row keeps only the strongest
    positive and strongest negative direction.
    """
    if not S.is_floating_point() or S.dim() != 2:
        return _logic_signal(S)
    out = torch.zeros_like(S)
    pos = S > 0
    neg = S < 0
    if pos.any():
        pos_vals = S.masked_fill(~pos, torch.finfo(S.dtype).min)
        pos_idx = pos_vals.argmax(dim=1, keepdim=True)
        has_pos = pos.any(dim=1, keepdim=True)
        out.scatter_(1, pos_idx, has_pos.to(S.dtype))
    if neg.any():
        neg_vals = S.masked_fill(~neg, torch.finfo(S.dtype).max)
        neg_idx = neg_vals.argmin(dim=1, keepdim=True)
        has_neg = neg.any(dim=1, keepdim=True)
        out.scatter_(1, neg_idx, -has_neg.to(S.dtype))
    return out


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
        a_bit = self._cache["a_bit"]
        if self.boundary_nu is not None:
            z = self._cache["z"].reshape(S.shape)
            elig = (z.abs() <= self.boundary_nu * self.in_dim).to(S.dtype)   # BEP Eq. 5
            S = S * elig
        S = _logic_signal(S)
        if a_bit.device.type == "cpu" and self.W.bit.device.type == "cpu":
            S_votes = S.to(torch.int32) if not S.is_floating_point() else S.to(torch.int8)
            # Eq. 7 — q_W = aggregate_k xnor(S_kj, a_ki), as integer votes.
            self.W.add_signal(brute.fast.ternary_matmul(
                S_votes.transpose(0, 1).contiguous(), a_bit, self.in_dim))
            # Eq. 8 — upstream variation votes through packed W.
            return brute.fast.ternary_matmul(S_votes.contiguous(), self.W.bit, self.in_dim)

        a_pm1 = self._cache["a_pm1"]
        self.W.add_signal(S.transpose(0, 1) @ a_pm1)
        return (S @ self.W.pm1) * signal_scale(self.out_dim)


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
        S = _logic_signal(S)
        c_pm1 = self._cache["c_pm1"]
        m_pm1 = self.m.pm1                                  # (D,)
        flat_S = S.reshape(-1, self.D)
        flat_c = c_pm1.reshape(-1, self.D)
        self.m.add_signal((flat_S * flat_c).sum(dim=0))     # q_m (D,)
        return S * m_pm1                                    # g_c (broadcast)


# ── Token codebook: fixed lexical frame + optional learned semantic bank ────────

class TokenCodebook:
    """Fixed lexical codebook ``E_lex`` (V,D) and optional learned ``E_sem`` (V,D).

    v3 keeps the **lexical codes fixed after initialisation** (a stable, balanced,
    maximally-separated target geometry simplifies discrete credit assignment).
    ``structured`` initialises ``E_lex`` from a Binary Equiangular Frame
    (BEP App. C); otherwise it is a random balanced frame.  The same fixed frame
    serves both the input embedding lookup and the min-Hamming lexical decode.

    ``E_sem`` is a *learned* per-token semantic prototype bank used **only** for
    the rerank term ``β·ham(c^L, E_sem[t])``; it is trained by BOLD.  When the
    rerank is disabled (``sem=False``) no semantic bank is allocated.
    """

    def __init__(self, vocab_size: int, D: int, *, name: str = "E",
                 structured: bool = True, bef_alpha: float = 1.0, bef_sweeps: int = 30,
                 sem: bool = True, sem_flip_scale: float = 0.5,
                 generator: Optional[torch.Generator] = None, device=None):
        self.V, self.D = vocab_size, D
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if structured:
            from vsa import binary_equiangular_frame
            frame = binary_equiangular_frame(vocab_size, D, alpha=bef_alpha,
                                             n_sweeps=bef_sweeps, generator=generator)
            bits = frame > 0
        else:
            bits = (torch.rand((vocab_size, D), generator=generator) < 0.5)
        self.E_lex = brute.as_tensor(bits, dtype=brute.bit1).to(self.device)   # FIXED
        self._E_lex_pm1: Optional[torch.Tensor] = None                          # lazy ±1
        self.sem = sem
        if sem:
            self.E_sem = random_bit_param((vocab_size, D), f"{name}.sem",
                                          generator=generator, device=device)
            self.E_sem.flip_scale = sem_flip_scale
        else:
            self.E_sem = None
        self._embed_cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.E_sem] if self.sem else []

    @property
    def E_lex_pm1(self) -> torch.Tensor:
        if self._E_lex_pm1 is None:
            self._E_lex_pm1 = to_pm1(self.E_lex)
        return self._E_lex_pm1

    # input side -------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> brute.Tensor:
        """``ids`` (B, n) long → (B, n, D) bit1 via packed row gather (fixed frame)."""
        B, n = ids.shape
        flat = ids.reshape(-1).long().to(self.E_lex._packed_buf.device)
        packed = self.E_lex._packed_buf.index_select(0, flat).reshape(B, n, -1)
        return _BT._make_bit1_from_packed(packed.contiguous(), [B, n, self.D])

    # output side (lexical, fixed) -------------------------------------------
    def decode(self, ell_bit: brute.Tensor) -> torch.Tensor:
        """``ell`` (M, D) bit1 → logits (M, V) int32 = ``<ell, E_lex(t)>`` (stateless)."""
        return brute.fast.matmul(ell_bit, self.E_lex)      # (M, V) int32

    def decode_backward(self, S_logits: torch.Tensor, ell_pm1: torch.Tensor) -> torch.Tensor:
        """Lexical decode backward.  ``E_lex`` is fixed → returns signal at ``ell`` only."""
        S_logits = _decode_logic_signal(S_logits)
        if self.E_lex.device.type == "cpu":
            S_votes = S_logits.to(torch.int32) if not S_logits.is_floating_point() else S_logits.to(torch.int8)
            return brute.fast.ternary_matmul(S_votes.contiguous(), self.E_lex, self.D)
        return (S_logits @ self.E_lex_pm1) * signal_scale(self.V)

    # output side (semantic, learned) ----------------------------------------
    def decode_sem(self, c_bit: brute.Tensor) -> torch.Tensor:
        """``c`` (M, D) bit1 → semantic logits (M, V) int32 = ``<c, E_sem(t)>``."""
        return brute.fast.matmul(c_bit, self.E_sem.bit)

    def decode_sem_backward(self, S_logits: torch.Tensor, c_pm1: torch.Tensor) -> torch.Tensor:
        """Semantic rerank backward.  Trains ``E_sem`` and returns signal at ``c``."""
        S_logits = _decode_logic_signal(S_logits)
        S_votes = S_logits.to(torch.int32) if not S_logits.is_floating_point() else S_logits.to(torch.int8)
        if getattr(c_pm1, "_is_bit1", False) and self.E_sem.bit.device.type == "cpu":
            if c_pm1.dim() != 2:
                c_pm1 = c_pm1.reshape(-1, self.D)
            self.E_sem.add_signal(brute.fast.ternary_matmul(
                S_votes.transpose(0, 1).contiguous(), c_pm1, self.D))
            return brute.fast.ternary_matmul(S_votes.contiguous(), self.E_sem.bit, self.D)
        if getattr(c_pm1, "_is_bit1", False):
            if c_pm1.dim() != 2:
                c_pm1 = c_pm1.reshape(-1, self.D)
            c_pm1 = to_pm1(c_pm1)
        self.E_sem.add_signal(S_logits.transpose(0, 1) @ c_pm1)
        return (S_logits @ self.E_sem.pm1) * signal_scale(self.V)

    # checkpoint -------------------------------------------------------------
    def state_dict(self) -> dict:
        sd = {"E_lex_packed": self.E_lex._packed_buf.detach().cpu().clone(),
              "shape": (self.V, self.D)}
        if self.sem:
            sd["E_sem"] = self.E_sem.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        packed = sd["E_lex_packed"].to(self.device)
        self.E_lex = _BT._make_bit1_from_packed(packed, list(sd["shape"]))
        self._E_lex_pm1 = None
        if self.sem and "E_sem" in sd:
            self.E_sem.load_state_dict(sd["E_sem"])


# ── Latent Hopfield Bank (binary associative memory / latent attention) ────────

class HopfieldBank:
    """M learned slots — key ``P`` (M,D), payload ``U`` (M,D).  Top-k WTA read.

    Static learned priors only (v3): the Hopfield bank holds reusable conceptual
    priors/schemas and is **not** a runtime write target.  Exact local recall is
    the episodic ring's job; compressed discourse is BSR's job.
    """

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
        S_read = _logic_signal(S_read)
        idx = self._cache["idx"]                           # (M_tok, k)
        q_pm1 = self._cache["q_pm1"]                       # (M_tok, D)
        Mtok, kk = idx.shape
        flat_idx = idx.reshape(-1)                         # (M_tok*k,)
        # payload flip-signal (loss-driven, pass-through through sign+sum)
        qU = torch.zeros(self.U.shape, dtype=S_read.dtype, device=S_read.device)
        qU.index_add_(0, flat_idx,
                      S_read.unsqueeze(1).expand(Mtok, kk, self.D).reshape(-1, self.D))
        self.U.add_signal(qU)
        # Hebbian key specialisation in BOLD's minimise-loss convention:
        # a negative query signal flips only bits that disagree with the query.
        qP = torch.zeros(self.P.shape, dtype=q_pm1.dtype, device=q_pm1.device)
        qP.index_add_(0, flat_idx,
                      -q_pm1.unsqueeze(1).expand(Mtok, kk, self.D).reshape(-1, self.D))
        self.P.add_signal(qP)
        return torch.zeros_like(S_read)


# ── Binary Episodic Slot Memory (canonical fixed-width causal ring) ─────────────

class EpisodicSlotMemory:
    """Fixed-width causal ring memory for exact local recall (linear-time).

    There is **one** canonical exact-memory algorithm: a fixed-width causal ring
    over the last ``W`` slots with read-before-write semantics and a direct
    Hamming search over the active window — used by both batched training and
    streaming inference.

    **Batched training (O(n·C)):** the sequence is processed in chunks of size
    ``C``.  Each token in chunk ``g`` searches causally over the ``≤2C`` window
    spanning chunks ``g-1`` and ``g`` (further bounded to width ``W`` when set).
    This is a chunked implementation of the same ring search — no dense
    ``(B, n, n)`` score matrix is ever formed.  When ``n ≤ C`` the whole
    sequence is one chunk.

    **Backward:** a windowed soft-attention (softmax surrogate) over each chunk
    supplies the address-projection gradient for the hard top-1 forward read;
    the value path routes the read signal back to the matched source concept.
    The address lane is additionally shaped by :meth:`margin_loss`.
    """

    def __init__(self, D: int, *, name: str = "epi", read_k: int = 1,
                 n_slots: Optional[int] = None, attn_inv_temp: Optional[float] = None,
                 epi_chunk: int = 64,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None):
        self.D = D
        self.k = read_k
        self.N = n_slots                        # fixed ring width W (None ⇒ unbounded by C)
        self.C = epi_chunk                      # chunk size
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

    def params(self) -> List[BoldParam]:
        return self.Kc.params() + self.Qc.params() + self.Qp.params()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _banded_sim(q_bit: brute.Tensor, k_bit: brute.Tensor) -> torch.Tensor:
        """(B, q_cnt, D) × (B, k_cnt, D) → (B, q_cnt, k_cnt) int32."""
        B = int(q_bit.shape[0])
        return torch.stack([
            brute.fast.matmul(q_bit[b], k_bit[b])
            for b in range(B)
        ], dim=0)

    # ── batched (parallel-training) form ──────────────────────────────────────

    def forward(self, c_bit: brute.Tensor, pos_bit: Optional[brute.Tensor]) -> brute.Tensor:
        """``c`` (B,n,D) bit1, ``pos`` (n,D) bit1 → read (B,n,D) bit1.

        Chunked causal ring search.  When n ≤ C this is a single chunk and the
        result is the full-sequence causal top-1 read.
        """
        B, n, D = c_bit.shape
        C = self.C
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

            content  = self._banded_sim(qc_chunk, kc_win)
            position = brute.fast.matmul(
                qp_chunk.reshape(B * q_cnt, D), pos_win).reshape(B, q_cnt, w_cnt)
            score = (content + position).to(torch.float32)

            # Causal mask + fixed ring width: key at global position k_start+j is
            # visible to query at q_start+i only if it is strictly earlier and
            # within W positions.
            q_glob = torch.arange(q_start, q_end, device=dev)  # (q_cnt,)
            k_glob = torch.arange(k_start, k_end, device=dev)  # (w_cnt,)
            causal  = k_glob.unsqueeze(0) < q_glob.unsqueeze(1)  # (q_cnt, w_cnt)
            if self.N is not None:
                in_win = (q_glob.unsqueeze(1) - k_glob.unsqueeze(0)) <= self.N
                causal = causal & in_win
            score = score.masked_fill(~causal.unsqueeze(0), neg)

            topv, topi = score.max(dim=2)             # (B, q_cnt)
            no_slot = ~(topv > (neg / 2))             # (B, q_cnt)

            t1_pay = torch.gather(
                pay_packed[:, k_start:k_end, :], 1,
                topi.clamp_min(0).unsqueeze(-1).expand(B, q_cnt, Kp))  # (B, q_cnt, Kp)

            out_chunk = torch.where(
                no_slot.unsqueeze(-1),
                pay_packed[:, q_start:q_end, :], t1_pay)
            out_packed[:, q_start:q_end, :] = out_chunk

            chunk_scores.append(score)
            chunk_ranges.append((q_start, q_end, k_start, k_end))
            chunk_no_slot.append(no_slot)

        no_slot_full = torch.cat(chunk_no_slot, dim=1)  # (B, n)

        cache_data = {
            "shape": (B, n, D), "kc_bit": kc_bit, "qc_bit": qc_bit,
            "qp_bit": qp_bit, "pos_bit": pos_bit, "c_bit": c_bit,
            "chunk_scores": chunk_scores, "chunk_ranges": chunk_ranges,
            "no_slot": no_slot_full,
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

        Windowed soft-attention (softmax surrogate) supplies the address-lane
        gradient; the value path routes the read signal to the matched source.
        """
        B, n, D = self._cache["shape"]
        kc, qc, qp = self._cache["kc"], self._cache["qc"], self._cache["qp"]
        c_pm1  = self._cache["c_pm1"]
        pos    = self._cache["pos"]
        no_slot = self._cache["no_slot"]
        chunk_scores = self._cache["chunk_scores"]
        chunk_ranges = self._cache["chunk_ranges"]

        g_val = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)
        g_qc  = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)
        g_kc  = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)
        g_qp  = torch.zeros(B, n, D, dtype=S_read.dtype, device=S_read.device)

        for score_g, (q_start, q_end, k_start, k_end) in zip(chunk_scores, chunk_ranges):
            a_g = torch.softmax(score_g * self.inv_temp, dim=2)  # (B, q_cnt, w_cnt)

            S_g       = S_read[:, q_start:q_end, :]              # (B, q_cnt, D)
            c_win     = c_pm1[:, k_start:k_end, :]               # (B, w_cnt, D)
            kc_win    = kc[:, k_start:k_end, :]
            pos_win   = pos[k_start:k_end, :]
            qc_chunk  = qc[:, q_start:q_end, :]

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

        g_c = g_val
        g_c = g_c + self.Kc.backward(g_kc.reshape(B * n, D)).reshape(B, n, D)
        g_c = g_c + self.Qc.backward(g_qc.reshape(B * n, D)).reshape(B, n, D)
        g_c = g_c + self.Qp.backward(g_qp.reshape(B * n, D)).reshape(B, n, D)
        return g_c

    # ── local Hamming-margin objective (address lane) ─────────────────────────

    def margin_loss(self, matched: torch.Tensor, *, theta_pos: float = 0.5,
                    theta_neg: float = 0.0, weight: float = 1.0) -> float:
        """Hinge margin over the ring window — trains the address projections.

        ``matched`` (B,n) long: for each query position ``t`` the global token
        index it *should* retrieve, or ``-1`` if unsupervised.  A matched slot
        outside the active window is automatically skipped.
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
            rel_g     = matched_g - k_start          # (B, q_cnt) offset into window
            in_range  = (rel_g >= 0) & (rel_g < w_cnt) & (matched_g >= 0)
            if not in_range.any():
                continue

            safe_rel  = rel_g.clamp(0, max(w_cnt - 1, 0))
            q_glob    = torch.arange(q_start, q_end, device=matched.device)
            k_at_safe = k_start + safe_rel  # (B, q_cnt) global key position
            causal_ok = k_at_safe < q_glob.unsqueeze(0)
            pos_score = torch.gather(score_g, 2, safe_rel.unsqueeze(-1)).squeeze(-1)  # (B, q_cnt)
            # Only supervise matched pairs inside the active causal ring window:
            # a matched slot masked out of the score carries the -inf fill, which
            # must not become a (huge) hinge violation.
            in_score  = pos_score > (torch.finfo(score_g.dtype).min / 8)
            valid     = in_range & causal_ok & in_score

            if not valid.any():
                continue

            n_valid = int(valid.sum())
            n_valid_total += n_valid
            s = scale / max(n_valid, 1)
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
        """Reset the causal ring buffer."""
        dev = device if device is not None else self.Kc.W.device
        Kp = (self.D + 63) // 64
        self._s_kc  = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_pos = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_pay = torch.zeros(batch_size, n_slots, Kp, dtype=torch.int64, device=dev)
        self._s_ptr = 0
        self._s_cnt = 0

    @torch.no_grad()
    def step(self, c_bit: brute.Tensor, pos_bit: brute.Tensor, n_slots: int) -> brute.Tensor:
        """One causal streaming step.  ``c`` (B,D), ``pos`` (D,) → read (B,D) bit1.

        ``n_slots`` is the fixed ring width ``W``.  Read-before-write: the current
        token cannot retrieve itself.
        """
        if c_bit.dim() == 1:
            c_bit = c_bit.reshape(1, self.D)
        B, D = c_bit.shape
        if (self._s_kc is None or self._s_kc.shape[0] != B
                or self._s_kc.shape[1] != n_slots):
            self.reset_stream(B, n_slots, device=c_bit.device)

        qc_bit = self.Qc.forward(c_bit)[0]
        qp_bit = self.Qp.forward(c_bit)[0]

        if self._s_cnt == 0:
            read_bit = c_bit        # nothing to retrieve yet
        else:
            cnt_buf = torch.full((B,), self._s_cnt, dtype=torch.int32, device=c_bit.device)
            read_bit, _, _ = brute.fast.episodic_causal_search(
                qc_bit, self._s_kc, qp_bit, self._s_pos, self._s_pay, cnt_buf, D)

        # Write into the ring buffer (read-before-write).
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
        return read_bit


# ── Bundling State Recurrence (BSR) — delta erase/write linear-time mixer ────────

def pow2_decay_palette(D: int, shifts=(1, 2, 3, 4, 0), device=None) -> torch.Tensor:
    """Per-coordinate decay multiplier from a small power-of-two palette.

    Each coordinate is assigned a shift ``s`` from ``shifts`` (channel groups,
    RetNet-style multi-timescale retention); its decay multiplier is
    ``1 - 2^-s``.  In the scan this is implemented as an integer shift update,
    ``A <- A - (A >> s)``.  ``s = 0`` means permanent memory (multiplier 1).
    """
    groups = len(shifts)
    idx = (torch.arange(D, device=device) * groups) // D            # (D,) group id
    mult = torch.empty(D, dtype=torch.float32, device=device)
    for g, s in enumerate(shifts):
        m = 1.0 if s == 0 else (1.0 - 2.0 ** (-s))
        mult[idx == g] = m
    return mult


def _per_group_int(D: int, values, device=None, *, dtype=torch.int8) -> torch.Tensor:
    """Assign one integer per channel group → a (D,) int tensor (group palette)."""
    groups = len(values)
    idx = (torch.arange(D, device=device) * groups) // D
    out = torch.empty(D, dtype=dtype, device=device)
    for g, v in enumerate(values):
        out[idx == g] = int(v)
    return out


class BSR:
    """Binary delta erase/write linear-recurrence context mixer.

    Per position the key/value/query are dense 1-bit Boolean projections of the
    *position-free* concept.  BSR is the compressed discourse mixer; exact recall
    and position live in the episodic ring.  The recurrence reads the prior
    causal state, then **edits its own prediction** with a decoupled erase + write
    delta over channel groups::

        S_i   = sign(A_i)                       # read-before-write state
        r_i   = q_i ⊗ S_i                       # read
        pred  = k_i ⊗ S_i                       # current estimate for this key
        d_i   = 1[ pred ≠ v_i ]                 # disagreement gate (pure XOR)
        A_dec = decay_palette ⊙ A_i             # power-of-two decay
        A_{i+1} = clip( A_dec − erase·d_i·S_i + write·d_i·(k_i⊗v_i) )

    The **erase** term pulls disagreeing coordinates off their stale value; the
    **write** term adds the corrected association.  ``erase`` and ``write`` are
    separate per-channel-group strengths.  Where the bundle already agrees the
    write does not fire, preventing over-counting and saturation.  Forward is an
    O(n) packed scan; backward is the O(n) reverse-scan adjoint with ``sign`` and
    the hard gate treated as pass-through.
    """

    def __init__(self, D: int, *, name: str = "bsr",
                 decay_shifts=(1, 2, 3, 4, 0),
                 erase=(1,), write=(1,), state_clip: int = 31,
                 generator: Optional[torch.Generator] = None, device=None,
                 boundary_nu: Optional[float] = None):
        self.D = D
        # K/V/Q must be genuine Boolean projections, not diagonal bindings:
        # (c⊗W_K)⊗(c⊗W_V) cancels c exactly.  Dense bitwise projections preserve
        # content in the association while staying in the packed XNOR path.
        self.K = BooleanLinear(D, D, name=f"{name}.K", generator=generator, device=device, boundary_nu=boundary_nu)
        self.V = BooleanLinear(D, D, name=f"{name}.V", generator=generator, device=device, boundary_nu=boundary_nu)
        self.Q = BooleanLinear(D, D, name=f"{name}.Q", generator=generator, device=device, boundary_nu=boundary_nu)
        if state_clip <= 0 or state_clip > 127:
            raise ValueError("BSR state_clip must fit in int8 (1..127)")
        if any(abs(int(v)) > 127 for v in tuple(erase) + tuple(write)):
            raise ValueError("BSR erase/write strengths must fit in int8")
        self.decay = pow2_decay_palette(D, decay_shifts, device=device)        # (D,) float multiplier
        self.decay_shift_values = tuple(sorted(set(int(s) for s in decay_shifts)))
        groups = len(decay_shifts)
        idx = (torch.arange(D, device=device) * groups) // D
        self.decay_shift_by_dim = torch.empty(D, dtype=torch.int32, device=device)
        for g, s in enumerate(decay_shifts):
            self.decay_shift_by_dim[idx == g] = int(s)
        # Separate per-channel-group erase / write strengths.
        self.erase_by_dim = _per_group_int(D, erase, device=device)            # (D,) int8
        self.write_by_dim = _per_group_int(D, write, device=device)            # (D,) int8
        self.state_clip = int(state_clip)
        self._cache: dict = {}
        self._stream_A: Optional[torch.Tensor] = None

    def params(self) -> List[BoldParam]:
        return self.K.params() + self.V.params() + self.Q.params()

    def _decay(self, A: torch.Tensor) -> torch.Tensor:
        """Apply the power-of-two decay palette to the integer accumulator."""
        out = A.clone()
        shifts = self.decay_shift_by_dim.to(A.device)
        for s in self.decay_shift_values:
            if s > 0:
                mask = shifts == s
                out[..., mask] = A[..., mask] - (A[..., mask] >> s)
        return out

    def forward(self, c_bit: brute.Tensor) -> brute.Tensor:
        """``c`` (B, n, D) bit1 → read ``r`` (B, n, D) bit1."""
        B, n, D = c_bit.shape
        dev = c_bit.device
        c_flat = c_bit.reshape(B * n, D)
        k_bit, _ = self.K.forward(c_flat)
        v_bit, _ = self.V.forward(c_flat)
        q_bit, _ = self.Q.forward(c_flat)
        k_bit = k_bit.reshape(B, n, D)
        v_bit = v_bit.reshape(B, n, D)
        q_bit = q_bit.reshape(B, n, D)
        assoc_bit = bind(k_bit, v_bit)

        if dev.type == "cpu":
            read_bit, state_bit, gate_bit = brute.fast.bsr_delta_scan(
                q_bit, assoc_bit, self.decay_shift_by_dim,
                self.erase_by_dim, self.write_by_dim, self.state_clip)
            self._cache = _LazyPM1Cache(
                {
                    "k_bit": k_bit, "v_bit": v_bit, "q_bit": q_bit,
                    "state_bit": state_bit, "gate_bit": gate_bit,
                    "shape": (B, n, D),
                },
                pm1_sources={
                    "k_pm1": ("k_bit", (B, n, D)),
                    "v_pm1": ("v_bit", (B, n, D)),
                    "q_pm1": ("q_bit", (B, n, D)),
                    "S_state": ("state_bit", (B, n, D)),
                },
                float01_sources={"gate": ("gate_bit", (B, n, D))},
            )
            return read_bit

        k_pm1 = to_i8_pm1(k_bit).reshape(B, n, D)
        v_pm1 = to_i8_pm1(v_bit).reshape(B, n, D)
        q_pm1 = to_i8_pm1(q_bit).reshape(B, n, D)
        assoc = (k_pm1 * v_pm1)                          # (B,n,D) int8 ±1

        erase = self.erase_by_dim.to(dev)
        write = self.write_by_dim.to(dev)
        clip  = self.state_clip

        A = torch.zeros(B, D, dtype=torch.int8, device=dev)
        S_state = torch.empty(B, n, D, dtype=torch.int8, device=dev)      # sign(A_i) per step
        gate    = torch.empty(B, n, D, dtype=torch.int8, device=dev)      # disagreement
        read_pm1 = torch.empty(B, n, D, dtype=torch.int8, device=dev)
        for i in range(n):
            S = torch.where(A >= 0, 1, -1).to(torch.int8)  # read-before-write state
            S_state[:, i, :] = S
            read_pm1[:, i, :] = q_pm1[:, i, :] * S
            d = (S != assoc[:, i, :]).to(torch.int8)     # disagreement gate
            gate[:, i, :] = d
            A_dec = self._decay(A)
            A_new = A_dec - erase * d * S + write * d * assoc[:, i, :]
            A = A_new.clamp_(-clip, clip).to(torch.int8)

        read_bit = sign_to_bit1(read_pm1.reshape(B * n, D)).reshape(B, n, D)
        self._cache = {
            "k_pm1": k_pm1, "v_pm1": v_pm1, "q_pm1": q_pm1,
            "S_state": S_state, "gate": gate, "shape": (B, n, D),
        }
        return read_bit

    def backward(self, S_r: torch.Tensor) -> torch.Tensor:
        """``S_r`` (B, n, D) signal at the read. Returns signal at ``c`` (B, n, D)."""
        B, n, D = self._cache["shape"]
        k_pm1 = self._cache["k_pm1"].to(S_r.dtype)
        v_pm1 = self._cache["v_pm1"].to(S_r.dtype)
        q_pm1 = self._cache["q_pm1"].to(S_r.dtype)
        S_state = self._cache["S_state"].to(S_r.dtype)
        gate = self._cache["gate"].to(S_r.dtype)
        decay = self.decay
        write_mag = self.write_by_dim.to(S_r.device).float()   # write coefficient

        # r_i = q_i ⊗ S_i  →  signal to q_i and to the state S_i
        g_q = S_r * S_state                                 # (B,n,D)
        gS = S_r * q_pm1                                    # signal entering sign(A_i)

        # reverse-scan adjoint of the delta recurrence (sign + hard gate treated
        # as pass-through; the write term carries the assoc gradient).
        g_assoc = torch.empty_like(gS)
        Abar = torch.zeros(B, D, dtype=torch.float32, device=S_r.device)
        for i in range(n - 1, -1, -1):
            g_assoc[:, i, :] = write_mag * gate[:, i, :] * Abar
            Abar = gS[:, i, :] + decay * Abar

        g_k = g_assoc * v_pm1
        g_v = g_assoc * k_pm1

        g_c = self.K.backward(g_k.reshape(B * n, D))
        g_c += self.V.backward(g_v.reshape(B * n, D))
        g_c += self.Q.backward(g_q.reshape(B * n, D))
        return g_c.reshape(B, n, D)

    @torch.no_grad()
    def reset_stream(self, batch_size: int, *, device=None) -> None:
        """Reset the recurrent inference state for streaming BSR reads."""
        dev = device if device is not None else self.decay.device
        self._stream_A = torch.zeros(batch_size, self.D, dtype=torch.int8, device=dev)

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
        A = self._stream_A
        S = torch.where(A >= 0, 1, -1).to(torch.int8)
        r = bind(q_bit, sign_to_bit1(A))
        assoc_pm1 = to_i8_pm1(assoc_bit)
        d = (S != assoc_pm1).to(torch.int8)
        erase = self.erase_by_dim.to(A.device)
        write = self.write_by_dim.to(A.device)
        A_dec = self._decay(A)
        A_new = A_dec - erase * d * S + write * d * assoc_pm1
        self._stream_A = A_new.clamp_(-self.state_clip, self.state_clip).to(torch.int8)
        return r
