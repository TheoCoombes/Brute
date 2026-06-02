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
from vsa import bind, sign_to_bit1, to_bit1, to_pm1


# ── packed broadcast binding (c ⊗ mask) ───────────────────────────────────────

def bind_mask(c_bit: brute.Tensor, mask_bit: brute.Tensor) -> brute.Tensor:
    """XNOR-bind ``c`` (..., D) with a single mask vector ``mask`` (D,), packed.

    Broadcasts on the packed buffers — no unpack to bool.
    """
    pc = c_bit._packed_buf
    pm = mask_bit._packed_buf
    xored = torch.bitwise_xor(pc, pm).contiguous()
    inv = torch.ops.brute.bit1_not_packed(xored, int(c_bit.shape[-1]))
    return _BT._make_bit1_from_packed(inv, list(c_bit.shape))


# ── Boolean linear (XNOR matmul + sign) ────────────────────────────────────────

class BooleanLinear:
    """``z = a · Wᵀ`` (XNOR/popcount), ``out = sign(z)``.  Weight ``W`` is (out, in)."""

    def __init__(self, in_dim: int, out_dim: int, *, name: str,
                 generator: Optional[torch.Generator] = None, device=None):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.W = random_bit_param((out_dim, in_dim), f"{name}.W",
                                  generator=generator, device=device)
        self._cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.W]

    def forward(self, a_bit: brute.Tensor) -> Tuple[brute.Tensor, torch.Tensor]:
        z = brute.fast.matmul(a_bit, self.W.bit)     # (M, out) int32, signed ±1 dot
        self._cache = {"a_pm1": to_pm1(a_bit)}       # (M, in)
        return sign_to_bit1(z), z

    def backward(self, S: torch.Tensor) -> torch.Tensor:
        """``S``: (M, out) signal at the output activation. Returns (M, in)."""
        a_pm1 = self._cache["a_pm1"]
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
        self._cache = {"c_pm1": to_pm1(c_bit)}
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
        g_bit = _broadcast_like(self.g.bit, skip_bit)
        closed = brute.fast.bitwise_and(skip_bit, brute.fast.bitwise_not(g_bit))
        open_ = brute.fast.bitwise_and(trans_bit, g_bit)
        out = brute.fast.bitwise_or(closed, open_)
        self._cache = {"skip_pm1": to_pm1(skip_bit), "trans_pm1": to_pm1(trans_bit)}
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
    pv_b = pv.expand(target).contiguous()
    return _BT._make_bit1_from_packed(pv_b, list(ref_bit.shape))


# ── Token codebook (shared input embedding + output decoder) ───────────────────

class TokenCodebook:
    """One learned binary codebook ``E`` (V, D): row lookup *and* min-Hamming decode."""

    def __init__(self, vocab_size: int, D: int, *, name: str = "E", flip_scale: float = 0.3,
                 generator: Optional[torch.Generator] = None, device=None):
        self.V, self.D = vocab_size, D
        self.E = random_bit_param((vocab_size, D), name, generator=generator, device=device)
        self.E.flip_scale = flip_scale  # the tied codebook is fragile → flip it slowly
        self._embed_cache: dict = {}
        self._decode_cache: dict = {}

    def params(self) -> List[BoldParam]:
        return [self.E]

    # input side -------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> brute.Tensor:
        """``ids`` (B, n) long → (B, n, D) bit1 (row gather on the packed buffer)."""
        B, n = ids.shape
        flat = ids.reshape(-1).long()
        rows = self.E.bit[flat]                            # (B*n, D) bit1
        self._embed_cache = {"ids": flat, "shape": (B, n)}
        return rows.reshape(B, n, self.D) if rows.dim() == 2 else rows

    def backward_embed(self, S_rows: torch.Tensor) -> None:
        """Scatter the signal at the embedding rows into ``E``'s flip-signal."""
        flat_ids = self._embed_cache["ids"]
        S_flat = S_rows.reshape(-1, self.D)
        qE = torch.zeros_like(self.E.q)
        qE.index_add_(0, flat_ids, S_flat)
        self.E.add_signal(qE)

    # output side ------------------------------------------------------------
    def decode(self, chat_bit: brute.Tensor) -> torch.Tensor:
        """``chat`` (M, D) bit1 → logits (M, V) int32 = ``<chat, E(t)>``."""
        logits = brute.fast.matmul(chat_bit, self.E.bit)   # (M, V) int32
        self._decode_cache = {"chat_pm1": to_pm1(chat_bit)}
        return logits

    def backward_decode(self, S_logits: torch.Tensor) -> torch.Tensor:
        """``S_logits`` (M, V). Returns signal at ``chat`` (M, D); updates ``E``."""
        chat_pm1 = self._decode_cache["chat_pm1"]
        # q_E from the output head:  (V, D) = S_logitsᵀ · chat_pm1
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
        _, idx = torch.topk(sim, k=kk, dim=1)              # (M_tok, k)
        U_pm1 = self.U.pm1                                 # (M, D)
        sel = U_pm1[idx]                                   # (M_tok, k, D)
        read = sel.sum(dim=1)                              # (M_tok, D) integer vote
        self._cache = {"idx": idx, "q_pm1": to_pm1(q_bit)}
        return sign_to_bit1(read)

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
        # Hebbian: keys drift toward the queries that select them.
        qP = torch.zeros_like(self.P.q)
        qP.index_add_(0, flat_idx,
                      q_pm1.unsqueeze(1).expand(Mtok, kk, self.D).reshape(-1, self.D))
        self.P.add_signal(qP)
        return torch.zeros_like(S_read)


# ── Bundling State Recurrence (BSR) — linear-time binary sequence mixer ─────────

class BSR:
    """Binary linear-recurrence context mixer (HÆMMR §5.1).

    Per position: key/value/query are dense 1-bit Boolean projections of the
    concept.  The query reads the prior causal state, then the current
    key/value association is written for later positions::

        S_i = sign(A_i),   r_i = q_i ⊗ S_i,
        A_{i+1} = γ·A_i + (k_i ⊗ v_i)

    Forward is an O(n) scan; backward is the exact O(n) reverse-scan adjoint of
    the linear recurrence (sign/decay treated as pass-through).  ``γ`` is a fixed
    per-coordinate decay (the paper's single low-precision concession).
    """

    def __init__(self, D: int, *, name: str = "bsr",
                 gamma_min: float = 0.90, gamma_max: float = 0.999,
                 generator: Optional[torch.Generator] = None, device=None):
        self.D = D
        # K/V/Q must be genuine Boolean projections, not diagonal bindings:
        # (c⊗W_K)⊗(c⊗W_V) cancels c exactly. Dense bitwise projections preserve
        # content in the association while staying in the packed XNOR path.
        self.K = BooleanLinear(D, D, name=f"{name}.K", generator=generator, device=device)
        self.V = BooleanLinear(D, D, name=f"{name}.V", generator=generator, device=device)
        self.Q = BooleanLinear(D, D, name=f"{name}.Q", generator=generator, device=device)
        # Multi-timescale fixed decay spread geometrically across coordinates.
        frac = torch.linspace(0, 1, D, device=device)
        self.gamma = (gamma_min * (gamma_max / gamma_min) ** frac).to(torch.float32)  # (D,)
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
        k_pm1 = to_pm1(k_bit).reshape(B, n, D)
        v_pm1 = to_pm1(v_bit).reshape(B, n, D)
        q_pm1 = to_pm1(q_bit).reshape(B, n, D)
        assoc_pm1 = k_pm1 * v_pm1                           # (B, n, D) bound association

        gamma = self.gamma
        A = torch.zeros(B, D, dtype=torch.float32, device=c_bit.device)
        S_state_pm1 = torch.empty(B, n, D, dtype=torch.float32, device=c_bit.device)
        for i in range(n):
            S_state_pm1[:, i, :] = torch.where(A >= 0, 1.0, -1.0)
            A = gamma * A + assoc_pm1[:, i, :]
        r_pm1 = q_pm1 * S_state_pm1                          # r_i = q_i ⊗ S_i
        self._cache = {
            "k_pm1": k_pm1, "v_pm1": v_pm1, "q_pm1": q_pm1,
            "S_state_pm1": S_state_pm1, "shape": (B, n, D),
        }
        return to_bit1(r_pm1)

    def backward(self, S_r: torch.Tensor) -> torch.Tensor:
        """``S_r`` (B, n, D) signal at the read. Returns signal at ``c`` (B, n, D)."""
        B, n, D = self._cache["shape"]
        k_pm1, v_pm1, q_pm1 = self._cache["k_pm1"], self._cache["v_pm1"], self._cache["q_pm1"]
        S_state_pm1 = self._cache["S_state_pm1"]
        gamma = self.gamma

        # r_i = q_i ⊗ S_i  →  signal to q_i and to the state S_i
        g_q = S_r * S_state_pm1                              # (B,n,D)
        gS = S_r * q_pm1                                     # signal entering sign(A_i)

        # reverse-scan adjoint of causal read-before-write:
        #   read_i = sign(A_i),  A_{i+1} = γ·A_i + assoc_i
        g_assoc = torch.empty_like(gS)
        Abar_next = torch.zeros(B, D, dtype=torch.float32, device=S_r.device)
        for i in range(n - 1, -1, -1):
            g_assoc[:, i, :] = Abar_next
            Abar_next = gS[:, i, :] + gamma * Abar_next

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
        dev = device if device is not None else self.gamma.device
        self._stream_A = torch.zeros(batch_size, self.D, dtype=torch.float32, device=dev)

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
        k_pm1 = to_pm1(k_bit)
        v_pm1 = to_pm1(v_bit)
        q_pm1 = to_pm1(q_bit)
        S_state_pm1 = torch.where(self._stream_A >= 0, 1.0, -1.0)
        r = to_bit1(q_pm1 * S_state_pm1)
        self._stream_A = self.gamma * self._stream_A + (k_pm1 * v_pm1)
        return r
