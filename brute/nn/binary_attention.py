"""Two-stage native 1-bit attention — fully packed bit1 throughout.

The hot loop is hand-tuned: each per-``(batch, head)`` iteration issues
~5 raw ``torch.ops.brute.*`` calls on packed int64 buffers, skipping
``__torch_function__`` dispatch entirely. Empirically this drops the
per-iteration cost from ~50 µs to ~5 µs.

Stage 1 — XNOR-popcount similarity with ALiBi bias:
    S[i, j] = popcount(XNOR(Q[i], K[j])) − d_h/2  +  α[h] · (i − j)

Stage 2 — exact masked sign-aggregation (see :func:`_stage2_packed`):
    Y = (matmul(A_sign, V.t()) − matmul(A_active XOR A_sign, V.t())) / 2

Per-``(batch, head)`` peak working set: ``(C, C) int32`` (the score
matrix). The transposed ``V.t()`` packed buffer is computed once per
iteration (cost ``O(C · d_h)`` bool intermediate) and used for the two
stage-2 matmuls.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch

import brute
from brute import nn
from brute.tensor import Tensor
from brute.dtype import _PACK_WIDTH
from brute.nn.linear import BruteLinear
from brute.nn.binary_norm import bit_balance


# Raw kernel handles — bound once at import to avoid repeated attribute
# lookups in the hot loop.
_XNOR_MATMUL = torch.ops.brute.xnor_popcount_matmul
_PACK_BOOL = torch.ops.brute.pack_bool
_UNPACK_BOOL = torch.ops.brute.unpack_bool


def alibi_slopes(n_heads: int, device=None) -> Tensor:
    """Standard ALiBi slopes (Press et al. 2021)."""
    h_idx = brute.arange(n_heads, dtype=brute.float32, device=device)
    return -(2.0 ** (-8.0 * (h_idx + 1) / n_heads))


def _alibi_bias_table(n_heads: int, c_max: int, device=None) -> Tensor:
    pos = brute.arange(c_max, dtype=brute.float32, device=device)
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    slopes = alibi_slopes(n_heads, device=device)
    bias_fp = slopes.view(-1, 1, 1) * delta.unsqueeze(0)
    return bias_fp.round().to(brute.int32)


def _stage2_packed(
    a_sign_packed: torch.Tensor,    # (C, C/64) int64
    a_active_packed: torch.Tensor,  # (C, C/64) int64
    vt_packed: torch.Tensor,        # (d_h, C/64) int64 — V.t() packed form
    c: int,                          # seq length (= K dim of stage-2 matmul)
) -> torch.Tensor:
    """Exact ``Y = (matmul(A_sign, V.t()) − matmul(B, V.t())) / 2``.

    All ops are raw kernel calls on packed int64 buffers — no
    ``__torch_function__`` involvement.
    """
    # B = A_active XOR A_sign  (bit1 XOR = packed int64 XOR).
    b_packed = a_active_packed ^ a_sign_packed
    y1 = _XNOR_MATMUL(a_sign_packed, vt_packed, c)
    y2 = _XNOR_MATMUL(b_packed,      vt_packed, c)
    # In-place (y1 -= y2; y1 >>= 1) is cheap.
    y1.sub_(y2)
    y1 //= 2
    return y1  # (C, d_h) int32


def stage2_aggregate(
    a_active: Tensor,                # (C, C) bit1
    a_sign: Tensor,                  # (C, C) bit1
    v: Tensor,                       # (C, d_h) bit1
) -> torch.Tensor:
    """Reference / verification wrapper around :func:`_stage2_packed`.

    Accepts the three bit1 brute.Tensors and returns the ``(C, d_h)``
    int32 aggregate, matching the algebraic identity
    ``Y = Σ_{j: A[i,j]≠0} A[i,j] · V[j, c]``.
    """
    c = a_active.shape[-1]
    d_h = v.shape[-1]
    # V.t() packed form — unpack V to bool, transpose, repack. Cost is one
    # (C, d_h) bool intermediate per call.
    v_bool = _UNPACK_BOOL(v._packed_buf.contiguous(), [c, d_h])
    vt_packed = _PACK_BOOL(v_bool.t().contiguous())
    return _stage2_packed(
        a_sign._packed_buf.contiguous(),
        a_active._packed_buf.contiguous(),
        vt_packed,
        c,
    )


class BinaryAttention(nn.Module):
    """Multi-head bit1 attention with ALiBi bias and 2-bit zero-band map.

    ``d_h = dim // n_heads`` must be a multiple of ``_PACK_WIDTH`` (64) so
    the per-head packed view is a zero-copy slice of the parent packed
    buffer.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_context: int,
        tau_init: int = 0,
        nu: float = 0.05,
        att_dec_target: float = 0.5,
        tau_accum_threshold: int = 64,
        device=None,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.d_h = self.dim // self.n_heads
        if self.d_h % _PACK_WIDTH != 0:
            raise ValueError(
                f"d_h = dim / n_heads = {self.d_h} must be a multiple of "
                f"{_PACK_WIDTH} (pack width) for zero-copy per-head views"
            )
        self.d_h_words = self.d_h // _PACK_WIDTH
        self.max_context = int(max_context)
        self.nu = float(nu)
        self.att_dec_target = float(att_dec_target)
        self.tau_accum_threshold = int(tau_accum_threshold)

        self.q_proj = BruteLinear(dim, dim, bias=False, device=device)
        self.k_proj = BruteLinear(dim, dim, bias=False, device=device)
        self.v_proj = BruteLinear(dim, dim, bias=False, device=device)
        self.out_proj = BruteLinear(dim, dim, bias=False, device=device)

        self.register_buffer(
            "tau",
            brute.full((self.n_heads,), int(tau_init), dtype=brute.int32, device=device),
        )
        self.register_buffer(
            "tau_accumulator",
            brute.zeros(self.n_heads, dtype=brute.int32, device=device),
        )
        self.register_buffer(
            "alibi_bias",
            _alibi_bias_table(self.n_heads, self.max_context, device=device),
        )
        # Causal mask as bool (one-time cost: max_context²/8 ratio matters
        # only at very large C; the bool form keeps the hot path lean since
        # we use it for two integer masked_fills per iteration).
        idx = brute.arange(self.max_context, device=device)
        causal_bool = idx.unsqueeze(0) > idx.unsqueeze(1)
        self.register_buffer("causal_mask", causal_bool)

    # --- Forward ---
    def forward(
        self,
        x: Tensor,                  # (B, C, dim) bit1
        return_diag: bool = False,
        return_tape: bool = False,
    ) -> Tuple[Tensor, Optional[dict], Optional[dict]]:
        b, c, d = x.shape
        if c > self.max_context:
            raise ValueError(f"sequence length {c} exceeds max_context {self.max_context}")
        d_h = self.d_h
        nwords = self.d_h_words
        n_heads = self.n_heads
        device = x.device
        tau_local = self.tau           # (n_heads,) int32 — small, cached
        alibi_local = self.alibi_bias  # (n_heads, max_c, max_c) int32

        # QKV projections + bit-balance. Gates are captured when training.
        nu = self.nu if return_tape else 0.0
        q_pre = self.q_proj(x)
        k_pre = self.k_proj(x)
        v_pre = self.v_proj(x)
        q_bit1, gate_q = bit_balance(q_pre, nu=nu)
        k_bit1, gate_k = bit_balance(k_pre, nu=nu)
        v_bit1, gate_v = bit_balance(v_pre, nu=nu)

        # Raw packed views, reshaped to (B, C, n_heads, nwords). Pure
        # view-op on the int64 storage when d_h % PACK_WIDTH == 0.
        q_packed = q_bit1._packed_buf.reshape(b, c, n_heads, nwords)
        k_packed = k_bit1._packed_buf.reshape(b, c, n_heads, nwords)
        v_packed = v_bit1._packed_buf.reshape(b, c, n_heads, nwords)

        # Causal-region bool slice — used once per iteration. (C, C) bool;
        # at typical C this is small (e.g. C=256 → 64 KB).
        causal_bool = self.causal_mask[:c, :c]
        causal_neg_int = -(1 << 30)

        # Aggregate output buffer — preallocated (B, C, n_heads, d_h) int32.
        agg = brute.empty((b, c, n_heads, d_h), dtype=brute.int32, device=device)

        att_dec_per_head = [0.0] * n_heads if return_diag else None
        if return_diag:
            causal_valid_count = int((~causal_bool).sum().item())
            causal_inv_bool = ~causal_bool  # bool, used inside diag block

        for bi in range(b):
            for hi in range(n_heads):
                # Slice per-head packed views (zero-copy on contiguous tail).
                q_p = q_packed[bi, :, hi, :].contiguous()
                k_p = k_packed[bi, :, hi, :].contiguous()

                # Stage 1: raw matmul → (C, C) int32. Add ALiBi, apply causal.
                s = _XNOR_MATMUL(q_p, k_p, d_h)
                s.add_(alibi_local[hi, :c, :c])
                s.masked_fill_(causal_bool, causal_neg_int)

                # Sign threshold. tau is int32 (n_heads,); .item() to scalar.
                tau = int(tau_local[hi].item())
                a_pos_bool = s > tau
                a_neg_bool = (s < -tau) & ~causal_bool
                # Pack each via the raw kernel — no Tensor wrapping.
                a_pos_packed = _PACK_BOOL(a_pos_bool.contiguous())
                a_neg_packed = _PACK_BOOL(a_neg_bool.contiguous())
                a_active_packed = a_pos_packed | a_neg_packed

                # Compute V_h.t() packed buffer (d_h, C/64) — unpack→transpose
                # →re-pack, all on the small per-head bool view.
                v_p = v_packed[bi, :, hi, :].contiguous()       # (C, nwords) int64
                v_h_bool = _UNPACK_BOOL(v_p, [c, d_h])           # (C, d_h) bool
                vt_packed = _PACK_BOOL(v_h_bool.t().contiguous())  # (d_h, C/64) int64

                # Stage 2: two raw matmuls, fused in _stage2_packed.
                y = _stage2_packed(a_pos_packed, a_active_packed, vt_packed, c)
                agg[bi, :, hi, :] = y

                if return_diag:
                    # Decisive cells (active AND causal-valid) — packed AND
                    # + raw popcount.
                    causal_inv_packed = _PACK_BOOL(causal_inv_bool.contiguous())
                    n_dec = int(torch.ops.brute.packed_popcount(
                        a_active_packed & causal_inv_packed
                    ).item())
                    att_dec_per_head[hi] += n_dec / max(causal_valid_count, 1) / b

        agg_flat = agg.reshape(b, c, self.dim)
        # Sign step on aggregate → bit1.
        head_out_bit1, _ = bit_balance(agg_flat, nu=0.0)
        # Output projection (+ gate).
        proj_pre = self.out_proj(head_out_bit1)
        out_bit1, gate_out = bit_balance(proj_pre, nu=nu)

        diag = None
        if return_diag:
            diag = {
                "att_dec": brute.tensor(att_dec_per_head, dtype=brute.float32),
                "att_sparsity": brute.tensor(
                    [1.0 - v for v in att_dec_per_head], dtype=brute.float32,
                ),
            }
        tape = None
        if return_tape:
            tape = {
                "head_out_bit1": head_out_bit1,
                "gate_q": gate_q,
                "gate_k": gate_k,
                "gate_v": gate_v,
                "gate_out": gate_out,
            }
        return out_bit1, diag, tape

    # --- τ feedback ---
    def tau_feedback_step(self, att_dec_per_head) -> int:
        if att_dec_per_head is None:
            return 0
        if isinstance(att_dec_per_head, (list, tuple)):
            att_dec_per_head = brute.tensor(att_dec_per_head, dtype=brute.float32)
        diff = att_dec_per_head.to(brute.float32) - self.att_dec_target
        delta_int = (diff * 1000.0).round().to(brute.int32)
        self.tau_accumulator.add_(delta_int)
        thr = self.tau_accum_threshold
        pos = self.tau_accumulator >=  thr
        neg = self.tau_accumulator <= -thr
        n_updates = int(pos.sum().item()) + int(neg.sum().item())
        if n_updates:
            self.tau[pos] = (self.tau[pos] + 1).clamp(0, self.d_h)
            self.tau[neg] = (self.tau[neg] - 1).clamp(0, self.d_h)
            self.tau_accumulator[pos] -= thr
            self.tau_accumulator[neg] += thr
        return n_updates


__all__ = ["BinaryAttention", "alibi_slopes"]
