"""Streaming flip-rule optimizer for bit1 weights (BGPT-1 §4.3).

Per layer we maintain:
- ``mu``: float temperature multiplier — feedback loop targets a per-layer
  flip rate. Bigger layers usually need lower mu; embeddings often need
  higher. Clipped to ``[mu_min, mu_max]``.
- ``f``: float flip-rate EMA over recent batches.
- ``c``: int8 ``(m,)`` confidence counter (per output row), clipped to
  ``[0, 3]`` (2-bit). Tracks whether the row's vote evidence is consistent.
- ``n_ref``: reference batch size for normalizing per-step flip probability
  (defaults to ``2^17 · sqrt(P_l / P_ref)`` where ``P_l`` is layer
  parameter count and ``P_ref`` is the geometric mean over registered
  layers).

Per-batch update (fully streamed per row-block — no full ``(m, n)`` tensor
ever materializes):
1. For each row block ``[row_start:row_end)`` of width ``B_v``:
   a. Compute the vote block ``v_block`` of shape ``(B_v, n)`` int32.
      Binary err: a single ``bit1 @ bit1`` matmul (XNOR-popcount). With a
      gate, two matmuls via the ``(Y1−Y2)/2`` identity.
   b. Per-row decisiveness ``mean(|v_block|) / B`` updates the rows'
      2-bit confidence counters.
   c. Build the descent flip mask (a bool ``(B_v, n)`` temporary) and
      pack it to bit1 in place. The packed flip mask is XOR'd into the
      row-block of ``W``'s packed buffer — a single packed XOR with no
      bool round-trip on the weight.
2. ``β = 1 - f`` damps the per-batch flip probability for thrashing layers.
3. After the layer's step, ``mu`` is updated multiplicatively based on the
   distance from the target flip rate.

Memory per layer-step is dominated by **one row-block's vote scratch**:
``B_v · n · 4`` bytes of int32. The bool flip mask for that row-block
adds ``B_v · n`` bytes. Everything is freed after the row-block iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import math

import torch

import brute
from brute import nn
from brute.tensor import Tensor
from brute.nn.binary_norm import bit_balance

# Raw kernel handles — bound once at import.
_XNOR_MATMUL = torch.ops.brute.xnor_popcount_matmul
_PACK_BOOL = torch.ops.brute.pack_bool
_PACKED_POPCOUNT = torch.ops.brute.packed_popcount


@dataclass
class LayerState:
    """Mutable per-layer state held by :class:`FlipRule`."""

    name: str
    weight: Tensor  # (m, n) bit1; modified in place
    n_ref: float
    mu: float = 1.0
    f: float = 0.0
    confidence: Optional[Tensor] = None  # int8 (m,) ∈ [0, 3]
    flip_total: int = 0

    def __post_init__(self):
        if self.confidence is None:
            m = self.weight.shape[0]
            self.confidence = brute.zeros(m, dtype=brute.int8, device=self.weight.device)


class FlipRule:
    """Stateful flip-rule optimizer for bit1 weights."""

    def __init__(
        self,
        t_init: float = 0.5,
        t_final: float = 0.01,
        total_steps: int = 10000,
        target_flip_rate: float = 0.01,
        mu_min: float = 0.1,
        mu_max: float = 10.0,
        mu_adjust: float = 0.05,
        decisive_threshold: float = 0.3,
        flip_rate_ema_decay: float = 0.9,
        row_block: int = 64,
        signed_chunk: int = 1024,
    ):
        self.t_init = float(t_init)
        self.t_final = float(t_final)
        self.total_steps = int(total_steps)
        self.target_flip_rate = float(target_flip_rate)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.mu_adjust = float(mu_adjust)
        self.decisive_threshold = float(decisive_threshold)
        self.flip_rate_ema_decay = float(flip_rate_ema_decay)
        self.row_block = int(row_block)
        # Batch-chunk size for the signed-int LM-head path. The signed-int
        # matmul needs to read ``x.unpack_pm1()`` — float32 (B, n). Chunking
        # over ``B`` caps the unpack temporary at ``signed_chunk · n · 4``
        # bytes.
        self.signed_chunk = int(signed_chunk)
        self.step_count = 0

        self.layers: Dict[str, LayerState] = {}
        self._n_ref_finalized = False

    # Registration
    def add_param(self, name: str, weight: Tensor, n_ref: Optional[float] = None) -> None:
        if name in self.layers:
            raise KeyError(f"layer {name!r} already registered")
        if not getattr(weight, "_is_bit1", False):
            raise TypeError(f"expected bit1 weight for {name!r}; got dtype {weight.dtype}")
        self.layers[name] = LayerState(name=name, weight=weight, n_ref=n_ref or 0.0)

    def _finalize_ref_sizes(self) -> None:
        if self._n_ref_finalized:
            return
        log_sum = 0.0
        n = 0
        for ls in self.layers.values():
            p = ls.weight.numel()
            log_sum += math.log(max(p, 1))
            n += 1
        if n == 0:
            self._n_ref_finalized = True
            return
        p_ref = math.exp(log_sum / n)
        n0 = float(1 << 17)
        for ls in self.layers.values():
            if ls.n_ref <= 0:
                p_l = ls.weight.numel()
                ls.n_ref = n0 * math.sqrt(max(p_l, 1) / p_ref)
        self._n_ref_finalized = True

    def base_temperature(self) -> float:
        """Cosine-annealed base temperature ``T_base``."""
        if self.total_steps <= 0:
            return self.t_init
        frac = min(self.step_count, self.total_steps) / self.total_steps
        return self.t_final + 0.5 * (self.t_init - self.t_final) * (1 + math.cos(math.pi * frac))

    # Internal: compute a single vote row block, given the pre-transposed
    # packed buffers err_t (m, B/64) and x_t (n, B/64).
    @staticmethod
    def _vote_block_bit1_packed(
        err_t_packed: torch.Tensor,           # (m, B/64) int64 — err.t() packed
        x_t_packed: torch.Tensor,             # (n, B/64) int64 — x.t()   packed
        row_start: int,
        row_end: int,
        B: int,
        gate_t_packed: Optional[torch.Tensor],  # (m, B/64) int64 — gate.t() packed
    ) -> torch.Tensor:
        """Vote row-block via two raw XNOR-popcount calls.

        The descent rule is symmetric in the m axis (output features), so
        a row slice of the *transposed* err is a contiguous view of the
        pre-transposed packed buffer (no per-block unpack).

        With a gate (BGPT-1 §4.2), applies BEP masking via ``(Y1 − Y2)/2``.
        Spec convention: ``gate[b, i] = 1`` means the position is near the
        decision boundary and SHOULD be included.
        """
        err_block = err_t_packed[row_start:row_end]            # (B_v, B/64) view
        y1 = _XNOR_MATMUL(err_block.contiguous(), x_t_packed, B)
        if gate_t_packed is None:
            return y1
        b_packed = gate_t_packed[row_start:row_end] ^ err_block  # (B_v, B/64)
        y2 = _XNOR_MATMUL(b_packed.contiguous(), x_t_packed, B)
        y1.sub_(y2)
        y1 //= 2
        return y1

    def _vote_block_signed(
        self,
        err_flat: Tensor,                     # (B, m) signed int (int8/int32)
        x_flat: Tensor,                       # (B, n) bit1
        row_slice: slice,
    ) -> Tensor:
        """Signed-int err path: chunk over the batch axis to bound the
        float32 ``x.unpack_pm1()`` temporary."""
        B, n = x_flat.shape
        chunk = max(1, min(self.signed_chunk, B))
        n_rows = row_slice.stop - row_slice.start
        vote_block = brute.zeros(n_rows, n, dtype=brute.float32, device=x_flat.device)
        # err_flat[:, row_slice] is small ((B, B_v) int8); slicing is cheap.
        err_sub_full = err_flat[:, row_slice].to(brute.float32)
        for s in range(0, B, chunk):
            e = s + chunk
            x_chunk_pm1 = x_flat[s:e].unpack_pm1()    # (chunk, n) float32
            vote_block.add_(err_sub_full[s:e].t() @ x_chunk_pm1)
        return vote_block.to(brute.int32)

    # Vote + flip — the inner loop, one layer per call.
    def flip_step(
        self,
        name: str,
        x: Tensor,                            # (..., n) bit1
        err: Tensor,                          # (..., m) bit1 OR signed int
        gate: Optional[Tensor] = None,        # (..., m) bit1 — True = saturated
        batch_size_override: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Stream the vote per row-block and apply stochastic flips.

        Args:
          name: Registered layer name.
          x: bit1 input activations ``(..., n)``.
          err: error at layer output ``(..., m)``. Bit1 inside the body;
            signed int (e.g. int8 4-bit-magnitude) at the LM-head boundary.
          gate: optional bit1 gate flag at the layer output. True means
            "saturated, mask this position from the vote and propagation".
            Only honored for the binary-err path (the signed-int path is
            used at the LM-head boundary, which has no gate).
          batch_size_override: use this ``B`` for the flip probability
            instead of inferring from ``x``.

        Returns:
          ``(n_rows_visited, flip_count)``. ``n_rows_visited`` is ``m``;
          ``flip_count`` is the number of bit positions flipped.
        """
        self._finalize_ref_sizes()
        ls = self.layers[name]
        W = ls.weight
        m, n = W.shape

        # Shape checks + flatten leading dims into B.
        if x.shape[-1] != n:
            raise ValueError(f"{name}: x.shape[-1]={x.shape[-1]} != W.shape[1]={n}")
        if err.shape[-1] != m:
            raise ValueError(f"{name}: err.shape[-1]={err.shape[-1]} != W.shape[0]={m}")
        x_flat = x.reshape(-1, n)
        err_flat = err.reshape(-1, m)
        gate_flat = gate.reshape(-1, m) if gate is not None else None
        B = int(batch_size_override) if batch_size_override is not None else int(x_flat.shape[0])

        # Schedule and β.
        T_base = self.base_temperature()
        beta = max(0.0, 1.0 - ls.f)
        p_flip = max(0.0, T_base * ls.mu * beta * (B / max(ls.n_ref, 1.0)))
        p_flip = min(p_flip, 1.0)

        # Stream per row-block.
        is_bit1_err = isinstance(err_flat, Tensor) and getattr(err_flat, "_is_bit1", False)
        flips_total = 0
        device = W.device
        decisive_threshold = self.decisive_threshold
        norm = float(n) * max(B, 1)

        # For the bit1 path we materialize one transposed packed buffer for
        # ``x`` only — it's reused unchanged across all row-blocks of this
        # layer. For ``err`` we slice in the m axis per row-block. The slice
        # is a clean packed view when ``row_block % PACK_WIDTH == 0`` AND
        # ``m % PACK_WIDTH == 0`` (both true at toy + Phase-4a scale).
        x_t_packed = None
        err_t_packed = None
        gate_t_packed = None
        if is_bit1_err:
            # x.t() — one transposed buffer for the whole layer's vote work.
            x_t_packed = x_flat.t()._packed_buf.contiguous()
            # err_t and gate_t — also materialized once. (The alternative
            # of slicing err in the m axis per-block and transposing each
            # tiny slice has the same total cost via brute.t()'s
            # unpack→transpose→repack, paid in many small calls instead of
            # one large one.)
            err_t_packed = err_flat.t()._packed_buf.contiguous()
            if gate_flat is not None:
                gate_t_packed = gate_flat.t()._packed_buf.contiguous()

        for row_start in range(0, m, self.row_block):
            row_end = min(row_start + self.row_block, m)
            row_slice = slice(row_start, row_end)

            # 1. Vote block.
            if is_bit1_err:
                vote_block = self._vote_block_bit1_packed(
                    err_t_packed, x_t_packed, row_start, row_end, B, gate_t_packed,
                )
            else:
                vote_block = self._vote_block_signed(err_flat, x_flat, row_slice)

            # 2. Per-row confidence update.
            row_decisive = (
                vote_block.abs().sum(dim=1).to(brute.float32) / norm
            ) > decisive_threshold
            c_block = ls.confidence[row_slice].to(brute.int16)
            c_up = (c_block + 1).clamp(max=3)
            c_dn = (c_block - 1).clamp(min=0)
            c_new = brute.where(row_decisive, c_up, c_dn).to(brute.int8)
            ls.confidence[row_slice] = c_new
            allow_block = c_new >= 1

            # 3. Stochastic flip — descent rule, raw packed buffers only.
            #    flip iff sign(vote) ≠ sign(W) ⇔ (vote > 0) XOR W_bool.
            # Each int-derived bool is packed to bit1 via the raw kernel;
            # peak bool overlap is one (B_v, n) tensor, freed before the
            # next comparison. W's row-slice packed buffer is XOR'd in
            # place once at the end.
            if p_flip <= 0.0:
                continue
            v_pos_packed = _PACK_BOOL((vote_block > 0).contiguous())
            # W's row-slice packed buffer — pure view, no unpack.
            w_block_packed = W._packed_buf[row_start:row_end]
            descent_packed = v_pos_packed ^ w_block_packed
            nonzero_packed = _PACK_BOOL((vote_block != 0).contiguous())
            descent_packed &= nonzero_packed
            coin_packed = _PACK_BOOL(
                (brute.rand(vote_block.shape, device=device) < p_flip).contiguous()
            )
            fm_packed = descent_packed & coin_packed
            # Per-row "allow" mask: multiply each packed row by 0/1.
            allow_i64 = allow_block.to(brute.int64).unsqueeze(1)
            fm_packed.mul_(allow_i64)
            n_flips = int(_PACKED_POPCOUNT(fm_packed).item())
            if n_flips == 0:
                continue
            w_block_packed.bitwise_xor_(fm_packed)
            flips_total += n_flips

        # Bool view of W is now stale relative to the packed buffer.
        if flips_total > 0:
            ls.weight.__dict__["_bool_dirty"] = True
            try:
                ls.weight.__dict__["_packed_ver"] = ls.weight._version
            except Exception:
                pass

        # 4. Flip-rate EMA + μ feedback.
        flip_rate_this = flips_total / max(m * n, 1)
        ls.f = self.flip_rate_ema_decay * ls.f + (1.0 - self.flip_rate_ema_decay) * flip_rate_this
        if ls.f > 2 * self.target_flip_rate:
            ls.mu = max(ls.mu * (1.0 - self.mu_adjust), self.mu_min)
        elif ls.f < 0.5 * self.target_flip_rate:
            ls.mu = min(ls.mu * (1.0 + self.mu_adjust), self.mu_max)
        ls.flip_total += flips_total
        return m, flips_total

    def step_global(self) -> None:
        self.step_count += 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        snap = {}
        for name, ls in self.layers.items():
            snap[name] = {
                "mu": ls.mu,
                "f": ls.f,
                "flip_total": ls.flip_total,
                "n_ref": ls.n_ref,
                "c_mean": float(ls.confidence.float().mean().item()),
            }
        snap["__global__"] = {"step": self.step_count, "T_base": self.base_temperature()}
        return snap


# ── Error propagation ──────────────────────────────────────────────────────

def propagate_error(
    err: Tensor,                              # (..., m) bit1 OR signed int
    weight: Tensor,                           # (m, n) bit1
    gate: Optional[Tensor] = None,            # (..., m) bit1 — True = saturated
    use_bit_balance: bool = True,
    signed_chunk: int = 1024,
) -> Tensor:
    """Backward operator: ``err_in = sign(W^T @ err)`` with optional gating.

    Binary err: a single bit1×bit1 XNOR-popcount along ``m``. With a gate,
    two matmuls via ``(Y1 − Y2)/2`` (BEP masking — see :class:`FlipRule`).
    Signed-int err (LM-head boundary): chunked over weight rows to bound
    the float32 ``weight.unpack_pm1()`` temporary at ``signed_chunk · n · 4``
    bytes.
    """
    m, n = weight.shape
    err_flat = err.reshape(-1, m)
    if isinstance(err_flat, Tensor) and getattr(err_flat, "_is_bit1", False):
        # bit1 path. brute matmul convention: A(M,K) @ B(N,K) -> (M, N).
        # err is (B, m) and weight is (m, n) — last dims (m, n) don't match
        # so we'd need weight.t(). Pre-pack it once (one unpack-transpose-
        # pack) so both matmul calls reuse the same packed buffer.
        weight_t_packed = weight.t()._packed_buf.contiguous()
        err_packed = err_flat._packed_buf.contiguous()
        y1 = _XNOR_MATMUL(err_packed, weight_t_packed, m)   # (B, n) int32
        if gate is None:
            score = y1
        else:
            # Spec §4.2: gate=1 means "include"; gate=0 means "mask out".
            gate_flat = gate.reshape(-1, m)
            b_xor_packed = err_packed ^ gate_flat._packed_buf.contiguous()
            y2 = _XNOR_MATMUL(b_xor_packed, weight_t_packed, m)
            y1.sub_(y2)
            y1 //= 2
            score = y1
    else:
        # Signed int path — chunk over m to cap the unpack temporary. The
        # accumulated value is integer-valued (err ∈ [-7, +7], weight ∈
        # ±1), so we do the matmul in float for hardware throughput but
        # cast back to int32 immediately after to avoid drift in
        # downstream sign comparisons.
        chunk = max(1, min(signed_chunk, m))
        B = err_flat.shape[0]
        score_f = brute.zeros(B, n, dtype=brute.float32, device=err_flat.device)
        err_f32 = err_flat.to(brute.float32)
        for s in range(0, m, chunk):
            e = s + chunk
            w_pm1_chunk = weight[s:e].unpack_pm1()  # (chunk, n) float32
            score_f.add_(err_f32[:, s:e] @ w_pm1_chunk)
        score = score_f.to(brute.int32)

    if not use_bit_balance:
        return score.reshape(*err.shape[:-1], n)
    err_in, _ = bit_balance(score, nu=0.0)
    return err_in.reshape(*err.shape[:-1], n)


__all__ = ["FlipRule", "LayerState", "propagate_error"]
