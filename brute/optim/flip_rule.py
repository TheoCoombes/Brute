"""BOLD-style flip-rule optimizer for bit1 weights.

References:
  * BOLD (Boolean Logic Deep Learning, NeurIPS 2024) — Algorithm 1, §3.3.
  * BGPT-1 informal spec §4.3, modified to use a BOLD per-weight accumulator
    instead of the original stochastic Bernoulli rule.

Per layer we maintain:
- ``accumulator``: int8 ``(m, n)`` per-weight signal integrator. Each step
  it gets ``sign(vote) · step_size`` added (saturating to ±127). When
  ``|accumulator| ≥ threshold`` AND ``sign(accumulator) ≠ sign(W)``, the
  weight is flipped and its accumulator entry is reset to 0. This is
  BOLD's ``m_{i,j}^{l,t}`` from Algorithm 1.
- ``mu``: float temperature multiplier per layer. Feedback loop targets a
  per-layer flip rate. Clipped to ``[mu_min, mu_max]``.
- ``f``: float flip-rate EMA over recent batches (diagnostic + mu feedback).
- ``beta_layer``: float, BOLD's β = N_unchanged / N_tot from previous step.
  Damps accumulator growth when a layer thrashes.
- ``n_ref``: reference batch size for normalizing per-step step size.

Per-batch update (fully streamed per row-block — accumulator row-block is
the only ``(B_v, n)`` tensor that materializes):
1. Compute vote_block (XNOR-popcount with optional gate masking via the
   ``(Y1 − Y2)/2`` identity).
2. acc[row_block] := clamp(beta·acc + step_size·sign(vote), -127, 127).
3. flip_mask = (|acc| ≥ threshold) AND (sign(acc) ≠ sign(W)).
4. W ^= flip_mask (packed XOR); acc[flip_mask] := 0.
5. Update layer-level β = N_unchanged / N_tot.
6. Update mu via flip-rate EMA feedback.

Memory: persistent accumulator costs ``Σ_l m_l · n_l`` bytes (~equal to
weight count). For a 10M-param model: 10 MB. Trade vs. BGPT-1 v1's
stateless Bernoulli rule: more memory, vastly better convergence.
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
_UNPACK_BOOL = torch.ops.brute.unpack_bool
_PACKED_POPCOUNT = torch.ops.brute.packed_popcount


@dataclass
class LayerState:
    """Mutable per-layer state held by :class:`FlipRule`."""

    name: str
    weight: Tensor  # (m, n) bit1; modified in place
    n_ref: float
    mu: float = 1.0
    f: float = 0.0
    # BOLD per-weight integer accumulator (m_{i,j}^{l,t}). int8, saturating.
    accumulator: Optional[Tensor] = None
    # BOLD's β = N_unchanged / N_tot from the previous step. 1.0 = no flips.
    beta_layer: float = 1.0
    # Optional per-layer accumulator threshold override (else uses the global
    # ``FlipRule.acc_threshold``). Higher = more patience before flipping.
    acc_threshold_override: Optional[int] = None
    # If True, this layer's weights are not updated in :meth:`FlipRule.flip_step`.
    # The accumulator is still updated (so unfreezing later doesn't restart
    # from cold).
    frozen: bool = False
    # Retained for backward-compat / diagnostics. No longer used as a gate.
    confidence: Optional[Tensor] = None
    flip_total: int = 0

    def __post_init__(self):
        m, n = self.weight.shape
        if self.accumulator is None:
            self.accumulator = brute.zeros(
                (m, n), dtype=brute.int8, device=self.weight.device,
            )
        if self.confidence is None:
            self.confidence = brute.zeros(m, dtype=brute.int8, device=self.weight.device)


class FlipRule:
    """Stateful flip-rule optimizer for bit1 weights."""

    def __init__(
        self,
        t_init: float = 0.5,
        t_final: float = 0.05,
        total_steps: int = 10000,
        target_flip_rate: float = 0.005,
        mu_min: float = 0.1,
        mu_max: float = 10.0,
        mu_adjust: float = 0.02,
        decisive_threshold: float = 0.02,  # legacy / diagnostic only
        flip_rate_ema_decay: float = 0.9,
        row_block: int = 64,
        signed_chunk: int = 1024,
        acc_threshold: int = 8,
        step_scale: float = 8.0,
        step_min: int = 1,
        target_flip_rate_final: Optional[float] = None,
    ):
        """BOLD-style flip rule optimizer.

        New parameters vs. v1:
          acc_threshold: ``|m| ≥ acc_threshold`` is the flip trigger. Higher =
            slower, cleaner flipping. Pure noise random walk hits threshold τ
            in roughly τ² steps, so τ=8 needs ~64 random-walk steps but only
            ~8 consistent-signal steps. Default 8.
          step_scale: step size per call = max(step_min, round(T·μ·step_scale)).
            At default T=0.5, μ=1, scale=8 → step=4 per call.
          step_min: minimum step size, ensuring late-schedule (small T) still
            advances the accumulator.
        """
        self.t_init = float(t_init)
        self.t_final = float(t_final)
        self.total_steps = int(total_steps)
        self.target_flip_rate_init = float(target_flip_rate)
        # If ``target_flip_rate_final`` is None, no annealing (constant rate).
        self.target_flip_rate_final = (
            float(target_flip_rate_final) if target_flip_rate_final is not None
            else float(target_flip_rate)
        )
        # Current target — recomputed each call via :attr:`target_flip_rate`.
        self.target_flip_rate = float(target_flip_rate)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.mu_adjust = float(mu_adjust)
        self.decisive_threshold = float(decisive_threshold)
        self.flip_rate_ema_decay = float(flip_rate_ema_decay)
        self.row_block = int(row_block)
        self.acc_threshold = int(acc_threshold)
        self.step_scale = float(step_scale)
        self.step_min = int(step_min)
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
        if not self.layers:
            self._n_ref_finalized = True
            return
        # n_ref = weight.numel() so that p_flip = T * mu * beta * (B / n_ref)
        # gives ~T*mu*beta flips-per-parameter per batch element. For B=16 and
        # T=0.3 this starts at ~0.3 flips/param/batch which is reasonable.
        # Layers with n_ref <= 0 (unset) receive this default.
        for ls in self.layers.values():
            if ls.n_ref <= 0:
                ls.n_ref = float(ls.weight.numel())
        self._n_ref_finalized = True

    def base_temperature(self) -> float:
        """Cosine-annealed base temperature ``T_base``."""
        if self.total_steps <= 0:
            return self.t_init
        frac = min(self.step_count, self.total_steps) / self.total_steps
        return self.t_final + 0.5 * (self.t_init - self.t_final) * (1 + math.cos(math.pi * frac))

    def current_target_flip_rate(self) -> float:
        """Cosine-annealed target flip rate. Falls back to constant rate
        if ``target_flip_rate_final`` was not provided."""
        if self.total_steps <= 0 or self.target_flip_rate_init == self.target_flip_rate_final:
            return self.target_flip_rate_init
        frac = min(self.step_count, self.total_steps) / self.total_steps
        return self.target_flip_rate_final + 0.5 * (
            self.target_flip_rate_init - self.target_flip_rate_final
        ) * (1 + math.cos(math.pi * frac))

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
        # Frozen layer: skip the whole step. (We deliberately do NOT update
        # the accumulator either — the upstream activations / err during
        # freeze are not the ones we'd be flipping against once unfrozen,
        # so accumulating now would create stale evidence.)
        if ls.frozen:
            return ls.weight.shape[0], 0
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

        # Schedule: temperature T_base · μ controls step size (= how fast the
        # accumulator advances per call). BOLD's η_t.
        T_base = self.base_temperature()
        eta = T_base * ls.mu
        step_size = max(self.step_min, int(round(eta * self.step_scale)))
        step_size = min(step_size, 64)  # cap to keep accumulator interpretable
        # BOLD's β_t multiplicative decay on the accumulator. β=1 means no
        # decay; β<1 dampens layers that have been thrashing.
        beta = ls.beta_layer
        # Per-layer threshold override (e.g. for depth-scaled patience).
        acc_threshold = ls.acc_threshold_override if ls.acc_threshold_override is not None else self.acc_threshold

        is_bit1_err = isinstance(err_flat, Tensor) and getattr(err_flat, "_is_bit1", False)
        flips_total = 0

        # Pre-transpose packed buffers once per call (binary err path only).
        x_t_packed = None
        err_t_packed = None
        gate_t_packed = None
        if is_bit1_err:
            x_t_packed = x_flat.t()._packed_buf.contiguous()
            err_t_packed = err_flat.t()._packed_buf.contiguous()
            if gate_flat is not None:
                gate_t_packed = gate_flat.t()._packed_buf.contiguous()

        # Get the underlying packed accumulator and weight buffers once.
        # The accumulator is int8 (m, n); the weight's _packed_buf is int64
        # (m, n/64).
        acc_full = ls.accumulator  # int8 view, shape (m, n)

        for row_start in range(0, m, self.row_block):
            row_end = min(row_start + self.row_block, m)
            row_slice = slice(row_start, row_end)

            # 1. Vote block — int32 (rows, n).
            if is_bit1_err:
                vote_block = self._vote_block_bit1_packed(
                    err_t_packed, x_t_packed, row_start, row_end, B, gate_t_packed,
                )
            else:
                vote_block = self._vote_block_signed(err_flat, x_flat, row_slice)

            # 2. Compute the accumulator step: sign(vote) · step_size.
            #    Using sign(vote) (rather than vote itself) makes step size
            #    independent of batch — strong signal is captured by
            #    *consistency over time*, not by magnitude per step.
            vote_sign = vote_block.sign().to(brute.int16)   # {-1, 0, +1}
            step_block = vote_sign * step_size               # int16

            # 3. Update accumulator: acc := clamp(β·acc + step, ±127).
            acc_block_i16 = acc_full[row_slice].to(brute.int16)
            if beta < 0.999:
                # Multiplicative decay; integer math (round toward zero).
                acc_block_i16 = (acc_block_i16.to(brute.float32) * beta).to(brute.int16)
            acc_block_i16 += step_block
            acc_block_i16.clamp_(-127, 127)

            # 4. Flip decision (deterministic, BOLD Eq. 9 in our sign convention):
            #    flip if |acc| ≥ threshold AND sign(acc) ≠ sign(W).
            #    Our W is stored as bit1 with bit=1 ↔ +1, bit=0 ↔ -1.
            big_pos = acc_block_i16 >= acc_threshold     # acc says "W should be +1"
            big_neg = acc_block_i16 <= -acc_threshold    # acc says "W should be -1"

            # Need W's row-block as bool. Unpack only the row slice. The
            # unpack_bool kernel takes the FULL logical shape (n_rows, n),
            # not just the last-dim size — passing [n] caused an OOB write.
            n_rows = row_end - row_start
            w_block_bool = _UNPACK_BOOL(
                W._packed_buf[row_start:row_end].contiguous(), [n_rows, n],
            )  # (n_rows, n) bool — True ↔ +1.
            # flip if (big_pos AND W=-1) OR (big_neg AND W=+1)
            flip_mask = (big_pos & ~w_block_bool) | (big_neg & w_block_bool)

            # 5. Apply flips: W ^= flip_mask (packed XOR); acc[flipped] := 0.
            # If frozen, skip the weight write but keep the accumulator
            # integration. (When the layer is unfrozen later, evidence is
            # warm rather than starting from zero.)
            n_flips_block = int(flip_mask.sum().item())
            if n_flips_block > 0 and not ls.frozen:
                fm_packed = _PACK_BOOL(flip_mask.contiguous())
                w_block_packed = W._packed_buf[row_start:row_end]
                w_block_packed.bitwise_xor_(fm_packed)
                # Zero the accumulator at flipped positions (BOLD's reset).
                acc_block_i16 = brute.where(
                    flip_mask, brute.zeros_like(acc_block_i16), acc_block_i16,
                )
                flips_total += n_flips_block

            # 6. Write back the int8-clamped accumulator block.
            acc_full[row_slice] = acc_block_i16.to(brute.int8)

        # Bool view of W is stale after packed XOR writes.
        if flips_total > 0:
            ls.weight.__dict__["_bool_dirty"] = True
            try:
                ls.weight.__dict__["_packed_ver"] = ls.weight._version
            except Exception:
                pass

        # 7. BOLD's β feedback: β = N_unchanged / N_tot (per-layer).
        n_tot = m * n
        ls.beta_layer = 1.0 - flips_total / max(n_tot, 1)

        # 8. EMA flip-rate + μ feedback (kept from v1 for layer self-tuning).
        # ``current_target_flip_rate`` is annealed alongside T_base so flip
        # pressure relaxes as training progresses.
        flip_rate_this = flips_total / max(n_tot, 1)
        ls.f = self.flip_rate_ema_decay * ls.f + (1.0 - self.flip_rate_ema_decay) * flip_rate_this
        target = self.current_target_flip_rate()
        self.target_flip_rate = target  # publish for snapshot()
        if ls.f > 2 * target:
            ls.mu = max(ls.mu * (1.0 - self.mu_adjust), self.mu_min)
        elif ls.f < 0.5 * target:
            ls.mu = min(ls.mu * (1.0 + self.mu_adjust), self.mu_max)
        ls.flip_total += flips_total
        return m, flips_total

    def step_global(self) -> None:
        self.step_count += 1

    def set_layer_threshold(self, name: str, threshold: int) -> None:
        """Override the per-layer accumulator threshold for ``name``."""
        self.layers[name].acc_threshold_override = int(threshold)

    def set_layer_frozen(self, name: str, frozen: bool) -> None:
        """Freeze / unfreeze a single layer's weight updates."""
        ls = self.layers[name]
        if ls.frozen and not frozen:
            # Cold start on unfreeze — clear any accumulated signal so we
            # don't trigger a flip-storm from pent-up votes.
            ls.accumulator.zero_()
        ls.frozen = bool(frozen)

    def freeze_all_except(self, allowed_prefixes: List[str]) -> int:
        """Freeze every layer whose name does NOT start with one of the
        given prefixes. Returns the number of layers left unfrozen.
        Layers that transition frozen → unfrozen get their accumulators
        zeroed."""
        n_active = 0
        for name, ls in self.layers.items():
            want_active = any(name.startswith(p) for p in allowed_prefixes)
            if want_active and ls.frozen:
                ls.accumulator.zero_()
            ls.frozen = not want_active
            if want_active:
                n_active += 1
        return n_active

    def unfreeze_all(self) -> None:
        for ls in self.layers.values():
            if ls.frozen:
                ls.accumulator.zero_()
            ls.frozen = False

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        snap = {}
        for name, ls in self.layers.items():
            acc_abs_mean = float(ls.accumulator.abs().float().mean().item())
            snap[name] = {
                "mu": ls.mu,
                "f": ls.f,
                "beta": ls.beta_layer,
                "flip_total": ls.flip_total,
                "n_ref": ls.n_ref,
                "acc_abs_mean": acc_abs_mean,
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
