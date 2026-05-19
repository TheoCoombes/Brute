"""Memory accounting helpers for BGPT-1 training.

The training memory budget (BGPT-1 spec §5) targets ≤ 10 % overhead over
inference. This module exposes:

- :func:`inference_bytes` — sum of model weights + INT8 LM-head bias +
  small per-attention buffers (τ, ALiBi, causal mask) + KV cache
  (binary).
- :func:`training_extra_bytes` — sum of activation/gate tape + flip-rule
  per-layer counters + LM-head bias accumulator.
- :func:`memory_report` — pretty-print a per-component table + the
  ratio.

The accounting is structural (reads tensor metadata), not introspective —
no callbacks into the allocator. It counts every ``torch.Tensor`` /
``brute.Tensor`` reachable from the model + optimizer + tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional

import torch

from brute.tensor import Tensor


def _bytes(t) -> int:
    """Return ``t``'s storage byte count. Works for bit1 (uses packed buf),
    bool, int, float, etc. Returns 0 for None / non-tensor."""
    if t is None:
        return 0
    if isinstance(t, Tensor) and getattr(t, "_is_bit1", False):
        pb = t._packed_buf
        return pb.nbytes if pb is not None else 0
    if isinstance(t, torch.Tensor):
        return t.nbytes
    return 0


def inference_bytes(model, lm_head=None) -> Dict[str, int]:
    """Return a dict {component: bytes} for the inference-only footprint.

    Counts:
    - Every named buffer of every binary linear module (weight + optional
      bias).
    - Per-attention τ, τ accumulator, ALiBi bias table, causal mask.
    - LM-head INT8 bias (50 KB at V=50k) — but NOT the accumulator (that
      is training-only).
    - KV cache placeholder if the caller passes one; otherwise zero.

    The embedding's weight is tied to ``lm_head.embedding.weight`` so we
    only count it once via the model's named_buffers walk.
    """
    counts: Dict[str, int] = {}
    for name, buf in model.named_buffers():
        counts[name] = buf.nbytes if not getattr(buf, "_is_bit1", False) else (
            buf._packed_buf.nbytes if buf._packed_buf is not None else 0
        )
    return counts


def training_extra_bytes(
    tape: Optional[Dict] = None,
    trainer = None,
    err_signed = None,
    model = None,
) -> Dict[str, int]:
    """Return a dict {component: bytes} for the training overhead.

    Counts:
    - Activation tape per block (x_in + ffn h + 6 gate flags, all bit1).
    - The LM-head bias *accumulator* (V × int32).
    - Per-layer flip-rule state (confidence counters, scalars).
    - The signed-int error tensor at the LM-head boundary (sparse int8).
    """
    counts: Dict[str, int] = {}
    if tape is not None:
        block_total = 0
        intermediate_total = 0
        for blk in tape.get("blocks", []):
            if blk.get("skipped"):
                continue
            block_total += _bytes(blk.get("x_in"))
            # Per-block intermediates — only present when forward was NOT
            # checkpointed. With checkpoint=True (default) these are
            # regenerated on the fly during backward and don't contribute
            # to the snapshot taken between forward and backward.
            for v in blk.get("ffn_tape", {}).values():
                intermediate_total += _bytes(v)
            for v in blk.get("attn_tape", {}).values():
                intermediate_total += _bytes(v)
        counts["tape.block_boundaries"] = block_total
        if intermediate_total > 0:
            counts["tape.intermediates"] = intermediate_total
        counts["tape.x_after_embed"] = _bytes(tape.get("x_after_embed"))
        counts["tape.x_final"] = _bytes(tape.get("x_final"))
        counts["tape.logits"] = _bytes(tape.get("logits"))
    if trainer is not None:
        cstate = 0
        acc_state = 0
        for ls in trainer.layers.values():
            cstate += _bytes(ls.confidence)
            acc_state += _bytes(ls.accumulator)
        counts["flip_rule_confidence"] = cstate
        counts["flip_rule_accumulator"] = acc_state
    # LM-head bias accumulator removed in v2; the lm_head no longer holds
    # a non-binary buffer, so nothing to account for here.
    if err_signed is not None:
        counts["err_signed"] = _bytes(err_signed)
    return counts


def memory_report(
    model,
    trainer = None,
    tape: Optional[Dict] = None,
    err_signed = None,
) -> str:
    """Format a memory breakdown + inference/training ratio.

    Reports TWO ratios:
      *snapshot*  — total memory right after forward (incl. logits).
      *peak (est)* — projected peak during backward: logits are freed at
        the start of backward; the per-block intermediate tape (with
        checkpointing) is recomputed for one block at a time. We model
        peak = inference + saved tape (minus logits) + one block of
        regenerated intermediates.
    """
    inf = inference_bytes(model)
    train_extra = training_extra_bytes(
        tape=tape, trainer=trainer, err_signed=err_signed, model=model,
    )
    inf_total = sum(inf.values())
    train_extra_total = sum(train_extra.values())
    train_total = inf_total + train_extra_total
    ratio = train_total / max(inf_total, 1)

    # Peak-during-backward estimate.
    logits_bytes = train_extra.get("tape.logits", 0)
    # Approximate one-block intermediate (regenerated under checkpointing).
    # For our shapes: h (4d × B × C / 8) + 6 gate flags ≈ 7/8 * (5*B*C*d).
    one_block_intermediates = 0
    if tape is not None and tape.get("blocks"):
        first = next((b for b in tape["blocks"] if not b.get("skipped")), None)
        if first is not None:
            # x_in is in saved tape. Estimate intermediates ≈ 7× x_in.
            x_in_bytes = _bytes(first.get("x_in"))
            one_block_intermediates = 7 * x_in_bytes
    train_peak = (
        inf_total
        + (train_extra_total - logits_bytes)
        + one_block_intermediates
    )
    peak_ratio = train_peak / max(inf_total, 1)

    lines: List[str] = []
    lines.append("INFERENCE memory:")
    grouped: Dict[str, int] = {}
    for k, v in inf.items():
        parts = k.split(".")
        if parts[0] == "blocks" and len(parts) >= 2:
            key = f"blocks.{parts[1]} (total)"
            grouped[key] = grouped.get(key, 0) + v
        else:
            grouped[k] = v
    for k in sorted(grouped):
        lines.append(f"  {k:<40s} {grouped[k] / 1024:>10.1f} KB")
    lines.append(f"  {'INFERENCE TOTAL':<40s} {inf_total / 1024:>10.1f} KB")
    lines.append("")
    lines.append("TRAINING EXTRA memory (snapshot after forward):")
    for k in sorted(train_extra):
        lines.append(f"  {k:<40s} {train_extra[k] / 1024:>10.1f} KB")
    lines.append(f"  {'TRAINING EXTRA TOTAL':<40s} {train_extra_total / 1024:>10.1f} KB")
    lines.append("")
    lines.append(f"snapshot total = {train_total / 1024:.1f} KB"
                  f"   ratio 1 : {ratio:.3f}")
    lines.append(f"peak (est)    = {train_peak / 1024:.1f} KB"
                  f"   ratio 1 : {peak_ratio:.3f}")
    if peak_ratio < 1.1:
        lines.append("  ✓ peak within 1.1× target")
    else:
        lines.append(f"  ⚠  peak exceeds 1.1× target by "
                      f"{(peak_ratio - 1.1) * 100:.1f}%")
    return "\n".join(lines)


__all__ = ["inference_bytes", "training_extra_bytes", "memory_report"]
