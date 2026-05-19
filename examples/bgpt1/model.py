"""BGPT-1 model: forward + custom backward (no autograd).

Composed entirely from brute.nn modules. The forward pass returns logits
plus a tape of intermediate activations and gate flags that the custom
backward walks in reverse to compute votes and call the flip-rule
optimizer.

Why a hand-written backward? PyTorch autograd doesn't run on bit1 ops,
and the spec's flip-rule update isn't gradient descent on FP weights.
The backward here is the BGPT-1 spec §4.2 procedure: XNOR-popcount with
the transposed weight, bit-balanced sign, gate masking, per-layer vote
and stochastic flip.

Everything stays on packed bit1 except where integer arithmetic is
strictly necessary (the per-batch vote, the signed error signal at the
LM head boundary, and the integer pre-activation that feeds bit_balance).
The parallel-residual sign is computed via packed majority-of-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import brute
from brute import nn as bn          # alias to avoid shadowing torch.nn
from brute.tensor import Tensor
from brute.nn.binary_norm import bit_balance
from brute.optim import FlipRule, propagate_error
from brute.nn.binary_lm_head import (
    signed_int_error,
    lm_head_bias_step,
)


def _majority3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Majority-of-3 on packed bit1 tensors — equivalent to ``sign(a+b+c)``
    when each input is ±1. Three packed AND + two packed OR; no unpacking.
    """
    ab = brute.bitwise_and(a, b)
    ac = brute.bitwise_and(a, c)
    bc = brute.bitwise_and(b, c)
    return brute.bitwise_or(brute.bitwise_or(ab, ac), bc)


@dataclass
class BGPT1Config:
    """Hyper-parameters for a BGPT-1 model.

    See BGPT-1 spec §B (Hyperparameter defaults). Phase-1 toy values are
    used here; scale up for Phase 4+.
    """

    vocab_size:   int   = 256
    dim:          int   = 64
    n_heads:      int   = 4
    n_layers:     int   = 2
    max_context:  int   = 64
    hidden_mult:  int   = 4
    nu:           float = 0.05            # gate threshold (fraction of d)
    p_drop_max:   float = 0.1             # max LayerDrop probability
    err_clip:     int   = 7               # 4-bit signed
    err_k:        int   = 8               # top-k distractors in err signal
    # Legacy fields retained for back-compat with older configs / training
    # scripts. The INT8 LM-head bias has been removed; these are ignored.
    bias_clip:    int   = 0
    bias_accum:   int   = 0


class BGPT1(bn.Module):
    """A fully 1-bit decoder-only transformer language model.

    Forward returns ``(logits, tape)`` where ``logits`` is int32 of shape
    ``(B, C, V)`` and ``tape`` is a dict of per-block activations and
    gate flags needed by the backward pass. Backward consumes the tape,
    walks layers in reverse, and calls into ``FlipRule.flip_step`` for
    each binary linear layer.
    """

    def __init__(self, config: BGPT1Config, device=None):
        super().__init__()
        self.config = config
        self.device_ = device

        self.embedding = bn.BinaryEmbedding(
            config.vocab_size, config.dim, device=device,
        )
        self.positions = bn.BinaryPositionEmbedding(
            config.max_context, config.dim,
        )
        # Per-block LayerDrop schedule: 0 on layer 0, p_drop_max on layer L-1.
        self.blocks = bn.ModuleList()
        for li in range(config.n_layers):
            p_drop = config.p_drop_max * li / max(1, config.n_layers - 1)
            blk = bn.BinaryTransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                max_context=config.max_context,
                hidden_mult=config.hidden_mult,
                nu=config.nu,
                p_drop=p_drop,
                device=device,
            )
            self.blocks.append(blk)
        self.lm_head = bn.BinaryLMHead(self.embedding, device=device)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def n_layers(self) -> int:
        return self.config.n_layers

    # FlipRule registration. Call once after FlipRule() construction.
    def register_with_optimizer(self, optimizer: FlipRule) -> None:
        """Register every binary weight in the model with the flip-rule optimizer."""
        # Embedding (tied with LM head).
        optimizer.add_param("embedding", self.embedding.weight)
        for li, blk in enumerate(self.blocks):
            optimizer.add_param(f"blk{li}.q_proj", blk.attention.q_proj.weight)
            optimizer.add_param(f"blk{li}.k_proj", blk.attention.k_proj.weight)
            optimizer.add_param(f"blk{li}.v_proj", blk.attention.v_proj.weight)
            optimizer.add_param(f"blk{li}.out_proj", blk.attention.out_proj.weight)
            optimizer.add_param(f"blk{li}.fc1", blk.ffn.fc1.weight)
            optimizer.add_param(f"blk{li}.fc2", blk.ffn.fc2.weight)

    # Forward producing tape
    def forward(
        self,
        ids: Tensor,
        return_tape: bool = False,
        return_diag: bool = False,
        checkpoint: bool = True,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """Run the model.

        Args:
          ids: long ids of shape ``(B, C)``.
          return_tape: If True, save per-layer activations and gates needed
            for the custom backward.
          return_diag: If True, also return attention diagnostics per block.

        Returns:
          ``(logits, tape_or_none)``. logits is int32 ``(B, C, V)``.
        """
        b, c = ids.shape
        x = self.embedding(ids)                  # (B, C, d) bit1
        pos = self.positions(c)                  # (C, d) bit1
        # Token ⊙ position = XNOR — fully packed: ~(x XOR pos).
        x = brute.fast.bitwise_not(brute.bitwise_xor(x, pos))
        tape: Optional[Dict] = None
        if return_tape:
            tape = {
                "ids": ids,
                "x_after_embed": x,    # input to first block
                "blocks": [],
            }

        diag_list: List[Dict] = []
        for li, blk in enumerate(self.blocks):
            # We want explicit gates / pre / tape only when training.
            x_in = x
            # Save per-block intermediate produces. We don't use the fused
            # block forward when collecting a tape — instead, we step
            # through attention and ffn separately so we can capture
            # gates and the sub-block outputs.
            if return_tape:
                # Stochastic depth decision (use module's training flag).
                do_skip = (blk.training and blk.p_drop > 0.0
                           and brute.rand((), device=x.device).item() < blk.p_drop)
                if do_skip:
                    tape["blocks"].append({"skipped": True, "x_in": x_in})
                    continue
                # With ``checkpoint=True`` (default), we only save the
                # block input. The per-block attn_tape / ffn_tape are
                # recomputed on the fly during backward_step — see
                # ``_recompute_block_tape``. With ``checkpoint=False`` we
                # save the full per-block tape (faster backward but ~5×
                # the activation memory).
                if checkpoint:
                    attn_out, attn_diag, _ = blk.attention(
                        x, return_diag=return_diag, return_tape=False,
                    )
                    ffn_out, _ = blk.ffn(x, return_tape=False)
                    x = _majority3(x, attn_out, ffn_out)
                    tape["blocks"].append({
                        "skipped": False,
                        "x_in": x_in,
                        "attn_diag": attn_diag,
                    })
                else:
                    attn_out, attn_diag, attn_tape = blk.attention(
                        x, return_diag=return_diag, return_tape=True,
                    )
                    ffn_out, ffn_tape = blk.ffn(x, return_tape=True)
                    x = _majority3(x, attn_out, ffn_out)
                    tape["blocks"].append({
                        "skipped": False,
                        "x_in": x_in,
                        "attn_tape": attn_tape,
                        "ffn_tape": ffn_tape,
                        "attn_diag": attn_diag,
                    })
                if return_diag and attn_diag is not None:
                    diag_list.append(attn_diag)
            else:
                x, diag, _ = blk(x, return_diag=return_diag, return_tape=False)
                if return_diag and diag is not None:
                    diag_list.append(diag)

        if return_tape:
            tape["x_final"] = x

        logits = self.lm_head(x)
        if return_tape:
            tape["logits"] = logits
        out = (logits, tape) if return_tape else (logits, None)
        return out

    # Custom backward
    def backward_step(
        self,
        tape: Dict,
        targets: Tensor,
        optimizer: FlipRule,
    ) -> Dict[str, float]:
        """One BGPT-1 backward + flip step.

        Args:
          tape: Forward tape from ``forward(return_tape=True)``.
          targets: long target IDs of shape ``(B, C)``.
          optimizer: a registered ``FlipRule`` instance.

        Returns:
          A dict of scalar diagnostics for this batch.
        """
        b, c = targets.shape
        cfg = self.config
        V = cfg.vocab_size
        d = cfg.dim

        logits = tape["logits"].reshape(-1, V)               # (M=B*C, V) int32
        tgt_flat = targets.reshape(-1).to(logits.device)
        err_signed = signed_int_error(
            logits, tgt_flat, k_distractors=cfg.err_k, err_clip=cfg.err_clip,
        )  # (M, V) int8
        # Free the int32 logits — only err_signed is needed past this point.
        del logits
        tape["logits"] = None

        # LM-head bias removed in v2 (purely binary LM head).
        n_bias_updates = 0

        # Vote + flip on the embedding (= LM-head weight): x is the final
        # block output reshaped to (M, dim). err is signed int8.
        x_final = tape["x_final"]
        x_final_flat_bit1 = x_final.reshape(-1, d)
        optimizer.flip_step("embedding", x_final_flat_bit1, err_signed, batch_size_override=b * c)

        # Propagate err_signed back to (M, dim) binary err for the body.
        err_body = propagate_error(
            err_signed.float(), self.embedding.weight, use_bit_balance=True,
        )
        # Reshape to (B, C, d) bit1 for residual splitting.
        err_block = err_body.reshape(b, c, d)

        # Walk blocks in reverse. Parallel residual splits the same err
        # to all three branches (identity + attn + ffn); STE through the
        # majority-of-3 sign.
        def _g(t, shape):
            """Reshape a gate tensor if present (else pass through None)."""
            return None if t is None else t.reshape(*shape)

        for li in reversed(range(cfg.n_layers)):
            blk_tape = tape["blocks"][li]
            if blk_tape["skipped"]:
                continue
            err_branch = err_block

            # Regenerate per-block tapes on demand if forward was
            # checkpointed (default). Otherwise use the saved tapes.
            blk = self.blocks[li]
            x_in_block = blk_tape["x_in"]
            if "attn_tape" in blk_tape:
                attn_tape = blk_tape["attn_tape"]
                ffn_tape = blk_tape["ffn_tape"]
            else:
                _, _, attn_tape = blk.attention(
                    x_in_block, return_diag=False, return_tape=True,
                )
                _, ffn_tape = blk.ffn(x_in_block, return_tape=True)

            # --- FFN backward ---
            blk_ffn = blk.ffn
            h = ffn_tape["h"]
            gate_fc1 = ffn_tape["gate_fc1"]
            gate_fc2 = ffn_tape["gate_fc2"]
            hidden = blk_ffn.fc1.out_features

            optimizer.flip_step(
                f"blk{li}.fc2",
                h.reshape(-1, hidden),
                err_branch.reshape(-1, d),
                gate=_g(gate_fc2, (-1, d)),
            )
            err_h = propagate_error(
                err_branch.reshape(-1, d),
                blk_ffn.fc2.weight,
                gate=_g(gate_fc2, (-1, d)),
                use_bit_balance=True,
            ).reshape(b, c, hidden)

            optimizer.flip_step(
                f"blk{li}.fc1",
                x_in_block.reshape(-1, d),
                err_h.reshape(-1, hidden),
                gate=_g(gate_fc1, (-1, hidden)),
            )
            err_ffn_in = propagate_error(
                err_h.reshape(-1, hidden),
                blk_ffn.fc1.weight,
                gate=_g(gate_fc1, (-1, hidden)),
                use_bit_balance=True,
            ).reshape(b, c, d)

            # --- Attention backward (Phase-3 with gating; stage-1/2 STE) ---
            attn_block = blk.attention
            attn_input = attn_tape["head_out_bit1"]
            gate_out = attn_tape.get("gate_out")
            gate_q   = attn_tape.get("gate_q")
            gate_k   = attn_tape.get("gate_k")
            gate_v   = attn_tape.get("gate_v")

            optimizer.flip_step(
                f"blk{li}.out_proj",
                attn_input.reshape(-1, d),
                err_branch.reshape(-1, d),
                gate=_g(gate_out, (-1, d)),
            )
            err_head_out = propagate_error(
                err_branch.reshape(-1, d),
                attn_block.out_proj.weight,
                gate=_g(gate_out, (-1, d)),
                use_bit_balance=True,
            ).reshape(b, c, d)
            err_head_flat = err_head_out.reshape(-1, d)
            qkv_in_flat = x_in_block.reshape(-1, d)

            optimizer.flip_step(
                f"blk{li}.q_proj", qkv_in_flat, err_head_flat,
                gate=_g(gate_q, (-1, d)),
            )
            optimizer.flip_step(
                f"blk{li}.k_proj", qkv_in_flat, err_head_flat,
                gate=_g(gate_k, (-1, d)),
            )
            optimizer.flip_step(
                f"blk{li}.v_proj", qkv_in_flat, err_head_flat,
                gate=_g(gate_v, (-1, d)),
            )
            attn_diag = blk_tape.get("attn_diag")
            if attn_diag is not None:
                attn_block.tau_feedback_step(attn_diag["att_dec"])
            err_q = propagate_error(
                err_head_flat, attn_block.q_proj.weight,
                gate=_g(gate_q, (-1, d)), use_bit_balance=True,
            ).reshape(b, c, d)
            err_k = propagate_error(
                err_head_flat, attn_block.k_proj.weight,
                gate=_g(gate_k, (-1, d)), use_bit_balance=True,
            ).reshape(b, c, d)
            err_v = propagate_error(
                err_head_flat, attn_block.v_proj.weight,
                gate=_g(gate_v, (-1, d)), use_bit_balance=True,
            ).reshape(b, c, d)
            err_attn_in = _majority3(err_q, err_k, err_v)

            err_block = _majority3(err_block, err_attn_in, err_ffn_in)

        return {"n_bias_updates": float(n_bias_updates)}


__all__ = ["BGPT1", "BGPT1Config"]
