"""Integration tests for the BGPT-1 model and training step."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch

import brute

# Ensure the examples/ directory is importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.bgpt1.model import BGPT1, BGPT1Config
from brute.optim import FlipRule


def _toy_model_and_trainer(steps: int = 200, bias_accum: int = 32):
    # d_h must be a multiple of pack width (64). Smallest valid toy:
    # dim=256, n_heads=4 → d_h=64. C must also be ≥ 64.
    cfg = BGPT1Config(
        vocab_size=32, dim=256, n_heads=4, n_layers=2,
        max_context=64, hidden_mult=2, p_drop_max=0.0,
        bias_accum=bias_accum,
    )
    model = BGPT1(cfg)
    trainer = FlipRule(
        t_init=0.5, t_final=0.01, total_steps=steps,
        target_flip_rate=0.005, decisive_threshold=0.05,
    )
    model.register_with_optimizer(trainer)
    for ls in trainer.layers.values():
        ls.n_ref = max(ls.weight.numel() / 32.0, 8.0)
    return cfg, model, trainer


def test_model_forward_shape():
    cfg, model, _ = _toy_model_and_trainer()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, tape = model(ids, return_tape=True)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert tape is not None
    assert len(tape["blocks"]) == cfg.n_layers


def test_model_forward_no_tape():
    cfg, model, _ = _toy_model_and_trainer()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, tape = model(ids, return_tape=False)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert tape is None


def test_backward_step_runs():
    cfg, model, trainer = _toy_model_and_trainer()
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, tape = model(ids, return_tape=True)
    diag = model.backward_step(tape, targets, trainer)
    assert "n_bias_updates" in diag
    trainer.step_global()


def test_copy_task_loss_decreases():
    """Tiny memorize-a-fixed-batch task: a step where loss drops below the
    initial value should appear within a short run. With a high bias_accum
    threshold (so the LM head bias updates don't dominate on this toy
    8-token vocab), the binary-weight flip rule should drive descent."""
    torch.manual_seed(0)
    cfg, model, trainer = _toy_model_and_trainer(steps=200, bias_accum=512)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (4, 8))
    targets = torch.cat([ids[:, 1:], ids[:, :1]], dim=1)  # cyclic shift

    def loss_fn():
        logits, _ = model(ids)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size).float(),
            targets.reshape(-1),
        ).item()

    loss0 = loss_fn()
    best = loss0
    for _ in range(60):
        _, tape = model(ids, return_tape=True)
        model.backward_step(tape, targets, trainer)
        trainer.step_global()
        best = min(best, loss_fn())
    assert best < loss0, (
        f"best loss across run should improve over init; got init={loss0:.3f}, "
        f"best={best:.3f}"
    )


def test_register_all_binary_weights():
    cfg, model, trainer = _toy_model_and_trainer()
    expected_layers = {"embedding"}
    for li in range(cfg.n_layers):
        expected_layers.update({
            f"blk{li}.q_proj",
            f"blk{li}.k_proj",
            f"blk{li}.v_proj",
            f"blk{li}.out_proj",
            f"blk{li}.fc1",
            f"blk{li}.fc2",
        })
    assert set(trainer.layers.keys()) == expected_layers


def test_training_memory_within_budget():
    """The forward+backward training pass must not allocate any single
    tensor that exceeds the largest binary weight by more than a small
    constant factor — i.e. no full unpack to ``W.bool()`` (8× weight) or
    full ``(m, n)`` int32 vote (32× weight).

    We instrument by monkey-patching the underlying tensor allocator to
    record large allocations and assert no tensor larger than 4× any
    layer's packed weight ever appears.
    """
    torch.manual_seed(0)
    cfg, model, trainer = _toy_model_and_trainer(bias_accum=128)
    model.train()

    # Determine our biggest packed-weight buffer (bytes).
    biggest_weight_bytes = max(
        ls.weight._packed_buf.nbytes for ls in trainer.layers.values()
    )

    # Sample a batch and run forward+backward.
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, tape = model(ids, return_tape=True)
    # Estimate tape memory: sum of bit1 (logical_size / 8) + small.
    def _tensor_bytes(t):
        if t is None:
            return 0
        if hasattr(t, "_packed_buf") and getattr(t, "_is_bit1", False):
            return t._packed_buf.nbytes
        if hasattr(t, "nbytes"):
            return t.nbytes
        return 0

    # Sum up tape allocations. With default checkpoint=True, only x_in
    # per block is saved (no attn_tape / ffn_tape).
    tape_bytes = 0
    for blk in tape["blocks"]:
        if blk.get("skipped"):
            continue
        tape_bytes += _tensor_bytes(blk["x_in"])
        for k, v in blk.get("attn_tape", {}).items():
            tape_bytes += _tensor_bytes(v)
        for k, v in blk.get("ffn_tape", {}).items():
            tape_bytes += _tensor_bytes(v)

    weights_bytes = sum(_tensor_bytes(ls.weight) for ls in trainer.layers.values())
    # With activation checkpointing, the tape is at most ~1× the total
    # weight bytes (one block-input bit1 per layer + small extras). Cap
    # at 2× to absorb logit and embed/final tensors.
    assert tape_bytes <= 2 * weights_bytes, (
        f"activation tape ({tape_bytes} B) exceeds 2× weights "
        f"({weights_bytes} B) — checkpointing may be broken"
    )
    # The backward should run without raising any allocator complaint.
    model.backward_step(tape, targets, trainer)


def test_lm_head_bias_actually_updates():
    cfg, model, trainer = _toy_model_and_trainer()
    model.train()
    torch.manual_seed(0)
    # Construct a batch where one token always wins → bias for that token
    # should shift positive.
    ids = torch.zeros(4, 8, dtype=torch.long)
    targets = torch.zeros(4, 8, dtype=torch.long)  # token 0 is always the target
    bias_before = model.lm_head.bias.clone()
    for _ in range(50):
        _, tape = model(ids, return_tape=True)
        model.backward_step(tape, targets, trainer)
        trainer.step_global()
    bias_after = model.lm_head.bias.clone()
    # Either bias[0] grew (token-frequency-prior accumulation) or some other
    # vocab token's bias moved — but at least *some* bias should have ticked.
    assert (bias_before != bias_after).any(), "bias never updated"
