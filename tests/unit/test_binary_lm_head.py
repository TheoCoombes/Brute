"""Unit tests for the binary LM head and the signed-int error helpers."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn import (
    BinaryEmbedding,
    BinaryLMHead,
    signed_int_error,
    lm_head_bias_step,
)


def test_lm_head_forward_returns_int_logits():
    emb = BinaryEmbedding(num_embeddings=128, dim=64)
    head = BinaryLMHead(emb)
    h = brute.randint(0, 2, (4, 8, 64), dtype=brute.bit1)
    logits = head(h)
    assert logits.shape == (4, 8, 128)
    assert logits.dtype == torch.int32


def test_lm_head_bias_initially_zero():
    emb = BinaryEmbedding(num_embeddings=32, dim=64)
    head = BinaryLMHead(emb)
    assert (head.bias == 0).all()
    assert (head.bias_accumulator == 0).all()


def test_signed_error_zero_when_target_already_top():
    """If logits[target] is strictly greater than every other logit,
    `signed_int_error` returns the all-zero error row."""
    logits = torch.tensor([[10, 0, 0, 0]], dtype=torch.int32)
    targets = torch.tensor([0])
    err = signed_int_error(logits, targets, k_distractors=2, err_clip=7)
    assert err.shape == (1, 4)
    assert (err == 0).all()


def test_signed_error_distractor_negative_target_positive():
    """When a distractor beats the target, target err is +mag and the
    distractor's err is -mag (clipped). Tied tokens (margin = 0) count
    as distractors with magnitude 1 per the spec."""
    logits = torch.tensor([[0, 5, -3, -7]], dtype=torch.int32)
    targets = torch.tensor([0])
    err = signed_int_error(logits, targets, k_distractors=2, err_clip=7)
    # Target row: margin = max distractor margin = 5
    assert err[0, 0] == 5
    # Distractor (col 1) got -margin = -5
    assert err[0, 1] == -5
    # Cols 2 and 3 have negative margins → not distractors → zero
    assert err[0, 2] == 0 and err[0, 3] == 0


def test_signed_error_clip_caps_magnitude():
    logits = torch.tensor([[0, 50, 0, 0]], dtype=torch.int32)
    targets = torch.tensor([0])
    err = signed_int_error(logits, targets, k_distractors=2, err_clip=7)
    assert err[0, 0] == 7
    assert err[0, 1] == -7


def test_lm_head_bias_step_no_crossing_means_no_update():
    """Below threshold, accumulator changes but bias does not."""
    bias = torch.zeros(8, dtype=torch.int8)
    accum = torch.zeros(8, dtype=torch.int32)
    err = torch.zeros(2, 8, dtype=torch.int8)
    err[0, 3] = 5  # small bump, well below threshold=32
    n = lm_head_bias_step(bias, accum, err, accum_threshold=32, bias_clip=127)
    assert n == 0
    assert (bias == 0).all()
    assert accum[3] == 5


def test_lm_head_bias_step_crossing_ticks_bias():
    """When accumulator crosses threshold, bias adjusts by 1.

    ``signed_int_error`` returns +mag on the target (logit too low → raise
    bias). So positive accumulator → bias INCREMENTS.
    """
    bias = torch.zeros(8, dtype=torch.int8)
    accum = torch.zeros(8, dtype=torch.int32)
    err = torch.zeros(2, 8, dtype=torch.int8)
    err[0, 3] = 20
    err[1, 3] = 15  # total = 35 > 32 → bias[3] += 1
    n = lm_head_bias_step(bias, accum, err, accum_threshold=32, bias_clip=127)
    assert n == 1
    assert bias[3] == 1, f"bias should increment on positive err sum; got {bias[3]}"
    assert accum[3] == 35 - 32
