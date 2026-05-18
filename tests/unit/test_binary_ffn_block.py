"""Unit tests for BinaryFFN and BinaryTransformerBlock."""

from __future__ import annotations

import pytest
import torch

import brute
from brute.nn import BinaryFFN, BinaryTransformerBlock


def test_ffn_runs():
    ffn = BinaryFFN(dim=256, hidden_mult=4, nu=0.05)
    x = brute.randint(0, 2, (2, 8, 256), dtype=brute.bit1)
    out, tape = ffn(x, return_tape=False)
    assert out.shape == (2, 8, 256)
    assert getattr(out, "_is_bit1", False)
    assert tape is None


def test_ffn_tape_carries_h_and_two_gates():
    ffn = BinaryFFN(dim=256, hidden_mult=4, nu=0.05)
    x = brute.randint(0, 2, (2, 8, 256), dtype=brute.bit1)
    out, tape = ffn(x, return_tape=True)
    # Slim tape: only the hidden activation (input to fc2) and the two
    # per-layer gate flags. No pres (pre-activations) — gates suffice.
    assert set(tape.keys()) == {"h", "gate_fc1", "gate_fc2"}
    assert tape["h"].shape == (2, 8, 1024)        # hidden = 4 × 256
    assert tape["gate_fc1"].shape == (2, 8, 1024)
    assert tape["gate_fc2"].shape == (2, 8, 256)
    # All bit1 — no bool / int round-trips on the tape.
    assert getattr(tape["h"], "_is_bit1", False)
    assert getattr(tape["gate_fc1"], "_is_bit1", False)
    assert getattr(tape["gate_fc2"], "_is_bit1", False)


def test_block_runs():
    blk = BinaryTransformerBlock(dim=256, n_heads=4, max_context=64, hidden_mult=4)
    x = brute.randint(0, 2, (2, 8, 256), dtype=brute.bit1)
    out, diag, tape = blk(x)
    assert out.shape == (2, 8, 256)
    assert getattr(out, "_is_bit1", False)


def test_block_skip_layer_passes_input_through():
    blk = BinaryTransformerBlock(dim=256, n_heads=4, max_context=64, p_drop=0.5)
    blk.train()
    x = brute.randint(0, 2, (2, 8, 256), dtype=brute.bit1)
    out, _, _ = blk(x, skip_layer=True)
    assert torch.equal(out.bool(), x.bool()), "skipped block should pass input through"


def test_block_eval_no_drop():
    blk = BinaryTransformerBlock(dim=256, n_heads=4, max_context=64, p_drop=1.0)
    blk.eval()
    x = brute.randint(0, 2, (2, 8, 256), dtype=brute.bit1)
    out, _, _ = blk(x)
    # Forward should not return x verbatim when p_drop applies only at training.
    # The block actually ran (output bits should differ from input bits at
    # least somewhere).
    assert not torch.equal(out.bool(), x.bool())


def test_block_diagnostic_dict_present_when_requested():
    blk = BinaryTransformerBlock(dim=256, n_heads=4, max_context=64, p_drop=0.0)
    x = brute.randint(0, 2, (1, 4, 256), dtype=brute.bit1)
    out, diag, _ = blk(x, return_diag=True)
    assert diag is not None
    assert "att_dec" in diag
