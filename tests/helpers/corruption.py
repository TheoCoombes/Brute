"""Invariant-corruption helpers for negative tests."""
from __future__ import annotations

import torch


def truncate_packed_buf(t):
    """Trim the last word from a bit1 tensor's packed buffer (corrupt size)."""
    pb = t._packed_buf
    if pb.numel() == 0:
        return t
    flat = pb.view(-1)
    truncated = flat[:-1].clone()
    t._packed_buf = truncated.view(-1)
    return t


def clear_packed_buf(t):
    """Set the packed buffer to None (force lazy recompute)."""
    t._packed_buf = None
    return t


def flip_is_bit1(t):
    """Toggle the bit1 flag (corrupt subclass invariant)."""
    t._is_bit1 = not getattr(t, "_is_bit1", False)
    return t
