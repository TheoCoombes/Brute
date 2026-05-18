"""Multi-scale binary positional encoding (parity-of-prefix construction).

The binary analog of sinusoidal positional encoding. Each position ``i`` maps
to a fixed ±1 vector of length ``d``. Channels are grouped by periodicity:
the ``k``-th group flips every ``2^k`` positions. So adjacent positions agree
on most bits and distant positions disagree on roughly half — preserving the
locality prior that autoregressive language models want.

Walsh-Hadamard rows are maximally orthogonal between adjacent indices, which
destroys that locality and is the failure mode this construction sidesteps.
"""

from __future__ import annotations

from typing import Optional
import math

import brute
from brute import nn
from brute.tensor import Tensor


def _channel_scales(d: int, c: int) -> Tensor:
    """Channel-to-bit-position mapping.

    Channel ``k`` is keyed off bit ``scale(k) = k // (d / log2(C))`` of the
    position index. The first ``d / log2(C)`` channels share period 2, the
    next group period 4, etc. The mapping cycles modulo ``log2(C)``.
    """
    log2c = max(1, int(math.ceil(math.log2(max(2, c)))))
    chans_per_group = max(1, d // log2c)
    idx = brute.arange(d, dtype=brute.int64)
    return (idx // chans_per_group) % log2c


def make_binary_positions(
    c: int,
    d: int,
    device=None,
) -> Tensor:
    """Return a bit1 ``brute.Tensor`` of shape ``(c, d)`` parity-of-prefix bits.

    Bit semantics: ``True`` → +1, ``False`` → -1. Index ``[i, k]`` equals
    ``((i >> scale(k)) & 1) == 0`` (so the unsigned LSB convention gives +1
    at position 0 for every channel; adjacent positions differ only in the
    lowest-period channels).

    Args:
      c: Context length.
      d: Embedding dim.
      device: Tensor device.

    Returns:
      bit1 brute.Tensor of shape ``(c, d)``.
    """
    if c < 1 or d < 1:
        raise ValueError(f"c and d must both be >= 1; got c={c}, d={d}")
    scales = _channel_scales(d, c).to(device)
    pos = brute.arange(c, dtype=brute.int64, device=device).unsqueeze(1)
    bits_bool = ((pos >> scales) & 1) == 0
    return brute.as_tensor(bits_bool, dtype=brute.bit1, device=device)


class BinaryPositionEmbedding(nn.Module):
    """Fixed multi-scale binary positional embedding (no learned params).

    A bit1 brute.Tensor of shape ``(max_context, dim)`` is materialized at
    construction and stored as a buffer. Forward returns a row-sliced view.

    Args:
      max_context: Maximum sequence length supported.
      dim: Embedding dim.
    """

    def __init__(self, max_context: int, dim: int):
        super().__init__()
        self.max_context = int(max_context)
        self.dim = int(dim)
        positions = make_binary_positions(self.max_context, self.dim)
        self.register_buffer("positions", positions)

    def forward(self, seq_len: int) -> Tensor:
        """Return positions ``[0, seq_len)`` as a bit1 brute.Tensor."""
        if seq_len > self.max_context:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_context={self.max_context}"
            )
        return self.positions[:seq_len]

    def extra_repr(self) -> str:
        return f"max_context={self.max_context}, dim={self.dim}"


__all__ = ["BinaryPositionEmbedding", "make_binary_positions"]
