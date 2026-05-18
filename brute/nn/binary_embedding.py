"""Binary embedding table — packed ±1 lookup over a vocabulary.

The embedding stores a ``brute.Tensor`` of dtype ``brute.bit1`` with shape
``(num_embeddings, dim)``. Forward returns a bit1 ``brute.Tensor`` view of
the gathered rows; index_select on a bit1 tensor slices the packed buffer
along its outer axis without ever touching the bool view.

The buffer participates as ordinary binary weights under the BGPT-1 flip
rule — flipping rows is a single XOR against a bit1 mask.
"""

from __future__ import annotations

from typing import Optional

import brute
from brute import nn
from brute.tensor import Tensor


class BinaryEmbedding(nn.Module):
    """Binary lookup table with ``num_embeddings`` rows of dim ``dim``.

    Args:
      num_embeddings: Vocab / table size.
      dim: Embedding width in bits.
      device: Device for the storage buffer.

    The table is registered as a buffer named ``weight`` so the BGPT-1 flip
    rule (which operates on packed buffers and looks up parameters by
    name) can find it via ``module.weight``.
    """

    num_embeddings: int
    dim: int

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        device=None,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        weight = brute.randint(
            0, 2, (self.num_embeddings, self.dim),
            dtype=brute.bit1, device=device,
        )
        self.register_buffer("weight", weight)

    def forward(self, ids: Tensor) -> Tensor:
        """Look up ``ids`` and return a bit1 brute.Tensor of gathered rows."""
        return self.weight[ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, dim={self.dim}"


__all__ = ["BinaryEmbedding"]
