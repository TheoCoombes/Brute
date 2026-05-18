"""Two-layer binary feed-forward network.

Stacked ``BinaryLinear`` layers (linear → bit-balanced sign → linear →
bit-balanced sign). The ``sign()`` inside ``bit_balance`` IS the
nonlinearity, so there is no explicit activation function.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import brute
from brute import nn
from brute.tensor import Tensor
from brute.nn.linear import BinaryLinear


class BinaryFFN(nn.Module):
    """Two-layer bit1 FFN with bit-balanced sign as the nonlinearity.

    Args:
      dim: Model dim ``d``.
      hidden_mult: Hidden dim is ``hidden_mult · d`` (default 4).
      nu: Gate-flag threshold for each layer (passed to BitBalancedNorm).
      device: Device for the weight buffers.

    ``forward`` returns ``(out, gates)`` where ``gates`` is a list of bit1
    gate tensors (one per layer) when ``return_gates=True``. The gates are
    used by the BGPT-1 backward pass.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 4,
        nu: float = 0.05,
        device=None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden_mult * dim)
        self.nu = float(nu)
        self.fc1 = BinaryLinear(self.dim, self.hidden, nu=nu, device=device)
        self.fc2 = BinaryLinear(self.hidden, self.dim, nu=nu, device=device)

    def forward(
        self,
        x: Tensor,
        return_tape: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """Compute the FFN output.

        Args:
          x: bit1 brute.Tensor of shape ``(..., dim)``.
          return_tape: If True, return the training tape: the hidden
            activation ``h`` (input to fc2) and the per-layer bit1 gate
            flags ``gate_fc1`` (at ``h``) and ``gate_fc2`` (at the FFN
            output). The block input ``x`` is NOT duplicated here — the
            caller already has it.

        Returns:
          ``(out, tape_or_none)``.
        """
        h, gate_fc1, _ = self.fc1(x, return_gate=return_tape, return_pre=False)
        out, gate_fc2, _ = self.fc2(h, return_gate=return_tape, return_pre=False)
        if not return_tape:
            return out, None
        return out, {"h": h, "gate_fc1": gate_fc1, "gate_fc2": gate_fc2}


__all__ = ["BinaryFFN"]
