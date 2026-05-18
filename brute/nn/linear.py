"""Binary linear layers.

``BruteLinear`` — the low-level binary matmul wrapper. Returns the int32
bipolar dot-product (``K − 2·Hamming``) for downstream non-linearities.
Weight is a ``brute.Tensor`` of dtype ``brute.bit1`` stored under
``self.weight``.

``BinaryLinear`` — convenience wrapper that fuses ``BruteLinear`` with a
``BitBalancedNorm`` for the canonical "binary linear → bit-balanced sign"
sub-block used in the BGPT-1 FFN and QKV/output projections.
"""

from __future__ import annotations

from typing import Optional, Tuple

import brute
from brute import nn
from brute.tensor import Tensor
from brute.nn.binary_norm import BitBalancedNorm


class BruteLinear(nn.Module):
    """Binary linear layer: ``y = x @ W`` with bit1 weight stored as ``(N, K)``.

    Both ``x`` and ``W`` are bit1; the kernel computes
    ``popcount(XNOR(x, W)) − K/2 = (1/2)·(K − 2·Hamming)``. We return the raw
    ``K − 2·Hamming`` int32 — i.e. twice the spec's pre-activation. Halving
    that just shifts the gate threshold by a constant, so we keep the kernel
    output directly.

    Args:
      in_features: K (input bit width).
      out_features: N (output unit count).
      bias: If True, add a learned int32 bias.
      device: Device for the weight buffer.

    Storage: ``self.weight`` is a ``brute.Tensor`` of dtype ``brute.bit1``
    and shape ``(N, K)``. The bit1 matmul convention does
    ``x @ W = x_(M×K) @ W_(N×K)`` mapping to a ``(M, N)`` int32 output (rows
    of ``W`` are output features), so no transpose is needed.
    """

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_features  = int(in_features)
        self.out_features = int(out_features)
        weight = brute.randint(
            0, 2, (self.out_features, self.in_features),
            dtype=brute.bit1, device=device,
        )
        self.register_buffer("weight", weight)
        if bias:
            # int32 buffer (not nn.Parameter — gradient-less training).
            self.register_buffer(
                "bias",
                brute.zeros(out_features, dtype=brute.int32, device=device),
            )
        else:
            self.register_buffer("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``y = x @ W``.

        Args:
          x: bit1 brute.Tensor of shape ``(..., K)`` (or any tensor whose
            last dim is K; non-bit1 inputs are converted via ``> 0``).

        Returns:
          int32 brute.Tensor of shape ``(..., N)`` carrying ``K − 2H``.
          Caller decides how to binarize (e.g. via ``BitBalancedNorm``).
        """
        if not isinstance(x, Tensor) or not getattr(x, "_is_bit1", False):
            # Coerce non-bit1 input to bit1 (last dim must be K).
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"input last dim {x.shape[-1]} != in_features {self.in_features}"
                )
            x = brute.as_tensor(x, dtype=brute.bit1, device=self.weight.device)

        leading = list(x.shape[:-1])
        x_flat = x.reshape(-1, self.in_features)
        # The bit1 @ bit1 fast path dispatches xnor_popcount_matmul: rows of
        # W are output features, so `x @ W` gives the (M, N) bipolar score.
        out_flat = x_flat @ self.weight
        if self.bias is not None:
            out_flat = out_flat + self.bias
        return out_flat.reshape(*leading, self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


class BinaryLinear(nn.Module):
    """Binary linear followed by bit-balanced sign.

    The canonical BGPT-1 sub-block: ``y = sign(x @ W − median(x @ W))`` with
    an optional 1-bit gate flag indicating which output neurons sit near the
    decision boundary (needed by the BGPT-1 backward pass).

    Args:
      in_features: K.
      out_features: N.
      nu: Gate threshold as fraction of N. Set to 0 to disable gate output
        and the per-neuron magnitude comparator at inference.
      device: Device for the weight buffer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nu: float = 0.05,
        device=None,
    ):
        super().__init__()
        self.linear = BruteLinear(in_features, out_features, bias=False, device=device)
        self.norm = BitBalancedNorm(nu=nu, dim=-1)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    def forward(
        self,
        x: Tensor,
        return_gate: bool = False,
        return_pre: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Compute ``(out, gate, pre)``.

        Args:
          x: bit1 brute.Tensor input, shape ``(..., K)``.
          return_gate: If True, also return per-element gate flags (bit1).
          return_pre: If True, also return the integer pre-activation
            tensor. Useful for diagnostics (``pre_median``, ``pre_variance``)
            and for the backward pass (gate recomputation).

        Returns:
          out: bit1 brute.Tensor of shape ``(..., N)``.
          gate: bit1 brute.Tensor or ``None``.
          pre: int32 torch.Tensor (or None).
        """
        pre = self.linear(x)
        out, gate = self.norm(pre, return_gate=return_gate)
        pre_out = pre if return_pre else None
        return out, gate, pre_out


__all__ = ["BruteLinear", "BinaryLinear"]
