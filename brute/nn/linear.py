"""Neural-network building blocks for bit1 weights.

`BruteLinear` is the killer use case: a fully-binary linear layer where the
weight is packed once at module-construction time and the matmul calls
`torch.ops.brute.xnor_popcount_matmul` directly. This skips:

  * Per-call `_pack_bool` of a fresh weight tensor.
  * The `__torch_function__` matmul-detection branch.
  * Any conversion between bit1 and bool when the weight is stored bit1.

Inputs may be plain bool/uint8/float tensors; they're packed transparently on
the forward call. For inference loops where the input is also a bit1 result,
no Python-level packing happens.
"""

from __future__ import annotations

from typing import Optional
import torch
from torch import nn

from brute.dtype import _PACK_WIDTH, _PACK_STORAGE_DTYPE
from brute.tensor import Tensor, _pack_bool


class BruteLinear(nn.Module):
    """Binary linear layer: y = K - 2 * popc_xor(x_packed, w_packed).

    Matches the semantic of a (-1/+1) bipolar dot-product (the same as a
    BitNet/XNOR-Net inference layer). Bias is optional and applied as a
    plain int32 add.

    Args:
      in_features: Input bit count K.
      out_features: Output feature count N.
      bias: Whether to add a learned int32 bias to the output.
      device: Device for the weight buffer.

    The weight is stored as a packed `(N, ceil(K/64))` int64 buffer so the
    matmul kernel can be invoked directly. Use `set_weight_from_bool(...)` to
    initialise from a bool / uint8 / +-1 float tensor.
    """

    in_features:  int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features  = int(in_features)
        self.out_features = int(out_features)
        self._k_words    = (self.in_features + _PACK_WIDTH - 1) // _PACK_WIDTH

        init_bool = torch.empty(
            (self.out_features, self.in_features), dtype=torch.bool, device=device,
        ).bernoulli_(0.5)
        packed = _pack_bool(init_bool)
        # Register as a buffer so .to(device) / state_dict serialisation works.
        self.register_buffer('weight_packed', packed)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features,
                                                 dtype=torch.int32, device=device))
        else:
            self.register_parameter('bias', None)

    def set_weight_from_bool(self, bool_w: torch.Tensor) -> None:
        """Replace `weight_packed` from a bool tensor of shape (N, K)."""
        if bool_w.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape mismatch: got {tuple(bool_w.shape)}, "
                f"expected {(self.out_features, self.in_features)}"
            )
        if bool_w.dtype != torch.bool:
            bool_w = bool_w.bool()
        self.weight_packed = _pack_bool(bool_w.contiguous()).to(self.weight_packed.device)

    def set_weight_from_pm1(self, pm1_w: torch.Tensor) -> None:
        """Replace `weight_packed` from a +1/-1 float tensor of shape (N, K)."""
        self.set_weight_from_bool(pm1_w > 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: returns a plain int32 tensor of shape (..., out_features).

        Input x:
          * brute.Tensor (bit1) with shape (..., K) — used directly.
          * torch.bool / torch.uint8 / float (treated as +/- via >0)
            with shape (..., K) — packed transparently.

        The output is int32 (the dot-product C = K - 2*H), suitable as input
        to a non-linearity (sign, tanh, popcount-cmp). It is *not* re-packed
        to bit1 — callers that want a binary activation should do
        `(out > 0).to(brute.bit1)` themselves.
        """
        if isinstance(x, Tensor) and getattr(x, '_is_bit1', False):
            leading = list(x.shape[:-1])
            packed_in = x._packed_buf.reshape(-1, self._k_words).contiguous()
        else:
            xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            if xt.shape[-1] != self.in_features:
                raise ValueError(
                    f"input last dim {xt.shape[-1]} != in_features {self.in_features}"
                )
            leading = list(xt.shape[:-1])
            if xt.dtype == torch.bool:
                bool_in = xt
            elif xt.is_floating_point():
                bool_in = xt > 0
            else:
                bool_in = xt != 0
            packed_in = _pack_bool(bool_in.reshape(-1, self.in_features).contiguous())

        out = torch.ops.brute.xnor_popcount_matmul(
            packed_in, self.weight_packed, self.in_features)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*leading, self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


__all__ = ["BruteLinear"]
