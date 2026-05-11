import torch
from brute.module import Module
from brute.tensor import Tensor, bit1


class Linear(Module):
    """
    1-bit linear layer.

    Stores full-precision (float32) latent weights. On every forward pass the
    weights are binarised (+1/-1) and packed into a bit1 tensor; the XNOR-
    popcount matmul is then used for the computation.

    The bias (if enabled) is kept in full precision and added after the binary
    matmul, matching the BitNet / XNOR-Net convention.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 pack_dtype: str = 'uint8'):
        super().__init__()
        self._in_features  = in_features
        self._out_features = out_features
        self._use_bias     = bias
        self._pack_dtype   = pack_dtype

        self.register_parameter('weight', Tensor(torch.randn(out_features, in_features) * 0.02))
        if bias:
            self.register_parameter('bias', Tensor(torch.zeros(out_features)))

    def forward(self, x: Tensor) -> Tensor:
        w_fp = self._params['weight']._data          # float32 latent weights
        w_bin = Tensor(w_fp, dtype=bit1, pack_dtype=self._pack_dtype)

        if x._is_bit1:
            # Both activations and weights are binary — use XNOR popcount.
            out = x @ w_bin   # returns int32 torch.Tensor
            out = out.float()
        else:
            # Float activations, binary weights — unpack weights and do fp matmul.
            out = x._data @ w_bin.unpack().t()   # float32 result

        if self._use_bias:
            out = out + self._params['bias']._data

        return Tensor(out)

    def extra_repr(self) -> str:
        return (f"in_features={self._in_features}, out_features={self._out_features}, "
                f"bias={self._use_bias}, pack_dtype={self._pack_dtype}")
