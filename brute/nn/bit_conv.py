"""Boolean 2D convolution layer (BOLD).

Reference
---------
Van Minh Nguyen, Cristian Ocampo, Aymen Askri, Louis Leconte, Ba-Hien Tran.
"BOLD: Boolean Logic Deep Learning." NeurIPS 2024.
- Definitions 3.1 / 3.5     (Boolean and mixed-type neurons)
- §3.3                       (Boolean backpropagation)
- Appendix C.2, Remark C.2   (convolutional backprop variance scaling)

Design
------
A drop-in replacement for ``torch.nn.Conv2d`` whose weight (and bias) live
in the Boolean domain ``{0, 1}``. Forward is im2col + brute bit1 matmul —
i.e. the per-output-pixel receptive field is unrolled with
``F.unfold``, packed to :class:`brute.bit1`, and the per-pixel ±1 dot
product is computed by the brute XNOR-popcount kernel. Backward stays in
float so the BooleanOptimizer can read real-valued gradients (BOLD
Algorithm 7).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, autograd, nn

import brute


__all__ = ["BitConv2d"]


# Packed-weight cache (mirrors bit_linear._packed_weight) — keyed by the
# Parameter's ``_version`` so we re-pack only after BooleanOptimizer flips.
def _packed_weight_flat(W: Tensor, n_rows: int, n_cols: int) -> Tensor:
    ver = W._version
    cache = W.__dict__.get("_bold_packed_cache")
    if cache is not None and cache[0] == ver:
        return cache[1]
    flat = W.detach().reshape(n_rows, n_cols)
    packed = brute.as_tensor(flat, dtype=brute.bit1)
    W.__dict__["_bold_packed_cache"] = (ver, packed)
    return packed


_size_2_t = Union[int, Tuple[int, int]]


def _pair(x: _size_2_t) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)


def _conv_out_dim(in_size: int, k: int, pad: int, dilation: int, stride: int) -> int:
    return (in_size + 2 * pad - dilation * (k - 1) - 1) // stride + 1


class _BitConv2dFunction(autograd.Function):
    """Autograd Function for BOLD's XOR conv.

    Forward
    -------
    For every spatial location ``o``, summed over the receptive field of
    size ``K = (C_in / groups) · k_h · k_w``::

        S(o) = Σ_k XOR(x_k, w_k) - K/2 + b

    Computed as ``im2col(X) `` ``@`` (bit1) `` W_flat`` returning the
    ``K - 2H`` ±1 dot product (via brute's bit1 matmul), then mapped to
    Algorithm 5's centred / un-centred pre-activation.

    Backward
    --------
    Algorithm 7 + Remark C.2. Delegates to the highly optimised cuDNN /
    Metal kernels via ``F.conv_transpose2d`` and ``torch.nn.grad.conv2d_weight``::

        G_X = conv_transpose2d(Z, 1 − 2W)
        G_W = conv2d_weight_grad(1 − 2X, Z)
        G_B = Z.sum(batch, h, w)
    """

    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        W: Tensor,
        B: Optional[Tensor],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        center: bool,
        bp_scale: float,
    ) -> Tensor:
        ctx.save_for_backward(X, W, B if B is not None else torch.empty(0, device=W.device))
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.center = center
        ctx.bp_scale = float(bp_scale)
        ctx.has_bias = B is not None
        ctx.input_shape = X.shape

        Cout, Cin_per_group, kH, kW = W.shape
        Bsz, Cin, Hin, Win = X.shape
        Hout = _conv_out_dim(Hin, kH, padding[0], dilation[0], stride[0])
        Wout = _conv_out_dim(Win, kW, padding[1], dilation[1], stride[1])
        K = Cin_per_group * kH * kW
        out_dtype = X.dtype if X.is_floating_point() else torch.float32

        # im2col → (B, Cin·kH·kW, L) where L = Hout·Wout. F.unfold pads
        # with 0 (the natural "False" bit for our Boolean convention).
        patches = F.unfold(X, (kH, kW), dilation=dilation, padding=padding, stride=stride)
        L = patches.shape[-1]

        if groups == 1:
            # Single contiguous matmul over the full receptive field.
            # Reorder to (B·L, K) for the brute bit1 matmul convention.
            a = patches.transpose(1, 2).reshape(Bsz * L, K)
            a_bit = brute.as_tensor(a, dtype=brute.bit1)
            b_bit = _packed_weight_flat(W, Cout, K)
            km2h = (a_bit @ b_bit).to(out_dtype)                 # (B·L, Cout)
            km2h = km2h.reshape(Bsz, L, Cout).transpose(1, 2)    # (B, Cout, L)
        else:
            # Per-group bit1 matmul. patches groups along channel dim, weight
            # already split by groups: each group contributes Cout/groups output
            # channels using Cin/groups input channels.
            cout_per_group = Cout // groups
            # (B, groups, Cin/groups·kH·kW, L)
            patches_g = patches.view(Bsz, groups, Cin_per_group * kH * kW, L)
            outs = []
            for g in range(groups):
                a = patches_g[:, g].transpose(1, 2).reshape(Bsz * L, K)
                b = W[g * cout_per_group:(g + 1) * cout_per_group].reshape(cout_per_group, K)
                a_bit = brute.as_tensor(a, dtype=brute.bit1)
                b_bit = brute.as_tensor(b, dtype=brute.bit1)
                outs.append((a_bit @ b_bit).to(out_dtype).reshape(Bsz, L, cout_per_group))
            # (B, L, Cout) → (B, Cout, L)
            km2h = torch.cat(outs, dim=-1).transpose(1, 2)

        # Map matmul output to BOLD Algorithm 5's pre-activation.
        if center:
            S = -km2h * 0.5                                       # = sum_xor - K/2
        else:
            S = (K - km2h) * 0.5                                  # = sum_xor

        # The transpose(1,2) above leaves S non-contiguous, which downstream
        # BatchNorm's view() complains about. .contiguous() materialises the
        # standard (N, C, H, W) memory layout.
        S = S.reshape(Bsz, Cout, Hout, Wout).contiguous()
        if B is not None:
            S = S + B.view(1, -1, 1, 1)
        return S

    @staticmethod
    def backward(ctx, Z: Tensor):
        X, W, B_buf = ctx.saved_tensors
        has_bias = ctx.has_bias
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        # G_X = conv_transpose2d(Z, 1 − 2W). For ``stride > 1`` we need the
        # ``output_padding`` so transpose conv lands on the original input
        # spatial shape exactly.
        W_pm = 1.0 - 2.0 * W

        in_h, in_w = X.shape[-2], X.shape[-1]
        z_h, z_w = Z.shape[-2], Z.shape[-1]
        out_h_default = (z_h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (W.shape[2] - 1) + 1
        out_w_default = (z_w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (W.shape[3] - 1) + 1
        out_pad = (max(0, in_h - out_h_default), max(0, in_w - out_w_default))
        G_X = F.conv_transpose2d(Z, W_pm, None, stride, padding, out_pad, groups, dilation)
        G_X = G_X[..., :in_h, :in_w]

        # G_W = conv2d_weight_grad(1 − 2X, Z).
        X_pm = 1.0 - 2.0 * X
        G_W = torch.nn.grad.conv2d_weight(
            X_pm, W.shape, Z, stride=stride, padding=padding,
            dilation=dilation, groups=groups,
        )

        G_B = Z.sum(dim=(0, 2, 3)) if has_bias else None

        if ctx.bp_scale != 1.0:
            G_X = G_X * ctx.bp_scale

        # forward sig: (X, W, B, stride, padding, dilation, groups, center, bp_scale) -> 9 grads
        return G_X, G_W, G_B if has_bias else None, None, None, None, None, None, None


class BitConv2d(nn.Conv2d):
    """Boolean 2D convolution layer (BOLD §3.1 + §3.3).

    Same signature as :class:`torch.nn.Conv2d`, plus ``center`` for
    0-centring the XOR-count pre-activation (BN-friendly, on by default)
    and ``maxpool_after`` to pick up the extra factor of 2 in the BOLD
    backprop-variance scaling (Remark C.2) when a 2×2 max-pool sits
    directly downstream.

    Weights / bias are stored as floats in ``{0, 1}`` and MUST be optimised
    by :class:`brute.optim.BooleanOptimizer`. Forward dispatches to brute's
    bit1 XNOR-popcount matmul via ``F.unfold`` + ``brute.as_tensor(..., dtype=brute.bit1)``.

    Notes
    -----
    Only the real-valued backprop path (Algorithm 7) is implemented; this
    is the only path used in BOLD's CIFAR-10 / ImageNet experiments, since
    the conv layers always feed either a BatchNorm or a real-valued
    activation gradient upstream.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        center: bool = True,
        maxpool_after: bool = False,
        bp_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode, device=device, dtype=dtype,
        )
        self.center = center
        # Appendix C.2 / Remark C.2:
        #   no maxpool: scale = sqrt(2v / (c_out · k_x · k_y))
        #   with 2×2 maxpool: scale = 2 · sqrt(2v / (c_out · k_x · k_y))
        # v = stride, c_out = out_channels, k_x · k_y = kernel area.
        if bp_scale is None:
            kx, ky = _pair(kernel_size)
            sy, sx = _pair(stride)
            v = sy * sx
            base = math.sqrt(2.0 * v / (out_channels * kx * ky))
            self.bp_scale = (2.0 * base) if maxpool_after else base
        else:
            self.bp_scale = float(bp_scale)
        self.maxpool_after = bool(maxpool_after)
        self.weight._bold_boolean = True
        if self.bias is not None:
            self.bias._bold_boolean = True

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.copy_(
                torch.randint(0, 2, self.weight.shape, device=self.weight.device).to(self.weight.dtype)
            )
            if self.bias is not None:
                self.bias.copy_(
                    torch.randint(0, 2, self.bias.shape, device=self.bias.device).to(self.bias.dtype)
                )

    def forward(self, X: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            X = F.pad(X, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            pad = (0, 0)
        else:
            pad = _pair(self.padding) if isinstance(self.padding, (int, tuple)) else (0, 0)
        return _BitConv2dFunction.apply(
            X, self.weight, self.bias,
            _pair(self.stride), pad, _pair(self.dilation),
            self.groups, self.center, self.bp_scale,
        )

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", center={self.center}, maxpool_after={self.maxpool_after}, bp_scale={self.bp_scale:.4g}"
        )
