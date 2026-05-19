"""Boolean linear layer (BOLD).

Reference
---------
Van Minh Nguyen, Cristian Ocampo, Aymen Askri, Louis Leconte, Ba-Hien Tran.
"BOLD: Boolean Logic Deep Learning." NeurIPS 2024.
- Definitions 3.1 / 3.5  (Boolean and mixed-type neurons)
- Algorithm 4 / 5         (XOR linear layer forward + autograd wiring)
- Algorithm 6 / 7         (Boolean / real backpropagation)
- Appendix C.2            (backprop variance scaling)

Design
------
A drop-in replacement for ``torch.nn.Linear`` where the weight (and bias) live
in the Boolean domain ``{0, 1}`` and the forward operator is XOR / count
rather than multiply / sum. The output is a real-valued pre-activation
identical (up to an additive constant) to the standard BNN sign-domain dot
product, suitable for downstream BatchNorm or Boolean activation.

The forward path runs natively on :class:`brute.bit1` packed tensors,
dispatching to the XNOR-popcount kernel via ``brute``'s high-level API
(``brute.as_tensor(..., dtype=brute.bit1)`` + ``@``). The backward path
stays in float so the BooleanOptimizer can read real-valued gradients —
matching BOLD Algorithm 7's "real-received-signal" form exactly.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, autograd, nn

import brute


__all__ = ["BitLinear"]


# ── Packed-weight cache ──────────────────────────────────────────────────────
#
# Re-packing the (m, n) Boolean weight from float to bit1 on every forward
# call is a real cost — it's a small but per-iter overhead that dominates for
# tiny inner-dim layers. The weight only changes when BooleanOptimizer flips
# bits, so cache the packed form keyed by ``tensor._version`` (PyTorch's
# built-in mutation counter — bumped by any in-place op including the
# optimizer's ``param.data[mask] = ...``).

def _packed_weight(W: Tensor) -> Tensor:
    ver = W._version
    cache = W.__dict__.get("_bold_packed_cache")
    if cache is not None and cache[0] == ver:
        return cache[1]
    packed = brute.as_tensor(W.detach(), dtype=brute.bit1)
    W.__dict__["_bold_packed_cache"] = (ver, packed)
    return packed


# Custom autograd Function
#
# Forward (BOLD Algorithm 5):
#     S = sum_k XOR(x_k, w_k) - n/2 + b
#
# We compute this via brute's bit1 matmul which returns ``K - 2H`` (the
# ±1-domain dot product), so:
#     S = sum_xor - n/2 = -(K - 2H) / 2 = -matmul/2
# in the centred form, and ``S = (K - matmul) / 2`` un-centred.
#
# Backward (BOLD Algorithm 6 / 7) — see the helper functions below.

class _BitLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, X: Tensor, W: Tensor, B: Optional[Tensor],
                bool_bprop: bool, center: bool, bp_scale: float) -> Tensor:
        ctx.save_for_backward(X, W, B if B is not None else torch.empty(0, device=W.device))
        ctx.bool_bprop = bool_bprop
        ctx.center = center
        ctx.bp_scale = float(bp_scale)
        ctx.has_bias = B is not None

        out_dtype = X.dtype if X.is_floating_point() else torch.float32
        n = W.shape[1]

        # Pack weight and input into bit1 and run the packed XNOR-popcount matmul.
        # Convention: brute matmul takes A(M, K) and B(N, K) (both packed on
        # the last dim) and returns int32 (M, N) with value ``K - 2H_{i,j}``,
        # where H is the Hamming distance — exactly the ±1-domain dot product.
        X_flat = X.reshape(-1, n)
        X_bit = brute.as_tensor(X_flat, dtype=brute.bit1)
        W_bit = _packed_weight(W)
        km2h = (X_bit @ W_bit).to(out_dtype)                # (B, m), int32 → float

        # Map matmul output to Algorithm 5's pre-activation.
        if center:
            S = -km2h * 0.5                                  # = sum_xor - n/2
        else:
            S = (n - km2h) * 0.5                             # = sum_xor

        if B is not None:
            S = S + B
        # Restore leading dims of the input. Use reshape (not view) because
        # the bit1 matmul output may not be stride-compatible with view.
        S = S.reshape(*X.shape[:-1], W.shape[0])
        return S

    @staticmethod
    def backward(ctx, Z: Tensor):
        X, W, B_buf = ctx.saved_tensors
        has_bias = ctx.has_bias

        bool_bprop = ctx.bool_bprop
        if bool_bprop:
            G_X, G_W, G_B = _backward_bool(X, W, B_buf if has_bias else None, Z)
        else:
            G_X, G_W, G_B = _backward_real(X, W, B_buf if has_bias else None, Z)
        # Appendix C.2: variance-stabilising scale on the OUTGOING upstream
        # signal (G_X only — weight gradient is per-layer-local).
        if ctx.bp_scale != 1.0:
            G_X = G_X * ctx.bp_scale
        # forward sig: (X, W, B, bool_bprop, center, bp_scale) -> 6 grads expected
        return G_X, G_W, G_B if has_bias else None, None, None, None


def _backward_real(X: Tensor, W: Tensor, B: Optional[Tensor], Z: Tensor):
    # Algorithm 7. Z is real-valued; weights & inputs are Boolean {0, 1}.
    # XOR(Real, Bool) reduces to Real·(1 - 2·Bool) (Prop A.3 & Def 3.5).
    Z2 = Z.reshape(-1, Z.shape[-1])
    X2 = X.reshape(-1, X.shape[-1])

    # G_X = Z @ (1 - 2W)
    G_X = Z2 @ (1.0 - 2.0 * W)
    G_X = G_X.reshape(*Z.shape[:-1], W.shape[1])

    # G_W = Z^T @ (1 - 2X). Aggregates over the batch dim.
    G_W = Z2.t() @ (1.0 - 2.0 * X2)

    # G_B = Z.sum(batch)
    G_B = Z2.sum(dim=0) if B is not None else None
    return G_X, G_W, G_B


def _backward_bool(X: Tensor, W: Tensor, B: Optional[Tensor], Z: Tensor):
    # Algorithm 6. Z is Boolean {0, 1}. Variation propagates by XOR, then
    # aggregates as "TRUEs - FALSEs = 2·TRUEs - TOT". We use the algebraic
    # identity ``XOR(z, w) = z + w - 2 z w`` to avoid the O(B·m·n) outer
    # broadcast.
    Z2 = Z.reshape(-1, Z.shape[-1])
    X2 = X.reshape(-1, X.shape[-1])
    Bsz, m = Z2.shape
    n = W.shape[1]

    sum_Z_out = Z2.sum(dim=1, keepdim=True)            # (B, 1)
    sum_W_in  = W.sum(dim=0).unsqueeze(0)              # (1, n)
    xor_zw_sum = sum_Z_out + sum_W_in - 2.0 * Z2 @ W   # (B, n)
    G_X = (2.0 * xor_zw_sum - m).reshape(*Z.shape[:-1], n)

    sum_Z_batch = Z2.sum(dim=0).unsqueeze(1)           # (m, 1)
    sum_X_batch = X2.sum(dim=0).unsqueeze(0)           # (1, n)
    xor_zx_sum = sum_Z_batch + sum_X_batch - 2.0 * Z2.t() @ X2  # (m, n)
    G_W = 2.0 * xor_zx_sum - Bsz

    G_B = 2.0 * Z2.sum(dim=0) - Bsz if B is not None else None
    return G_X, G_W, G_B


class BitLinear(nn.Linear):
    """Boolean linear layer (BOLD Algorithm 4).

    Identical signature to :class:`torch.nn.Linear` plus three flags:

    - ``bool_bprop`` (default ``False``): if ``True``, the backward path
      expects a Boolean (0/1) received signal and aggregates via XOR
      (Algorithm 6). If ``False``, the backward path is the real-valued
      variant (Algorithm 7), which is what you want any time a downstream
      module (BatchNorm, real activation, real layer) is involved.

    - ``center`` (default ``True``): 0-centre the XOR-count pre-activation
      by subtracting ``in_features / 2``. The paper centres for BN
      compatibility; turn it off to recover the raw count.

    - ``bp_scale`` (default ``sqrt(2/out_features)``): Appendix C.2
      variance-stabilising scale on the outgoing upstream gradient. Pass
      ``1.0`` to disable.

    Weights and bias are stored as float tensors taking values in
    ``{0.0, 1.0}`` (the "T/F as 0/1" embedding from Appendix A.1).
    Forward dispatches to the bit1 XNOR-popcount matmul via brute's
    ``brute.as_tensor(..., dtype=brute.bit1)`` + ``@`` high-level API.
    They MUST be optimised by :class:`brute.optim.BooleanOptimizer`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bool_bprop: bool = False,
        center: bool = True,
        bp_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.bool_bprop = bool_bprop
        self.center = center
        # Appendix C.2: default ``sqrt(2/m)`` keeps Var(Z^{l-1}) ≈ Var(Z^l)
        # through the layer (m = out_features). Pass ``bp_scale=1.0`` to
        # disable, or any other float to override.
        self.bp_scale = math.sqrt(2.0 / out_features) if bp_scale is None else float(bp_scale)
        # Tag for the BooleanOptimizer auto-discovery (see brute.optim).
        self.weight._bold_boolean = True
        if self.bias is not None:
            self.bias._bold_boolean = True

    def reset_parameters(self) -> None:
        # Boolean Bernoulli(0.5) init (Algorithm 4) — uniform random in {0, 1}.
        # Stored as the underlying float dtype so vanilla optimizers don't
        # error on type mismatch.
        with torch.no_grad():
            self.weight.copy_(
                torch.randint(0, 2, self.weight.shape, device=self.weight.device).to(self.weight.dtype)
            )
            if self.bias is not None:
                self.bias.copy_(
                    torch.randint(0, 2, self.bias.shape, device=self.bias.device).to(self.bias.dtype)
                )

    def forward(self, X: Tensor) -> Tensor:
        return _BitLinearFunction.apply(
            X, self.weight, self.bias, self.bool_bprop, self.center, self.bp_scale,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bool_bprop={self.bool_bprop}, "
            f"center={self.center}, bp_scale={self.bp_scale:.4g}"
        )
