"""Parity tests for the BOLD recreation.

Verifies that:
  1. BitLinear.forward    ≡  Algorithm 5 (logical_xor reference).
  2. BitLinear.backward   ≡  Algorithms 6 (Boolean) and 7 (real).
  3. BitConv2d.forward    ≡  naive nested-loop XOR-count.
  4. BitConv2d.backward   ≡  conv_transpose2d / conv2d_weight reference.
  5. BooleanOptimizer.step ≡ Algorithm 8 line-by-line.
  6. VGGSmall builds, forwards, and trains for one step end-to-end on
     synthetic data without diverging.

Run with:

    python test_recreation.py
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from bit_linear import BitLinear, _BitLinearFunction
from bit_conv import BitConv2d, _BitConv2dFunction
from boolean_optimizer import BooleanOptimizer, split_parameters

from vgg_small import VGGSmall, count_parameters


def _check(name: str, ours: torch.Tensor, ref: torch.Tensor, atol: float = 1e-5) -> None:
    err = (ours - ref).abs().max().item()
    ok = err <= atol
    flag = "ok" if ok else "FAIL"
    print(f"  [{flag}] {name}: max |err| = {err:.3e}")
    if not ok:
        raise AssertionError(f"{name} parity failed: max err = {err:.3e}")


def test_bitlinear_forward() -> None:
    print("BitLinear forward vs. Algorithm 5 (logical_xor):")
    torch.manual_seed(0)
    B, n, m = 4, 7, 5
    X = torch.randint(0, 2, (B, n)).float()
    W = torch.randint(0, 2, (m, n)).float()
    b = torch.randint(0, 2, (m,)).float()

    # Algorithm 5 verbatim.
    S_ref = (
        torch.logical_xor(X[:, None, :].bool(), W[None, :, :].bool())
        .sum(dim=2).float()
        + b - n / 2
    )

    layer = BitLinear(n, m, bias=True, center=True)
    with torch.no_grad():
        layer.weight.copy_(W)
        layer.bias.copy_(b)
    S = layer(X)
    _check("forward", S, S_ref)


def test_bitlinear_backward_real() -> None:
    print("BitLinear backward (real Z) vs. Algorithm 7:")
    torch.manual_seed(0)
    B, n, m = 4, 7, 5
    X = torch.randint(0, 2, (B, n)).float().requires_grad_(True)
    W = torch.nn.Parameter(torch.randint(0, 2, (m, n)).float())
    b = torch.nn.Parameter(torch.randint(0, 2, (m,)).float())

    # bp_scale = 1.0 to compare against the un-scaled reference.
    S = _BitLinearFunction.apply(X, W, b, False, True, 1.0)
    Z = torch.randn(B, m)
    S.backward(Z)

    _check("G_X", X.grad, Z @ (1.0 - 2.0 * W))
    _check("G_W", W.grad, Z.t() @ (1.0 - 2.0 * X))
    _check("G_B", b.grad, Z.sum(dim=0))


def test_bitlinear_backward_bool() -> None:
    print("BitLinear backward (bool Z) vs. Algorithm 6:")
    torch.manual_seed(0)
    B, n, m = 4, 7, 5
    X = torch.randint(0, 2, (B, n)).float().requires_grad_(True)
    W = torch.nn.Parameter(torch.randint(0, 2, (m, n)).float())
    b = torch.nn.Parameter(torch.randint(0, 2, (m,)).float())

    S = _BitLinearFunction.apply(X, W, b, True, True, 1.0)
    Z = torch.randint(0, 2, (B, m)).float()
    S.backward(Z)

    G_X_ref = 2 * torch.logical_xor(Z[:, :, None].bool(), W[None, :, :].bool()).sum(dim=1).float() - m
    G_W_ref = 2 * torch.logical_xor(Z[:, :, None].bool(), X[:, None, :].bool()).sum(dim=0).float() - B
    G_B_ref = 2 * Z.sum(dim=0) - B

    _check("G_X", X.grad, G_X_ref)
    _check("G_W", W.grad, G_W_ref)
    _check("G_B", b.grad, G_B_ref)


def test_bitconv2d_forward() -> None:
    print("BitConv2d forward vs. naive XOR-count loop:")
    torch.manual_seed(0)
    B, Cin, H, Wd = 2, 3, 5, 5
    Cout, K = 4, 3
    X = torch.randint(0, 2, (B, Cin, H, Wd)).float()
    W = torch.randint(0, 2, (Cout, Cin, K, K)).float()

    layer = BitConv2d(Cin, Cout, K, padding=0, bias=False, center=True)
    with torch.no_grad():
        layer.weight.copy_(W)
    S = layer(X)

    K_total = Cin * K * K
    S_ref = torch.zeros_like(S)
    for bi in range(B):
        for co in range(Cout):
            for h in range(H - K + 1):
                for w in range(Wd - K + 1):
                    patch = X[bi, :, h:h+K, w:w+K].bool()
                    wpatch = W[co].bool()
                    S_ref[bi, co, h, w] = torch.logical_xor(patch, wpatch).sum().float() - K_total / 2
    _check("forward", S, S_ref)


def test_bitconv2d_backward_real() -> None:
    print("BitConv2d backward vs. conv_transpose2d / conv2d_weight reference:")
    torch.manual_seed(0)
    B, Cin, H, Wd = 2, 3, 7, 7
    Cout, K = 4, 3
    X = torch.randint(0, 2, (B, Cin, H, Wd)).float().requires_grad_(True)
    W = torch.nn.Parameter(torch.randint(0, 2, (Cout, Cin, K, K)).float())

    # bp_scale = 1.0 for direct comparison with the un-scaled algorithmic ref.
    S = _BitConv2dFunction.apply(X, W, None, (1, 1), (1, 1), (1, 1), 1, True, 1.0)
    Z = torch.randn_like(S)
    S.backward(Z)

    G_X_ref = F.conv_transpose2d(Z, 1.0 - 2.0 * W, stride=1, padding=1)
    G_W_ref = torch.nn.grad.conv2d_weight(1.0 - 2.0 * X, W.shape, Z, stride=1, padding=1)
    _check("G_X", X.grad, G_X_ref)
    _check("G_W", W.grad, G_W_ref)


def test_boolean_optimizer_step() -> None:
    print("BooleanOptimizer.step vs. Algorithm 8 verbatim:")
    torch.manual_seed(0)
    m_, n_ = 5, 7
    W0 = torch.randint(0, 2, (m_, n_)).float()

    W_ours = torch.nn.Parameter(W0.clone())
    opt = BooleanOptimizer([W_ours], lr=0.5)
    W_ours.grad = torch.randn(m_, n_)
    opt.step()

    ref_W = W0.clone()
    ref_accum = torch.zeros_like(ref_W)
    ref_ratio = 0.0
    ref_accum = ref_ratio * ref_accum + 0.5 * W_ours.grad
    ref_flip = ref_accum * (2 * ref_W - 1) >= 1
    ref_W[ref_flip] = torch.logical_not(ref_W[ref_flip].bool()).float()
    ref_accum[ref_flip] = 0.0

    _check("W", W_ours.data, ref_W)


def test_end_to_end_step() -> None:
    print("End-to-end smoke: VGGSmall forward + backward + step:")
    torch.manual_seed(0)
    device = torch.device("cpu")
    for use_bn in (True, False):
        model = VGGSmall(use_bn=use_bn).to(device)
        bool_params, real_params = split_parameters(model)
        bool_opt = BooleanOptimizer(bool_params, lr=12.0)
        real_opt = torch.optim.Adam(real_params, lr=1e-3)

        x = torch.randn(4, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (4,), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        assert torch.isfinite(loss), f"non-finite initial loss (use_bn={use_bn})"

        loss.backward()
        bool_opt.step()
        real_opt.step()

        # Boolean weights must still live in {0, 1} after the step.
        for p in bool_params:
            assert ((p.data == 0) | (p.data == 1)).all(), \
                f"boolean param escaped {{0, 1}} after step (use_bn={use_bn})"
        n_b, n_r = count_parameters(model)
        flips = bool_opt.nb_flips
        print(f"  [ok] use_bn={use_bn}: loss={loss.item():.3f}  "
              f"flips={flips:,}  bool={n_b:,}  real={n_r:,}")


def main() -> int:
    tests = [
        test_bitlinear_forward,
        test_bitlinear_backward_real,
        test_bitlinear_backward_bool,
        test_bitconv2d_forward,
        test_bitconv2d_backward_real,
        test_boolean_optimizer_step,
        test_end_to_end_step,
    ]
    print(f"Running {len(tests)} parity tests …\n")
    for t in tests:
        t()
        print()
    print("All BOLD parity tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
