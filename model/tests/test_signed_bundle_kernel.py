"""Stage-B kernel parity — ``brute.signed_bundle`` vs the reference.

The soft value-combine ``oᵢ = sign(Σⱼ wᵢⱼ·pm1(vⱼ))`` is the one int×bit1→bit1
reduction ``brute`` does not get from XNOR-popcount, so it ships as a tri-backend
kernel.  These tests assert the compiled kernel matches the Stage-A reference
bit-for-bit on every backend that runs here (CPU always; MPS when present).  They
are skipped automatically until the extension carrying the op has been rebuilt
(``pip install --no-build-isolation -ve .``).  The CUDA path is compile-checked
only on this machine (no NVIDIA GPU) and is not exercised here.
"""

from __future__ import annotations

import pytest
import torch

import brute
from attention import signed_bundle, _kernel_available


pytestmark = pytest.mark.skipif(not _kernel_available(),
                                reason="brute.signed_bundle kernel not built (Stage B)")


def _ref(W, V):
    return signed_bundle(W, V, prefer_kernel=False)


def _ker(W, V):
    return signed_bundle(W, V, prefer_kernel=True)


def _rand(B, M, N, D, *, lo=-4, hi=5, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randint(lo, hi, (B, M, N), generator=g, dtype=torch.int8)
    Vb = brute.as_tensor(torch.randint(0, 2, (B, N, D), generator=g).bool(), dtype=brute.bit1)
    if device != "cpu":
        W = W.to(device)
        Vb = Vb.to(device)
    return W, Vb


@pytest.mark.parametrize("shape", [(1, 1, 1, 64), (2, 5, 7, 64), (3, 8, 8, 128),
                                   (1, 16, 16, 192), (2, 4, 33, 256)])
def test_kernel_matches_reference_cpu(shape):
    W, V = _rand(*shape)
    assert torch.equal(_ker(W, V).unpack_pm1(), _ref(W, V).unpack_pm1()), shape


def test_kernel_handles_int8_saturation_cpu():
    # weighted sum that overflows int8 — kernel and reference must saturate alike.
    W, V = _rand(2, 4, 200, 64, lo=1, hi=2)     # all +1 weights, |sum| up to 200
    assert torch.equal(_ker(W, V).unpack_pm1(), _ref(W, V).unpack_pm1())


def test_kernel_transposed_backward_use_cpu():
    """The backward value path calls the kernel with the transposed weights."""
    W, V = _rand(2, 9, 9, 128)
    Wt = W.transpose(1, 2).contiguous()
    assert torch.equal(_ker(Wt, V).unpack_pm1(), _ref(Wt, V).unpack_pm1())


def test_kernel_zero_weights_sign_is_positive_cpu():
    W = torch.zeros(1, 3, 5, dtype=torch.int8)
    V = brute.as_tensor(torch.randint(0, 2, (1, 5, 64)).bool(), dtype=brute.bit1)
    out = _ker(W, V)
    assert bool(out.bool().all()), "sign(0) must be +1 everywhere for zero weights"


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
@pytest.mark.parametrize("shape", [(2, 5, 7, 64), (3, 8, 8, 128), (2, 4, 33, 256)])
def test_kernel_matches_reference_mps(shape):
    W, V = _rand(*shape, device="mps")
    out_k = _ker(W, V).to("cpu")
    out_r = _ref(W, V).to("cpu")
    assert torch.equal(out_k.unpack_pm1(), out_r.unpack_pm1()), shape


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_kernel_cpu_mps_agree():
    W, V = _rand(2, 6, 6, 128)
    out_cpu = _ker(W, V)
    out_mps = _ker(W.to("mps"), V.to("mps")).to("cpu")
    assert torch.equal(out_cpu.unpack_pm1(), out_mps.unpack_pm1())
