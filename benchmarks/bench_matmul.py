"""
Matrix multiplication benchmarks: brute XNOR-popcount matmul vs float32.

  brute_xnor  a @ b on bit1 tensors — dispatches to xnor_popcount_matmul
  float32_ref a_f @ b_f.t() on float32 — torch baseline
  pm1_ref     ±1 float matmul — semantically equivalent to xnor_popcount_matmul

Shape convention: A is (M, K) and B is (N, K); result is (M, N).
The xnor_popcount_matmul computes A @ B^T in {-1,+1} encoding, returning int32.

Run:
    pip install pytest-benchmark
    pytest benchmarks/bench_matmul.py -v --benchmark-sort=name
    pytest benchmarks/bench_matmul.py -v --benchmark-json=results_matmul.json
"""

import pytest
import torch
import brute


# ── Shapes ────────────────────────────────────────────────────────────────────
# Each entry is (M, K, N): A=(M,K), B=(N,K), result=(M,N)

SHAPES = pytest.mark.parametrize(
    "mnk",
    [
        (128,  128,  128),
        (512,  512,  512),
        (1024, 1024, 1024),
        (2048, 1024, 2048),
    ],
    ids=["128", "512", "1024", "2048x1024"],
)


# ── brute XNOR-popcount ───────────────────────────────────────────────────────

@SHAPES
def test_xnor_matmul_brute(benchmark, device, synced, mnk):
    m, k, n = mnk
    a = brute.rand(m, k, dtype=brute.bit1, device=device)
    b = brute.rand(n, k, dtype=brute.bit1, device=device)
    def run():
        r = a @ b   # dispatches to xnor_popcount_matmul via __matmul__
        synced()
        return r
    benchmark(run)


# ── float32 baseline ──────────────────────────────────────────────────────────

@SHAPES
def test_matmul_float32(benchmark, device, synced, mnk):
    m, k, n = mnk
    a = torch.rand(m, k, device=device)
    b = torch.rand(n, k, device=device)
    def run():
        r = a @ b.t()
        synced()
        return r
    benchmark(run)


# ── ±1 float matmul (semantic equivalent) ─────────────────────────────────────

@SHAPES
def test_matmul_pm1_float32(benchmark, device, synced, mnk):
    """Semantically equivalent to xnor_popcount_matmul using standard float32 ops."""
    m, k, n = mnk
    raw_a = torch.randint(0, 2, (m, k), dtype=torch.bool, device=device)
    raw_b = torch.randint(0, 2, (n, k), dtype=torch.bool, device=device)
    a = (raw_a.float() * 2 - 1)   # {False,True} → {-1.0, +1.0}
    b = (raw_b.float() * 2 - 1)
    def run():
        r = a @ b.t()
        synced()
        return r
    benchmark(run)


# ── int8 quantised matmul (alternative baseline where available) ───────────────

@SHAPES
def test_matmul_int8(benchmark, device, synced, mnk):
    """int8 matmul via torch._int_mm (CPU/CUDA only; skipped otherwise)."""
    if device.type == "mps":
        pytest.skip("int8 matmul not supported on MPS")
    m, k, n = mnk
    # _int_mm requires contiguous row-major inputs
    a = torch.randint(-1, 2, (m, k), dtype=torch.int8, device=device)
    b = torch.randint(-1, 2, (n, k), dtype=torch.int8, device=device).t().contiguous()
    try:
        torch._int_mm(a, b)  # smoke-test availability
    except (RuntimeError, AttributeError):
        pytest.skip("torch._int_mm not available on this build")
    def run():
        r = torch._int_mm(a, b)
        synced()
        return r
    benchmark(run)
