"""XNOR-popcount matmul benchmarks across devices and (M, K, N) scales.

On CUDA-sm_80+ this exercises the CUTLASS B1 XOR-popc tensor-core path; on
older arches or unaligned K it falls back to the hand-tuned kernel. Compare
against the equivalent dense float32 matmul.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import MATMUL_SCALES, with_sync, set_throughput


PACK_WIDTH_PARAMS = [
    pytest.param(torch.uint8,  8,  id="pw_08"),
    pytest.param(torch.uint32, 32, id="pw_32"),
    pytest.param(torch.uint64, 64, id="pw_64"),
]


@pytest.mark.parametrize("MKN", MATMUL_SCALES)
def test_xnor_matmul(benchmark, device, MKN):
    benchmark.group = "matmul/xnor-popcount"
    M, K, N = MKN
    a = bit1(torch.randint(0, 2, (M, K), dtype=torch.bool, device=device),
             pack_dtype=torch.uint64)
    b = bit1(torch.randint(0, 2, (N, K), dtype=torch.bool, device=device),
             pack_dtype=torch.uint64)
    # 2*M*N*K bit-ops; report total bit-ops/s so devices are comparable.
    set_throughput(benchmark, 2 * M * N * K, "bit-ops")
    benchmark(with_sync(lambda: torch.ops.brute.xnor_popcount_matmul(
        a._packed_buf, b._packed_buf, K, 64), device))


@pytest.mark.parametrize("MKN", MATMUL_SCALES)
def test_float_matmul(benchmark, device, MKN):
    """Reference dense float32 matmul — the thing bit1 is trying to beat."""
    benchmark.group = "matmul/float"
    M, K, N = MKN
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    set_throughput(benchmark, 2 * M * N * K, "flops")
    benchmark(with_sync(lambda: a @ b, device))


@pytest.mark.parametrize("pack_dtype, pw", PACK_WIDTH_PARAMS)
def test_xnor_matmul_pack_width(benchmark, device, pack_dtype, pw):
    """Pack-width sweep at a fixed medium shape — surfaces word-stride effects."""
    benchmark.group = "matmul/pack-width sweep"
    M, K, N = 512, 1024, 512
    a = bit1(torch.randint(0, 2, (M, K), dtype=torch.bool, device=device),
             pack_dtype=pack_dtype)
    b = bit1(torch.randint(0, 2, (N, K), dtype=torch.bool, device=device),
             pack_dtype=pack_dtype)
    set_throughput(benchmark, 2 * M * N * K, "bit-ops")
    benchmark(with_sync(lambda: torch.ops.brute.xnor_popcount_matmul(
        a._packed_buf, b._packed_buf, K, pw), device))
