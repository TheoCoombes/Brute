"""XNOR-popcount matmul benchmarks vs torch float matmul."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


SHAPES = [(64, 64, 64), (128, 256, 128), (256, 512, 256)]


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_bench_xnor_matmul(benchmark, M, K, N):
    a = bit1(torch.randint(0, 2, (M, K), dtype=torch.bool))
    b = bit1(torch.randint(0, 2, (N, K), dtype=torch.bool))

    def run():
        return torch.ops.brute.xnor_popcount_matmul(
            a._packed_buf, b._packed_buf, K, 8
        )

    benchmark(run)


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_bench_torch_float_matmul(benchmark, M, K, N):
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    benchmark(lambda: a @ b)


@pytest.mark.parametrize("pack_dtype", [torch.uint8, torch.uint32, torch.uint64])
def test_bench_xnor_matmul_pack_width(benchmark, pack_dtype):
    M, K, N = 128, 512, 128
    a = bit1(torch.randint(0, 2, (M, K), dtype=torch.bool), pack_dtype=pack_dtype)
    b = bit1(torch.randint(0, 2, (N, K), dtype=torch.bool), pack_dtype=pack_dtype)
    pw = {torch.uint8: 8, torch.uint32: 32, torch.uint64: 64}[pack_dtype]
    benchmark(lambda: torch.ops.brute.xnor_popcount_matmul(
        a._packed_buf, b._packed_buf, K, pw))
