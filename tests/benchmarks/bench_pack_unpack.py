"""Benchmarks for bit packing / unpacking."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


@pytest.mark.parametrize("shape", [(1024,), (4096,), (1024, 1024)])
def test_bench_pack(benchmark, shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    benchmark(lambda: bit1(src))


@pytest.mark.parametrize("shape", [(1024,), (4096,), (1024, 1024)])
def test_bench_unpack(benchmark, shape):
    bit = bit1(torch.randint(0, 2, shape, dtype=torch.bool))
    benchmark(lambda: bit.bool())


@pytest.mark.parametrize("shape", [(1024,), (1024, 1024)])
def test_bench_unpack_pm1(benchmark, shape):
    bit = bit1(torch.randint(0, 2, shape, dtype=torch.bool))
    benchmark(lambda: bit.unpack_pm1())


@pytest.mark.parametrize("pack_dtype", [torch.uint8, torch.uint32, torch.uint64])
def test_bench_pack_by_pack_dtype(benchmark, pack_dtype):
    src = torch.randint(0, 2, (1024, 1024), dtype=torch.bool)
    benchmark(lambda: bit1(src, pack_dtype=pack_dtype))
