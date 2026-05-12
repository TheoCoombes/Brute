"""Popcount benchmarks: bit1.popcount() vs torch.sum on bool."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


SIZES = [(1024,), (1 << 16,), (1 << 18,)]


@pytest.mark.parametrize("shape", SIZES)
def test_bench_bit1_popcount(benchmark, shape):
    bit = bit1(torch.randint(0, 2, shape, dtype=torch.bool))
    benchmark(lambda: bit.popcount())


@pytest.mark.parametrize("shape", SIZES)
def test_bench_torch_bool_sum(benchmark, shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    benchmark(lambda: src.long().sum())


@pytest.mark.parametrize("shape", SIZES)
def test_bench_per_element_popcount(benchmark, shape):
    bit = bit1(torch.randint(0, 2, shape, dtype=torch.bool))
    benchmark(lambda: torch.ops.brute.popcount(bit._packed_buf))
