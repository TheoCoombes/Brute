"""Dispatch overhead benchmarks: __torch_function__ cost vs plain torch."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


@pytest.mark.parametrize("shape", [(64,), (1024,), (1024, 1024)])
def test_bench_logical_not_bit1(benchmark, shape):
    bit = bit1(torch.randint(0, 2, shape, dtype=torch.bool))
    benchmark(lambda: torch.logical_not(bit))


@pytest.mark.parametrize("shape", [(64,), (1024,), (1024, 1024)])
def test_bench_logical_not_torch_bool(benchmark, shape):
    src = torch.randint(0, 2, shape, dtype=torch.bool)
    benchmark(lambda: torch.logical_not(src))


@pytest.mark.parametrize("shape", [(64,), (1024,), (1024, 1024)])
def test_bench_and_bit1(benchmark, shape):
    a = bit1(torch.randint(0, 2, shape, dtype=torch.bool))
    b = bit1(torch.randint(0, 2, shape, dtype=torch.bool))
    benchmark(lambda: a & b)


@pytest.mark.parametrize("shape", [(64,), (1024,), (1024, 1024)])
def test_bench_and_torch_bool(benchmark, shape):
    a = torch.randint(0, 2, shape, dtype=torch.bool)
    b = torch.randint(0, 2, shape, dtype=torch.bool)
    benchmark(lambda: a & b)
