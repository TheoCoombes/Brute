"""Dim-keep reductions — sum / all / any / count_nonzero along a dimension.

The existing `bench_reductions.py` covers full reductions (where bit1 wins via
packed_popcount). Dim-reductions currently fall through to the bool path, so
bit1 should be at parity with bool here. These benchmarks make any regression
visible.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import MATRIX_SCALES, with_sync, set_throughput


def _rand_bool_2d(shape, device):
    return torch.randint(0, 2, shape, dtype=torch.bool, device=device)


# sum(dim=0) / sum(dim=1) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_sum_dim0(benchmark, device, shape):
    benchmark.group = "reduce_dim/sum_dim0"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.sum(x, dim=0), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_sum_dim0(benchmark, device, shape):
    benchmark.group = "reduce_dim/sum_dim0"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.sum(x, dim=0), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_sum_dim1(benchmark, device, shape):
    benchmark.group = "reduce_dim/sum_dim1"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.sum(x, dim=1), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_sum_dim1(benchmark, device, shape):
    benchmark.group = "reduce_dim/sum_dim1"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.sum(x, dim=1), device))


# all(dim=0) / any(dim=-1) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_all_dim0(benchmark, device, shape):
    benchmark.group = "reduce_dim/all_dim0"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.all(x, dim=0), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_all_dim0(benchmark, device, shape):
    benchmark.group = "reduce_dim/all_dim0"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.all(x, dim=0), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_any_dim_last(benchmark, device, shape):
    benchmark.group = "reduce_dim/any_dim_last"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.any(x, dim=-1), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_any_dim_last(benchmark, device, shape):
    benchmark.group = "reduce_dim/any_dim_last"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.any(x, dim=-1), device))


# count_nonzero(dim=0) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_count_nonzero_dim0(benchmark, device, shape):
    benchmark.group = "reduce_dim/count_nonzero_dim0"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.count_nonzero(x, dim=0), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_count_nonzero_dim0(benchmark, device, shape):
    benchmark.group = "reduce_dim/count_nonzero_dim0"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.count_nonzero(x, dim=0), device))
