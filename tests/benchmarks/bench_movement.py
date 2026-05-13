"""Shape & memory movement benchmarks — t / transpose / reshape / view /
contiguous / clone / to(device).

These exercise the "lifecycle" cost of bit1 tensors — every transformation
either preserves the packed cache (cheap) or invalidates it (forces a re-pack
on next read). The targets here are speedups close to bool: bit1 should be at
least *competitive* with bool on shape ops.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import MATRIX_SCALES, with_sync, set_throughput


def _rand_bool_2d(shape, device):
    return torch.randint(0, 2, shape, dtype=torch.bool, device=device)


# .t() / .transpose(0, 1) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_t(benchmark, device, shape):
    benchmark.group = "movement/t"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.t(), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_t(benchmark, device, shape):
    benchmark.group = "movement/t"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.t(), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_transpose(benchmark, device, shape):
    benchmark.group = "movement/transpose"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.transpose(0, 1), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_transpose(benchmark, device, shape):
    benchmark.group = "movement/transpose"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.transpose(0, 1), device))


# .clone() 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_clone(benchmark, device, shape):
    benchmark.group = "movement/clone"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.clone(), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_clone(benchmark, device, shape):
    benchmark.group = "movement/clone"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.clone(), device))


# .contiguous() (no-op on contiguous input — measures dispatch cost) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_contiguous(benchmark, device, shape):
    benchmark.group = "movement/contiguous"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.contiguous(), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_contiguous(benchmark, device, shape):
    benchmark.group = "movement/contiguous"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x.contiguous(), device))


# .reshape((N*M,)) — view-able when contiguous 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_reshape_flat(benchmark, device, shape):
    benchmark.group = "movement/reshape_flat"
    x = bit1(_rand_bool_2d(shape, device))
    n = shape[0] * shape[1]
    set_throughput(benchmark, n, "bits")
    benchmark(with_sync(lambda: x.reshape((n,)), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_reshape_flat(benchmark, device, shape):
    benchmark.group = "movement/reshape_flat"
    x = _rand_bool_2d(shape, device)
    n = shape[0] * shape[1]
    set_throughput(benchmark, n, "bits")
    benchmark(with_sync(lambda: x.reshape((n,)), device))


# .view((N*M,)) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_view_flat(benchmark, device, shape):
    benchmark.group = "movement/view_flat"
    x = bit1(_rand_bool_2d(shape, device))
    n = shape[0] * shape[1]
    set_throughput(benchmark, n, "bits")
    benchmark(with_sync(lambda: x.view((n,)), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_view_flat(benchmark, device, shape):
    benchmark.group = "movement/view_flat"
    x = _rand_bool_2d(shape, device)
    n = shape[0] * shape[1]
    set_throughput(benchmark, n, "bits")
    benchmark(with_sync(lambda: x.view((n,)), device))
