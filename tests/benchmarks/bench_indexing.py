"""Indexing benchmarks — bit1 vs bool for __getitem__ / __setitem__.

Indexing is the area where bit1 is *expected* to be slower than bool today:
slicing on the bool view is followed by a re-pack to bit1 in
`__torch_function__`. Phase 8 of the optimisation roadmap introduces packed
fast paths for the easy patterns (axis-0 row slice, scalar row setitem).

These benchmarks expose the gap so progress can be measured.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import MATRIX_SCALES, with_sync, set_throughput


def _rand_bool_2d(shape, device):
    return torch.randint(0, 2, shape, dtype=torch.bool, device=device)


# x[0] — leading-axis integer index 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_index_row(benchmark, device, shape):
    benchmark.group = "indexing/row"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[0], device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_index_row(benchmark, device, shape):
    benchmark.group = "indexing/row"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[0], device))


# x[:N//2] — leading-axis slice 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_slice_row(benchmark, device, shape):
    benchmark.group = "indexing/slice_row"
    x = bit1(_rand_bool_2d(shape, device))
    half = shape[0] // 2
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[:half], device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_slice_row(benchmark, device, shape):
    benchmark.group = "indexing/slice_row"
    x = _rand_bool_2d(shape, device)
    half = shape[0] // 2
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[:half], device))


# x[:, 0] — last-axis column extract (worst case) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_index_col(benchmark, device, shape):
    benchmark.group = "indexing/col"
    x = bit1(_rand_bool_2d(shape, device))
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[:, 0], device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_index_col(benchmark, device, shape):
    benchmark.group = "indexing/col"
    x = _rand_bool_2d(shape, device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[:, 0], device))


# x[mask] — bool-mask index 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_index_mask(benchmark, device, shape):
    benchmark.group = "indexing/mask"
    x = bit1(_rand_bool_2d(shape, device))
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[mask], device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_index_mask(benchmark, device, shape):
    benchmark.group = "indexing/mask"
    x = _rand_bool_2d(shape, device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[mask], device))


# x[idx] — integer-tensor row gather 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_gather_rows(benchmark, device, shape):
    benchmark.group = "indexing/gather_rows"
    x = bit1(_rand_bool_2d(shape, device))
    idx = torch.randperm(shape[0], device=device)[: shape[0] // 2]
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[idx], device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_gather_rows(benchmark, device, shape):
    benchmark.group = "indexing/gather_rows"
    x = _rand_bool_2d(shape, device)
    idx = torch.randperm(shape[0], device=device)[: shape[0] // 2]
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: x[idx], device))


# x[:, 0] = 1 — strided scalar setitem (worst case) 
# Note: the bit1 path bumps the storage version, so the next `_packed_buf`
# access re-packs the whole tensor. We don't time that re-pack here (only the
# setitem itself), but the benchmark caller is welcome to verify by reading
# `_packed_buf` after.

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_setitem_col_scalar(benchmark, device, shape):
    benchmark.group = "indexing/setitem_col_scalar"
    base = bit1(_rand_bool_2d(shape, device))

    def _run():
        x = base.clone()
        x[:, 0] = 1
        return x

    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(_run, device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_setitem_col_scalar(benchmark, device, shape):
    benchmark.group = "indexing/setitem_col_scalar"
    base = _rand_bool_2d(shape, device)

    def _run():
        x = base.clone()
        x[:, 0] = 1
        return x

    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(_run, device))


# x[0] = 1 — row scalar setitem (axis-0, fast-path candidate) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_setitem_row_scalar(benchmark, device, shape):
    benchmark.group = "indexing/setitem_row_scalar"
    base = bit1(_rand_bool_2d(shape, device))

    def _run():
        x = base.clone()
        x[0] = 1
        return x

    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(_run, device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_setitem_row_scalar(benchmark, device, shape):
    benchmark.group = "indexing/setitem_row_scalar"
    base = _rand_bool_2d(shape, device)

    def _run():
        x = base.clone()
        x[0] = 1
        return x

    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(_run, device))


# torch.index_select(x, 0, idx) 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bit1_index_select(benchmark, device, shape):
    benchmark.group = "indexing/index_select"
    x = bit1(_rand_bool_2d(shape, device))
    idx = torch.randperm(shape[0], device=device)[: shape[0] // 2]
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.index_select(x, 0, idx), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_bool_index_select(benchmark, device, shape):
    benchmark.group = "indexing/index_select"
    x = _rand_bool_2d(shape, device)
    idx = torch.randperm(shape[0], device=device)[: shape[0] // 2]
    set_throughput(benchmark, shape[0] * shape[1], "bits")
    benchmark(with_sync(lambda: torch.index_select(x, 0, idx), device))
