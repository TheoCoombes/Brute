"""Comparison-op benchmarks — bit1 vs bool.

`==`, `!=`, `torch.eq`, `torch.ne`, `torch.equal`. Full-tensor `torch.equal` for
bit1 short-circuits via `bit1_hamming_total == 0`; the elementwise paths
fall through to the bool kernel via __torch_function__.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import VECTOR_SCALES, with_sync, set_throughput


def _rand_bool(n, device):
    return torch.randint(0, 2, (n,), dtype=torch.bool, device=device)


# == (EQ)

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_eq(benchmark, device, n_bits):
    benchmark.group = "compare/eq"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a == b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_eq(benchmark, device, n_bits):
    benchmark.group = "compare/eq"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a == b, device))


# != (NE) 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_ne(benchmark, device, n_bits):
    benchmark.group = "compare/ne"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a != b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_ne(benchmark, device, n_bits):
    benchmark.group = "compare/ne"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a != b, device))


# torch.equal (full-tensor) 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_equal(benchmark, device, n_bits):
    benchmark.group = "compare/equal"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.equal(a, b), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_equal(benchmark, device, n_bits):
    benchmark.group = "compare/equal"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.equal(a, b), device))
