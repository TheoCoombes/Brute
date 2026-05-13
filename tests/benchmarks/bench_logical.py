"""torch.logical_* benchmarks — bit1 vs bool.

torch.logical_and/or/xor/not are semantically the same as the bitwise ops on
bool dtype. They route through __torch_function__ rather than the Python
__and__/__or__/__xor__/__invert__ dunders, so they exercise a slightly
different dispatch path.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import VECTOR_SCALES, with_sync, set_throughput


def _rand_bool(n, device):
    return torch.randint(0, 2, (n,), dtype=torch.bool, device=device)


# torch.logical_and 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_logical_and(benchmark, device, n_bits):
    benchmark.group = "logical/and"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_and(a, b), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_logical_and(benchmark, device, n_bits):
    benchmark.group = "logical/and"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_and(a, b), device))


# torch.logical_or 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_logical_or(benchmark, device, n_bits):
    benchmark.group = "logical/or"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_or(a, b), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_logical_or(benchmark, device, n_bits):
    benchmark.group = "logical/or"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_or(a, b), device))


# torch.logical_xor 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_logical_xor(benchmark, device, n_bits):
    benchmark.group = "logical/xor"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_xor(a, b), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_logical_xor(benchmark, device, n_bits):
    benchmark.group = "logical/xor"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_xor(a, b), device))


# torch.logical_not 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_logical_not(benchmark, device, n_bits):
    benchmark.group = "logical/not"
    a = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_not(a), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_logical_not(benchmark, device, n_bits):
    benchmark.group = "logical/not"
    a = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_not(a), device))


# torch.bitwise_not (named functional form) 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_bitwise_not_func(benchmark, device, n_bits):
    benchmark.group = "logical/bitwise_not_func"
    a = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.bitwise_not(a), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_bitwise_not_func(benchmark, device, n_bits):
    benchmark.group = "logical/bitwise_not_func"
    a = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.bitwise_not(a), device))
