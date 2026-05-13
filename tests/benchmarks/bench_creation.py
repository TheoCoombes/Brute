"""Tensor-creation benchmarks — bit1 vs bool.

`brute.zeros / ones / empty` should pay only the packed-buffer allocation cost
plus the (small) bool backing alloc. Compared to bool, bit1 should be
competitive at large sizes since the packed buffer is 8/32/64× smaller. At
tiny sizes the constant-cost overhead can dominate.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.benchmarks._helpers import VECTOR_SCALES, with_sync, set_throughput


# brute.zeros / torch.zeros (bool) 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_zeros(benchmark, device, n_bits):
    benchmark.group = "creation/zeros"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: brute.zeros(n_bits, dtype=brute.bit1, device=device),
        device,
    ))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_zeros(benchmark, device, n_bits):
    benchmark.group = "creation/zeros"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: torch.zeros(n_bits, dtype=torch.bool, device=device),
        device,
    ))


# ones 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_ones(benchmark, device, n_bits):
    benchmark.group = "creation/ones"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: brute.ones(n_bits, dtype=brute.bit1, device=device),
        device,
    ))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_ones(benchmark, device, n_bits):
    benchmark.group = "creation/ones"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: torch.ones(n_bits, dtype=torch.bool, device=device),
        device,
    ))


# empty 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_empty(benchmark, device, n_bits):
    benchmark.group = "creation/empty"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: brute.empty(n_bits, dtype=brute.bit1, device=device),
        device,
    ))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_empty(benchmark, device, n_bits):
    benchmark.group = "creation/empty"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: torch.empty(n_bits, dtype=torch.bool, device=device),
        device,
    ))


# fill_ on existing tensor 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_fill_true(benchmark, device, n_bits):
    benchmark.group = "creation/fill_true"
    x = brute.zeros(n_bits, dtype=brute.bit1, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.fill_(True), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_fill_true(benchmark, device, n_bits):
    benchmark.group = "creation/fill_true"
    x = torch.zeros(n_bits, dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.fill_(True), device))


# zero_() 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_zero_(benchmark, device, n_bits):
    benchmark.group = "creation/zero_"
    x = brute.ones(n_bits, dtype=brute.bit1, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.zero_(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_zero_(benchmark, device, n_bits):
    benchmark.group = "creation/zero_"
    x = torch.ones(n_bits, dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.zero_(), device))
