"""Elementwise bitwise / logical benchmarks — measures __torch_function__
dispatch overhead vs raw torch on bool, across devices and scales.

These ops currently take the bool-fallback path inside __torch_function__, so
the bit1 numbers include: unpack-to-bool semantics (the bool base is already
materialized), the op itself on bool storage, and the lazy repack of the result.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import VECTOR_SCALES, with_sync, set_throughput


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_logical_not(benchmark, device, n_bits):
    benchmark.group = "dispatch/logical_not"
    bit = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_not(bit), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_logical_not(benchmark, device, n_bits):
    benchmark.group = "dispatch/logical_not"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.logical_not(src), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_and(benchmark, device, n_bits):
    benchmark.group = "dispatch/and"
    a = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    b = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a & b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_and(benchmark, device, n_bits):
    benchmark.group = "dispatch/and"
    a = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    b = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a & b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_xor(benchmark, device, n_bits):
    benchmark.group = "dispatch/xor"
    a = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    b = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a ^ b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_xor(benchmark, device, n_bits):
    benchmark.group = "dispatch/xor"
    a = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    b = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a ^ b, device))
