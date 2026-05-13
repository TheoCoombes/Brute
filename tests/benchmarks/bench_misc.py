"""bit1-specific method benchmarks — randomize_, hamming, word_popcount,
unpack_pm1. These have no bool equivalent; we still benchmark them so any
regression in the fused-packed kernels is visible.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import VECTOR_SCALES, with_sync, set_throughput


def _rand_bool(n, device):
    return torch.randint(0, 2, (n,), dtype=torch.bool, device=device)


# randomize_() — in-place packed random fill 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_randomize_(benchmark, device, n_bits):
    benchmark.group = "misc/randomize_"
    x = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.randomize_(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_random_(benchmark, device, n_bits):
    """Reference: bool .random_() fills bytes with 0/1."""
    benchmark.group = "misc/randomize_"
    x = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.random_(0, 2), device))


# hamming() — fused XOR+popcount, total distance 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_hamming(benchmark, device, n_bits):
    benchmark.group = "misc/hamming"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a.hamming(b), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_hamming_ref(benchmark, device, n_bits):
    """Reference: (a ^ b).long().sum() — what a bool user would write."""
    benchmark.group = "misc/hamming"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: (a ^ b).long().sum(), device))


# word_popcount() — per-packed-word popcount 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_word_popcount(benchmark, device, n_bits):
    benchmark.group = "misc/word_popcount"
    a = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a.word_popcount(), device))


# unpack_pm1() — decode to +1/-1 float32 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_unpack_pm1(benchmark, device, n_bits):
    benchmark.group = "misc/unpack_pm1"
    a = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a.unpack_pm1(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_to_float_pm1_ref(benchmark, device, n_bits):
    """Reference: (b.float() * 2 - 1) — what a bool user would write."""
    benchmark.group = "misc/unpack_pm1"
    a = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a.float() * 2 - 1, device))
