"""Element-wise bitwise benchmarks — bit1 vs bool across all bitwise ops.

Covers both out-of-place (`&`, `|`, `^`, `~`) and in-place (`&=`, `|=`, `^=`)
variants. Speedup > 1.0× means bit1 beats bool; < 1.0× means bool beats bit1.

The bit1 hot path runs entirely on the packed integer buffer (1 bit per element
of DRAM traffic), so on a memory-bound workload we expect 8× / 32× / 64× over
bool for uint8 / uint32 / uint64 packing respectively, less any per-op overhead.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import VECTOR_SCALES, with_sync, set_throughput


def _rand_bool(n, device):
    return torch.randint(0, 2, (n,), dtype=torch.bool, device=device)


# & (AND)

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_and(benchmark, device, n_bits):
    benchmark.group = "bitwise/and"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a & b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_and(benchmark, device, n_bits):
    benchmark.group = "bitwise/and"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a & b, device))


# | (OR)

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_or(benchmark, device, n_bits):
    benchmark.group = "bitwise/or"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a | b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_or(benchmark, device, n_bits):
    benchmark.group = "bitwise/or"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a | b, device))


# ^ (XOR) 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_xor(benchmark, device, n_bits):
    benchmark.group = "bitwise/xor"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a ^ b, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_xor(benchmark, device, n_bits):
    benchmark.group = "bitwise/xor"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a ^ b, device))


# ~ (NOT/invert)

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_invert(benchmark, device, n_bits):
    benchmark.group = "bitwise/invert"
    a = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: ~a, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_invert(benchmark, device, n_bits):
    benchmark.group = "bitwise/invert"
    a = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: ~a, device))


# &= (IAND)

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_iand(benchmark, device, n_bits):
    benchmark.group = "bitwise/iand"
    b = bit1(_rand_bool(n_bits, device))
    a = bit1(_rand_bool(n_bits, device))

    def _run():
        x = a.clone()
        x &= b
        return x

    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(_run, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_iand(benchmark, device, n_bits):
    benchmark.group = "bitwise/iand"
    b = _rand_bool(n_bits, device)
    a = _rand_bool(n_bits, device)

    def _run():
        x = a.clone()
        x &= b
        return x

    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(_run, device))


# |= (IOR)

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_ior(benchmark, device, n_bits):
    benchmark.group = "bitwise/ior"
    b = bit1(_rand_bool(n_bits, device))
    a = bit1(_rand_bool(n_bits, device))

    def _run():
        x = a.clone()
        x |= b
        return x

    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(_run, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_ior(benchmark, device, n_bits):
    benchmark.group = "bitwise/ior"
    b = _rand_bool(n_bits, device)
    a = _rand_bool(n_bits, device)

    def _run():
        x = a.clone()
        x |= b
        return x

    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(_run, device))


# ^= (IXOR)

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_ixor(benchmark, device, n_bits):
    benchmark.group = "bitwise/ixor"
    b = bit1(_rand_bool(n_bits, device))
    a = bit1(_rand_bool(n_bits, device))

    def _run():
        x = a.clone()
        x ^= b
        return x

    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(_run, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_ixor(benchmark, device, n_bits):
    benchmark.group = "bitwise/ixor"
    b = _rand_bool(n_bits, device)
    a = _rand_bool(n_bits, device)

    def _run():
        x = a.clone()
        x ^= b
        return x

    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(_run, device))


# chain: (a & b) ^ c 
# Stress test for the lazy-bool design: the intermediate (a & b) should never
# materialise its bool view because c is also bit1 and the consumer is bitwise.

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_chain_and_xor(benchmark, device, n_bits):
    benchmark.group = "bitwise/chain_and_xor"
    a = bit1(_rand_bool(n_bits, device))
    b = bit1(_rand_bool(n_bits, device))
    c = bit1(_rand_bool(n_bits, device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: (a & b) ^ c, device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_chain_and_xor(benchmark, device, n_bits):
    benchmark.group = "bitwise/chain_and_xor"
    a = _rand_bool(n_bits, device)
    b = _rand_bool(n_bits, device)
    c = _rand_bool(n_bits, device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: (a & b) ^ c, device))
