"""Reduction fast-path benchmarks: torch.all/any/equal/sum on bit1 vs bool.

All four are routed through `packed_popcount` / `bit1_hamming_total` in the
bit1 case, so they should all collapse to the same underlying kernel cost — the
test is whether the dispatch overhead and the no-bool-materialization path
beats the equivalent bool reduction.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import (
    VECTOR_SCALES, with_sync, set_throughput, skip_if_mps_composite,
)


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_torch_all(benchmark, device, n_bits):
    benchmark.group = "reduce/torch.all"
    src = torch.ones(n_bits, dtype=torch.bool, device=device)
    bit = bit1(src)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.all(bit), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_torch_all(benchmark, device, n_bits):
    benchmark.group = "reduce/torch.all"
    src = torch.ones(n_bits, dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.all(src), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_torch_any(benchmark, device, n_bits):
    benchmark.group = "reduce/torch.any"
    src = torch.zeros(n_bits, dtype=torch.bool, device=device)
    src[-1] = True   # one set bit, worst case for early-exit but representative
    bit = bit1(src)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.any(bit), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_torch_any(benchmark, device, n_bits):
    benchmark.group = "reduce/torch.any"
    src = torch.zeros(n_bits, dtype=torch.bool, device=device)
    src[-1] = True
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.any(src), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_torch_equal(benchmark, device, n_bits):
    """Fused XOR+popcount via bit1_hamming_total."""
    skip_if_mps_composite(device)
    benchmark.group = "reduce/torch.equal"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    a = bit1(src)
    b = bit1(src.clone())
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.equal(a, b), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_torch_equal(benchmark, device, n_bits):
    benchmark.group = "reduce/torch.equal"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    a = src
    b = src.clone()
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.equal(a, b), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_hamming_total(benchmark, device, n_bits):
    """Direct op call — bit1_hamming_total over the packed buffer."""
    skip_if_mps_composite(device)
    benchmark.group = "reduce/hamming-total"
    a = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    b = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.ops.brute.bit1_hamming_total(
        a._packed_buf, b._packed_buf), device))
