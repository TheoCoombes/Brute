"""Popcount benchmarks across devices and scales.

Compares:
  * brute bit1 popcount (packed_popcount under the hood)
  * torch.bool .long().sum() (the naive equivalent)
  * per-element popcount kernel

Group naming: `popcount/<flavour>` so pytest-benchmark groups apples-with-apples
in the report.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import VECTOR_SCALES, with_sync, set_throughput


# ──────────────────────────────────────────────────────────────────────────────
# Full-tensor popcount: bit1 (packed) vs bool (.long().sum()) — the canonical
# "is bit packing actually faster" comparison.
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_popcount(benchmark, device, n_bits):
    benchmark.group = "popcount/full"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    bit = bit1(src)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: bit.popcount(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_sum(benchmark, device, n_bits):
    benchmark.group = "popcount/full"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: src.long().sum(), device))


# ──────────────────────────────────────────────────────────────────────────────
# Per-element popcount on the packed buffer (one popcount per uint64 word).
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_per_element_popcount(benchmark, device, n_bits):
    benchmark.group = "popcount/per-element"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    bit = bit1(src)
    pb = bit._packed_buf
    set_throughput(benchmark, pb.numel(), "words")
    benchmark(with_sync(lambda: torch.ops.brute.popcount(pb), device))


# ──────────────────────────────────────────────────────────────────────────────
# Fast-path reductions wired through __torch_function__.
# These should match the bit1.popcount() throughput exactly — they share the
# same underlying packed_popcount kernel.
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_torch_sum(benchmark, device, n_bits):
    benchmark.group = "popcount/torch.sum"
    bit = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.sum(bit), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_torch_sum(benchmark, device, n_bits):
    benchmark.group = "popcount/torch.sum"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.sum(src), device))
