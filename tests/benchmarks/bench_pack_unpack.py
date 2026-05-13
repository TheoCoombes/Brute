"""Pack / unpack benchmarks across devices, scales, and pack widths.

Covers:
  * `pack_bool` / `pack_bits(float)` — the bool→packed and float→packed paths.
  * `unpack_pm1`  — packed→±1 float (used by mixed-precision flows).
  * `unpack_bool` — packed→bool   (used by the bit1↔bool round-trip).
  * pack-width sweep on the largest scale (cache effects of pw=8/32/64).
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1
from tests.benchmarks._helpers import (
    VECTOR_SCALES, MATRIX_SCALES, with_sync, set_throughput,
)


# 1-D pack/unpack across scales 

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_pack_bool(benchmark, device, n_bits):
    benchmark.group = "pack/bool→packed"
    src = torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: bit1(src), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_pack_float(benchmark, device, n_bits):
    """Float→packed via pack_bits — slower than pack_bool (one float compare per bit)."""
    benchmark.group = "pack/float→packed"
    src = torch.randn(n_bits, device=device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: torch.ops.brute.pack_bits(src, 64), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_unpack_pm1(benchmark, device, n_bits):
    benchmark.group = "unpack/packed→±1 float"
    bit = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: bit.unpack_pm1(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_unpack_bool(benchmark, device, n_bits):
    benchmark.group = "unpack/packed→bool"
    bit = bit1(torch.randint(0, 2, (n_bits,), dtype=torch.bool, device=device))
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: bit.bool(), device))


# Pack-width sweep at the largest 2-D shape 
# pw doesn't change the bit content, only the leading-dim alignment. Surfaces
# any cache / strided-access cliffs in the kernel.

@pytest.mark.parametrize("pack_dtype", [
    pytest.param(torch.uint8,  id="pw_08"),
    pytest.param(torch.uint32, id="pw_32"),
    pytest.param(torch.uint64, id="pw_64"),
])
def test_pack_pack_width(benchmark, device, pack_dtype):
    benchmark.group = "pack/pack-width sweep"
    src = torch.randint(0, 2, (1024, 1024), dtype=torch.bool, device=device)
    set_throughput(benchmark, src.numel(), "bits")
    benchmark(with_sync(lambda: bit1(src, pack_dtype=pack_dtype), device))


# 2-D matrix pack/unpack — row-major batching exercise 

@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_pack_2d(benchmark, device, shape):
    benchmark.group = "pack/2D"
    src = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    set_throughput(benchmark, src.numel(), "bits")
    benchmark(with_sync(lambda: bit1(src), device))


@pytest.mark.parametrize("shape", MATRIX_SCALES)
def test_unpack_bool_2d(benchmark, device, shape):
    benchmark.group = "unpack/2D"
    bit = bit1(torch.randint(0, 2, shape, dtype=torch.bool, device=device))
    set_throughput(benchmark, bit.numel(), "bits")
    benchmark(with_sync(lambda: bit.bool(), device))
