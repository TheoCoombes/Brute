"""Direct-kernel benchmarks for ops that don't fit the catalog model.

The catalog drives parity-paired bench_bit1 / bench_bool entries. The ops
here are different:

  • `pack_bool`, `unpack_bits` — these are the bit1 ↔ bool round-trip
    primitives. The "bool equivalent" of pack_bool is the identity (already
    bool); we compare against the cost of *materializing* an equivalent
    bool tensor, which is the bool side of `creation/randint`.
  • `xnor_popcount_matmul` — the raw kernel. The catalog times the Python-
    side `a @ b` path; here we time the bare op to surface dispatch overhead.
  • `bit1_hamming_total` — same: catalog times `a.hamming(b)`; here we time
    the bare kernel.

Group names are unique (so make_report doesn't try to pair them with a
non-existent bool variant), and surface in the "unpaired" panel of the report.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.benchmarks._helpers import (
    VECTOR_SCALES, MATMUL_SCALES, with_sync, set_throughput,
)


def _rand_bool(shape, device, *, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cpu = torch.randint(0, 2, shape, generator=g, dtype=torch.uint8).bool()
    return cpu.to(device=device)


# pack_bool: cost of building a bit1 from a bool tensor.

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_pack_bool(benchmark, device, n_bits):
    benchmark.group = "kernels/pack_bool"
    src = _rand_bool((n_bits,), device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.ops.brute.pack_bool(src, 64), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_clone(benchmark, device, n_bits):
    """Reference for pack_bool: the cost of a plain bool .clone()."""
    benchmark.group = "kernels/pack_bool"
    src = _rand_bool((n_bits,), device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: src.clone(), device))


# unpack_bits: cost of materializing a bool view from a packed buffer.

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_unpack_bool(benchmark, device, n_bits):
    benchmark.group = "kernels/unpack_bool"
    bit = brute.tensor(_rand_bool((n_bits,), device), dtype=brute.bit1)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: bit.bool(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_unpack_noop(benchmark, device, n_bits):
    """Reference for unpack_bool: bool→bool .clone() (identity cost)."""
    benchmark.group = "kernels/unpack_bool"
    src = _rand_bool((n_bits,), device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: src.bool().clone(), device))


# xnor_popcount_matmul: direct kernel vs torch.float matmul reference.

@pytest.mark.parametrize("MKN", MATMUL_SCALES)
def test_bit1_xnor_matmul_kernel(benchmark, device, MKN):
    benchmark.group = "kernels/xnor_matmul"
    M, K, N = MKN
    a = brute.tensor(_rand_bool((M, K), device), dtype=brute.bit1, pack_dtype=torch.uint64)
    b = brute.tensor(_rand_bool((N, K), device), dtype=brute.bit1, pack_dtype=torch.uint64)
    set_throughput(benchmark, 2 * M * N * K, "bit-ops")
    benchmark(with_sync(lambda: torch.ops.brute.xnor_popcount_matmul(
        a._packed_buf, b._packed_buf, K, 64), device))


@pytest.mark.parametrize("MKN", MATMUL_SCALES)
def test_bool_float_matmul(benchmark, device, MKN):
    """Reference: dense float32 matmul — what bit1 is competing against."""
    benchmark.group = "kernels/xnor_matmul"
    M, K, N = MKN
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    set_throughput(benchmark, 2 * M * N * K, "flops")
    benchmark(with_sync(lambda: a @ b, device))


# bit1_hamming_total: direct kernel vs (a^b).long().sum() reference.

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_hamming_kernel(benchmark, device, n_bits):
    benchmark.group = "kernels/hamming_total"
    a = brute.tensor(_rand_bool((n_bits,), device), dtype=brute.bit1)
    b = brute.tensor(_rand_bool((n_bits,), device, seed=1), dtype=brute.bit1)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: torch.ops.brute.bit1_hamming_total(
        a._packed_buf, b._packed_buf), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_hamming_ref(benchmark, device, n_bits):
    """Reference: (a ^ b).long().sum() — naïve bool implementation."""
    benchmark.group = "kernels/hamming_total"
    a = _rand_bool((n_bits,), device, seed=0)
    b = _rand_bool((n_bits,), device, seed=1)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: (a ^ b).long().sum(), device))


# Creation: torch.bool randint vs brute.bit1 random creation.

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_randomize_(benchmark, device, n_bits):
    benchmark.group = "kernels/randomize_"
    x = brute.tensor(_rand_bool((n_bits,), device), dtype=brute.bit1)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.randomize_(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_random_(benchmark, device, n_bits):
    """Reference: bool .random_(0, 2) — torch's native in-place random."""
    benchmark.group = "kernels/randomize_"
    x = _rand_bool((n_bits,), device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.random_(0, 2), device))


# word_popcount: per-word popcount kernel.

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_word_popcount(benchmark, device, n_bits):
    benchmark.group = "kernels/word_popcount"
    a = brute.tensor(_rand_bool((n_bits,), device), dtype=brute.bit1)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: a.word_popcount(), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_word_popcount_ref(benchmark, device, n_bits):
    """Reference: bool.long().sum() — single fused scalar reduction."""
    benchmark.group = "kernels/word_popcount"
    x = _rand_bool((n_bits,), device)
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(lambda: x.long().sum(), device))


# creation: brute.zeros / torch.zeros (bool).

@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_create_zeros(benchmark, device, n_bits):
    benchmark.group = "kernels/create_zeros"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: brute.zeros(n_bits, dtype=brute.bit1, device=device), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_create_zeros(benchmark, device, n_bits):
    benchmark.group = "kernels/create_zeros"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: torch.zeros(n_bits, dtype=torch.bool, device=device), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bit1_create_ones(benchmark, device, n_bits):
    benchmark.group = "kernels/create_ones"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: brute.ones(n_bits, dtype=brute.bit1, device=device), device))


@pytest.mark.parametrize("n_bits", VECTOR_SCALES)
def test_bool_create_ones(benchmark, device, n_bits):
    benchmark.group = "kernels/create_ones"
    set_throughput(benchmark, n_bits, "bits")
    benchmark(with_sync(
        lambda: torch.ones(n_bits, dtype=torch.bool, device=device), device))
