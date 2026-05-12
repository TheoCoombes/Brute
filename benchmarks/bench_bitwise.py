"""
Bitwise operation benchmarks: brute bit1 vs torch.bool.

Three variants per operation
  brute_hl   high-level API  (a & b on bit1 tensors — bool op + repack)
  brute_ll   low-level C++   (torch.ops.brute.* on packed uint8 buffers)
  bool_ref   torch baseline  (torch.bool tensors, 8 bytes per logical bit)

Install deps then run:
    pip install pytest-benchmark
    pytest benchmarks/bench_bitwise.py -v --benchmark-sort=name
    pytest benchmarks/bench_bitwise.py -v --benchmark-json=results_bitwise.json
"""

import pytest
import torch
import brute


# ── Sizes (logical element count) ─────────────────────────────────────────────

SIZES = pytest.mark.parametrize(
    "n",
    [8_192, 1_048_576, 8_388_608],
    ids=["8K", "1M", "8M"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bit1(n, device):
    return brute.rand(n, dtype=brute.bit1, pack_dtype=brute.uint64, device=device)

def _bool(n, device):
    return torch.randint(0, 2, (n,), dtype=torch.bool, device=device)


# ── AND ───────────────────────────────────────────────────────────────────────

@SIZES
def test_and_brute_hl(benchmark, device, synced, n):
    a, b = _bit1(n, device), _bit1(n, device)
    def run():
        r = a & b
        synced()
        return r
    benchmark(run)


@SIZES
def test_and_brute_ll(benchmark, device, synced, n):
    a, b = _bit1(n, device), _bit1(n, device)
    pa, pb = a._packed_buf, b._packed_buf
    def run():
        r = torch.ops.brute.bitwise_and(pa, pb)
        synced()
        return r
    benchmark(run)


@SIZES
def test_and_bool_ref(benchmark, device, synced, n):
    a, b = _bool(n, device), _bool(n, device)
    def run():
        r = a & b
        synced()
        return r
    benchmark(run)


# ── OR ────────────────────────────────────────────────────────────────────────

@SIZES
def test_or_brute_hl(benchmark, device, synced, n):
    a, b = _bit1(n, device), _bit1(n, device)
    def run():
        r = a | b
        synced()
        return r
    benchmark(run)


@SIZES
def test_or_brute_ll(benchmark, device, synced, n):
    a, b = _bit1(n, device), _bit1(n, device)
    pa, pb = a._packed_buf, b._packed_buf
    def run():
        r = torch.ops.brute.bitwise_or(pa, pb)
        synced()
        return r
    benchmark(run)


@SIZES
def test_or_bool_ref(benchmark, device, synced, n):
    a, b = _bool(n, device), _bool(n, device)
    def run():
        r = a | b
        synced()
        return r
    benchmark(run)


# ── XOR ───────────────────────────────────────────────────────────────────────

@SIZES
def test_xor_brute_hl(benchmark, device, synced, n):
    a, b = _bit1(n, device), _bit1(n, device)
    def run():
        r = a ^ b
        synced()
        return r
    benchmark(run)


@SIZES
def test_xor_brute_ll(benchmark, device, synced, n):
    a, b = _bit1(n, device), _bit1(n, device)
    pa, pb = a._packed_buf, b._packed_buf
    def run():
        r = torch.ops.brute.bitwise_xor(pa, pb)
        synced()
        return r
    benchmark(run)


@SIZES
def test_xor_bool_ref(benchmark, device, synced, n):
    a, b = _bool(n, device), _bool(n, device)
    def run():
        r = a ^ b
        synced()
        return r
    benchmark(run)


# ── NOT ───────────────────────────────────────────────────────────────────────

@SIZES
def test_not_brute_hl(benchmark, device, synced, n):
    a = _bit1(n, device)
    def run():
        r = ~a
        synced()
        return r
    benchmark(run)


@SIZES
def test_not_brute_ll(benchmark, device, synced, n):
    a = _bit1(n, device)
    pa = a._packed_buf
    def run():
        r = torch.ops.brute.bitwise_not(pa)
        synced()
        return r
    benchmark(run)


@SIZES
def test_not_bool_ref(benchmark, device, synced, n):
    a = _bool(n, device)
    def run():
        r = ~a
        synced()
        return r
    benchmark(run)


# ── POPCOUNT ──────────────────────────────────────────────────────────────────

@SIZES
def test_popcount_brute_hl(benchmark, device, synced, n):
    """a.popcount() — calls packed_popcount internally, no bool materialisation."""
    a = _bit1(n, device)
    def run():
        r = a.popcount()
        synced()
        return r
    benchmark(run)


@SIZES
def test_popcount_brute_ll(benchmark, device, synced, n):
    a = _bit1(n, device)
    pa = a._packed_buf
    def run():
        r = torch.ops.brute.packed_popcount(pa)
        synced()
        return r
    benchmark(run)


@SIZES
def test_popcount_bool_ref(benchmark, device, synced, n):
    a = _bool(n, device)
    def run():
        r = a.long().sum()
        synced()
        return r
    benchmark(run)


# ── HAMMING DISTANCE ──────────────────────────────────────────────────────────
# No high-level Python API; compare C++ packed op vs XOR+sum on bool.

@SIZES
def test_hamming_brute_ll(benchmark, device, synced, n):
    a, b = _bit1(n, device), _bit1(n, device)
    pa, pb = a._packed_buf, b._packed_buf
    def run():
        r = torch.ops.brute.hamming_distance(pa, pb)
        synced()
        return r
    benchmark(run)


@SIZES
def test_hamming_bool_ref(benchmark, device, synced, n):
    a, b = _bool(n, device), _bool(n, device)
    def run():
        r = (a ^ b).long().sum()
        synced()
        return r
    benchmark(run)


# ── RANDOMIZE BITS ────────────────────────────────────────────────────────────
# brute_ll  : in-place fill of packed uint8 buffer (n//8 bytes)
# uint8_ref : in-place fill of a uint8 buffer the same byte size (memory-fair)
# bool_alloc: allocate + fill a fresh bool tensor (8x more storage, shows cost)

@SIZES
def test_randomize_brute_ll(benchmark, device, synced, n):
    a = _bit1(n, device)
    pa = a._packed_buf
    def run():
        torch.ops.brute.randomize_bits(pa)
        synced()
    benchmark(run)


@SIZES
def test_randomize_uint8_ref(benchmark, device, synced, n):
    """Same byte count as the packed buffer; uses PyTorch's built-in random_."""
    packed_n = (n + 7) // 8
    buf = torch.empty(packed_n, dtype=torch.uint8, device=device)
    def run():
        buf.random_()  # fills uint8 uniformly in [0, 255]
        synced()
    benchmark(run)


@SIZES
def test_randomize_bool_alloc(benchmark, device, synced, n):
    """Fresh bool tensor allocation each call (8x the memory of packed buffer)."""
    def run():
        r = torch.randint(0, 2, (n,), dtype=torch.bool, device=device)
        synced()
        return r
    benchmark(run)
