"""Benchmark + catalog scale tiers and device-sync helpers.

Scales are 3-tier (dispatch / medium / huge) so the matrix stays small enough
to iterate on without dropping signal at either end:

- **dispatch**: the minimum input size (a single packed word for 1D, (1,1) for
  matrices). Times almost-pure dispatch + kernel-launch overhead. If bit1 is
  slower than bool here, it is a pure Python-side cost.
- **medium**:  ~4 Mi bits / (2048, 2048) — exits L2 on most CPUs, fits in the
  GPU's outer-cache levels. The "fair" comparison point.
- **huge**:    ~64 Mi bits / (8192, 8192) — multi-pass DRAM on CPU, fills a
  decent chunk of GPU memory. Surfaces the asymptotic speedup.

Scale IDs use a `NN_label` prefix so the default lexicographic sort
pytest-benchmark applies produces a sensible small→huge ordering.
"""
from __future__ import annotations

import torch
import pytest


# 1-D / flat-buffer scales (counted in logical bits).
# `dispatch=64` = exactly one uint64 packed word; smallest representative input.
VECTOR_SCALES = [
    pytest.param(1 << 6,  id="01_dispatch"),   # 64 bits  (1 uint64 word)
    pytest.param(1 << 22, id="02_medium"),     # ~4 Mi bits  (~512 KiB)
    pytest.param(1 << 26, id="03_huge"),       # ~64 Mi bits (~8 MiB)
]


# 1-D scales for ops whose output is O(N²) (e.g. diag_embed, diagflat).
# Keep N small enough that the (N, N) output doesn't OOM.
DIAG_VECTOR_SCALES = [
    pytest.param(64,   id="01_dispatch"),   # → (64, 64)   ≈ 4 Ki elements
    pytest.param(2048, id="02_medium"),     # → (2048, 2048) ≈ 4 Mi elements
    pytest.param(4096, id="03_huge"),       # → (4096, 4096) ≈ 16 Mi elements
]


# 2-D matrix scales (rows, cols) — bool/bit1 elementwise.
MATRIX_SCALES = [
    pytest.param((1,    1),    id="01_dispatch"),
    pytest.param((2048, 2048), id="02_medium"),
    pytest.param((8192, 8192), id="03_huge"),
]


# Matmul scales: (M, K, N). K aligned to pack_width=64.
MATMUL_SCALES = [
    pytest.param((1,    64,   1),    id="01_dispatch"),
    pytest.param((512,  1024, 512),  id="02_medium"),
    pytest.param((2048, 4096, 2048), id="03_huge"),
]


# Number of bits / elements per scale tier — used to set throughput labels.
def scale_size(scale_param) -> int:
    """Return the bit/element count for a scale param."""
    v = scale_param
    if isinstance(v, tuple):
        n = 1
        for d in v:
            n *= d
        return n
    return int(v)


# Device sync helpers
def sync(device: str) -> None:
    """Block on the current device. Required to time GPU kernels honestly."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def with_sync(fn, device: str):
    """Wrap a thunk so its returned value is materialized before timing stops."""
    if device == "cpu":
        return fn
    def _run():
        out = fn()
        sync(device)
        return out
    return _run


# Throughput helper
def set_throughput(benchmark, n_elems: int, label: str = "bits") -> None:
    """Attach a throughput metric to the benchmark for easier comparison."""
    benchmark.extra_info[label] = n_elems
