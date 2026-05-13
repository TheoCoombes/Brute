"""Helpers shared across benchmark files.

The CPU vs CUDA vs MPS contrast only makes sense if we (a) synchronize the
device before stopping the timer, and (b) actually load enough work that we
escape the kernel-launch / dispatch noise floor. Scales below are chosen with
those two goals in mind.

Scale ID naming uses a `NN_label` prefix so the default lexicographic sort
pytest-benchmark applies produces a sensible small→huge ordering in the report.
"""
from __future__ import annotations

import torch
import pytest


# Scales for 1-D vector / flat-buffer benchmarks 
# Counted in logical bits. Ranges chosen to span:
#   tiny   – kernel-launch / dispatch overhead dominates
#   small  – L2 resident
#   medium – DRAM bandwidth
#   large  – multi-pass DRAM (CPU) / fits in GPU memory comfortably
#   huge   – stresses the device (skip-by-default if too slow)
VECTOR_SCALES = [
    pytest.param(1 << 10, id="01_tiny"),       # 1 Ki bits  ≈ 128 B
    pytest.param(1 << 16, id="02_small"),      # 64 Ki bits ≈   8 KiB
    pytest.param(1 << 20, id="03_medium"),     #  1 Mi bits ≈ 128 KiB
    pytest.param(1 << 24, id="04_large"),      # 16 Mi bits ≈   2 MiB
    pytest.param(1 << 28, id="05_huge"),       # 256 Mi bits ≈ 32 MiB
]


# Scales for 2-D matrices (rows × cols), bool/bit1 elementwise 
MATRIX_SCALES = [
    pytest.param((128,  128),  id="01_tiny"),
    pytest.param((512,  512),  id="02_small"),
    pytest.param((1024, 1024), id="03_medium"),
    pytest.param((2048, 4096), id="04_large"),
    pytest.param((4096, 8192), id="05_huge"),
]


# Scales for matmul: (M, K, N) — K stresses the inner dim 
MATMUL_SCALES = [
    pytest.param((128,  128,  128),   id="01_tiny"),
    pytest.param((256,  512,  256),   id="02_small"),
    pytest.param((512,  1024, 512),   id="03_medium"),
    pytest.param((1024, 2048, 1024),  id="04_large"),
    pytest.param((2048, 4096, 2048),  id="05_huge"),
]


# Device sync helpers 

def sync(device: str) -> None:
    """Block on the current device. Required to actually time GPU kernels."""
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

def set_throughput(benchmark, n_elems: int, label: str = "elems/s") -> None:
    """Attach a throughput metric to the benchmark for easier comparison."""
    benchmark.extra_info[label] = n_elems


# Scale guards 

def skip_if_mps_composite(device: str) -> None:  # noqa: ARG001
    """Historical guard for the pre-overhaul MPS path. After the Apple
    Silicon backend rewrite, `bit1_hamming_total` has a native MSL kernel
    (fused XOR + popcount + simdgroup reduce), so the composite fallback is
    no longer used on MPS and this guard is a no-op. Kept as a stable hook
    so individual benchmarks can re-introduce a device-specific skip without
    touching every call site."""
    return
