"""Global pytest fixtures and configuration for the brute test suite.

Covers:
- device parametrization (cpu / cuda / mps)
- pack_dtype parametrization (uint8 / uint32 / uint64)
- contiguity / view transforms
- a deterministic seed fixture
"""
from __future__ import annotations

import os
import random

import pytest
import torch

import brute


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


PACK_DTYPES: list[torch.dtype] = [torch.uint8, torch.uint32, torch.uint64]


# Full torch dtype matrix (used by dtype-promotion tests).
ALL_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=DEVICES)
def device(request):
    """Run a test on every available device."""
    return request.param


@pytest.fixture(params=PACK_DTYPES, ids=lambda d: str(d).replace("torch.", ""))
def pack_dtype(request):
    """Run a test against every supported packing width."""
    return request.param


@pytest.fixture
def seed():
    """Deterministic seed for the duration of one test."""
    s = int(os.environ.get("BRUTE_TEST_SEED", "1234"))
    torch.manual_seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


# ---------------------------------------------------------------------------
# Pytest markers / configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "stress: long-running stress tests"
    )
    config.addinivalue_line(
        "markers", "fuzz: hypothesis-driven fuzz tests"
    )
    config.addinivalue_line(
        "markers", "cuda: requires CUDA"
    )
    config.addinivalue_line(
        "markers", "mps: requires MPS"
    )
    config.addinivalue_line(
        "markers", "needs_dispatch_parity: requires bit1 dispatch parity with torch.bool"
    )


def pytest_collection_modifyitems(config, items):
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_mps = pytest.mark.skip(reason="MPS not available")
    for item in items:
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not torch.backends.mps.is_available():
            item.add_marker(skip_mps)


# Expose helpers to tests via the `brute_test` namespace (avoids long imports).
@pytest.fixture(scope="session")
def brute_pkg():
    return brute
