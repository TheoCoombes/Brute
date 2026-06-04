"""Pytest config for the binary-transformer test suite.

Puts ``model/`` on ``sys.path`` so the suite imports the model modules
(``model``, ``attention``, ``layers``, ``bep``, ``vsa``, ``data``) the same way
``train.py`` does, and pins deterministic CPU threading for reproducibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_MODEL_DIR = Path(__file__).resolve().parent.parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


@pytest.fixture(autouse=True)
def _determinism():
    torch.manual_seed(0)
    yield


@pytest.fixture
def gen():
    """A fresh seeded CPU generator (bit1 factories require a CPU generator)."""
    return torch.Generator(device="cpu").manual_seed(1234)


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run slow end-to-end learnability probes")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end training probe (use --run-slow)")
    config.addinivalue_line("markers", "mps: requires an Apple MPS device")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
