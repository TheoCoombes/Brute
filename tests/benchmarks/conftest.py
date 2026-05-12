"""Benchmark-suite conftest — skip cleanly if pytest-benchmark is missing."""
import pytest

pytest.importorskip("pytest_benchmark")
