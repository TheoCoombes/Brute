"""Memory leak / weakref tests."""
from __future__ import annotations

import gc
import weakref

import pytest
import torch

import brute

pytestmark = pytest.mark.stress


def test_no_reference_cycles_after_op():
    x = brute.randint(0, 2, (128, 128), dtype=brute.bit1)
    ref = weakref.ref(x)
    del x
    gc.collect()
    assert ref() is None


def test_chained_ops_release_intermediates():
    """A long expression should not retain its intermediates after the result is dropped."""
    leaked = []
    for _ in range(8):
        a = brute.randint(0, 2, (64,), dtype=brute.bit1)
        b = brute.randint(0, 2, (64,), dtype=brute.bit1)
        c = (a & b) | (a ^ b)
        leaked.append(weakref.ref(c))
        del a, b, c
    gc.collect()
    alive = [r for r in leaked if r() is not None]
    assert alive == [], f"{len(alive)} intermediate(s) leaked"


def test_clone_does_not_leak():
    base = brute.zeros(128, dtype=brute.bit1)
    refs = []
    for _ in range(64):
        c = base.clone()
        refs.append(weakref.ref(c))
        del c
    gc.collect()
    alive = [r for r in refs if r() is not None]
    assert alive == [], f"{len(alive)} clones leaked"
