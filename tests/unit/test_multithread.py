"""Multithreaded mutation: must not crash, deadlock, or corrupt memory."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import brute
from tests.helpers import bit1


def _readonly_worker(tensor, iters: int):
    """Read-only worker — exercises packed-buffer cache from multiple threads."""
    total = 0
    for _ in range(iters):
        total += int(tensor.popcount().item())
    return total


def test_multithreaded_readonly():
    """Concurrent readers must not crash or corrupt state."""
    x = bit1(torch.randint(0, 2, (256,), dtype=torch.bool))
    expected = int(x.popcount().item())
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_readonly_worker, x, 50) for _ in range(4)]
        results = [f.result() for f in futures]
    for r in results:
        assert r == expected * 50


def _clone_worker(tensor, iters: int):
    for _ in range(iters):
        _ = tensor.clone()


def test_multithreaded_clone():
    """Cloning from multiple threads must not crash."""
    x = bit1(torch.zeros((128,), dtype=torch.bool))
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_clone_worker, x, 50) for _ in range(4)]
        for f in futures:
            f.result()


def _ops_worker(iters: int):
    """Stress the dispatcher with many independent ops."""
    for _ in range(iters):
        a = bit1(torch.randint(0, 2, (32,), dtype=torch.bool))
        b = bit1(torch.randint(0, 2, (32,), dtype=torch.bool))
        _ = a & b
        _ = a | b
        _ = ~a


def test_multithreaded_dispatch():
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_ops_worker, 50) for _ in range(4)]
        for f in futures:
            f.result()  # must not raise
