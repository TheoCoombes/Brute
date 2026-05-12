"""Automatic mixed precision (autocast) interop."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


@pytest.mark.cuda
def test_autocast_cuda():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        a = brute.tensor(torch.randn(4, 8, device="cuda"))
        b = brute.tensor(torch.randn(8, 4, device="cuda"))
        out = a @ b
    assert out.dtype in (torch.float16, torch.float32, torch.bfloat16)


def test_autocast_cpu():
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        a = brute.tensor(torch.randn(4, 8))
        b = brute.tensor(torch.randn(8, 4))
        out = a @ b
    # bf16 or float32 depending on dispatch.
    assert out.dtype in (torch.bfloat16, torch.float32)
