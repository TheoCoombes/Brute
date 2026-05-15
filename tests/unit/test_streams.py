"""CUDA stream sanity."""
from __future__ import annotations

import pytest
import torch

import brute


@pytest.mark.cuda
def test_op_on_stream():
    stream = torch.cuda.Stream()
    x = brute.randint(0, 2, (1024, 1024), dtype=brute.bit1, device="cuda")
    with torch.cuda.stream(stream):
        y = brute.logical_not(x)
    torch.cuda.synchronize()
    assert y.device.type == "cuda"


@pytest.mark.cuda
def test_async_transfer():
    x = brute.zeros(256, dtype=brute.bit1)
    y = x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert y.device.type == "cuda"


@pytest.mark.cuda
def test_event_record_wait():
    """A second stream waits on an event from the first stream."""
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    x = brute.randint(0, 2, (128,), dtype=brute.bit1, device="cuda")
    evt = torch.cuda.Event()
    with torch.cuda.stream(s1):
        y = brute.logical_not(x)
        evt.record(s1)
    with torch.cuda.stream(s2):
        evt.wait(s2)
        z = brute.logical_not(y)
    torch.cuda.synchronize()
    assert z.dtype == brute.bit1
