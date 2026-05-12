"""CUDA stream sanity."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


@pytest.mark.cuda
def test_op_on_stream():
    stream = torch.cuda.Stream()
    x = bit1(torch.randint(0, 2, (1024, 1024), dtype=torch.bool, device="cuda"))
    with torch.cuda.stream(stream):
        y = torch.logical_not(x)
    torch.cuda.synchronize()
    assert y.device.type == "cuda"


@pytest.mark.cuda
def test_async_transfer():
    x = bit1(torch.zeros((256,), dtype=torch.bool))
    y = x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert y.device.type == "cuda"


@pytest.mark.cuda
def test_event_record_wait():
    """A second stream waits on an event from the first stream."""
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    x = bit1(torch.randint(0, 2, (128,), dtype=torch.bool, device="cuda"))
    evt = torch.cuda.Event()
    with torch.cuda.stream(s1):
        y = torch.logical_not(x)
        evt.record(s1)
    with torch.cuda.stream(s2):
        evt.wait(s2)
        z = torch.logical_not(y)
    torch.cuda.synchronize()
    assert z.dtype == brute.bit1
