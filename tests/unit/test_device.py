"""Device handling: transfers, pinned memory, async copies, default device."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.conftest import DEVICES
from tests.helpers import assert_bit1_matches_bool, assert_device, bit1


@pytest.mark.parametrize("dev", DEVICES)
def test_creation_on_device(dev):
    t = brute.tensor([True, False, True], dtype=brute.bit1, device=dev)
    assert_device(t, dev)


@pytest.mark.parametrize("src", DEVICES)
@pytest.mark.parametrize("dst", DEVICES)
def test_device_roundtrip(src, dst):
    """Cross-device transfer must preserve values."""
    cpu_ref = torch.tensor([True, False, True, False, True, True], dtype=torch.bool)
    x = bit1(cpu_ref.to(src))
    y = x.to(dst)
    assert_device(y, dst)
    assert y.dtype == brute.bit1
    assert torch.equal(y.bool().cpu().as_subclass(torch.Tensor), cpu_ref)


@pytest.mark.parametrize("dst", DEVICES)
def test_to_device_string(dst):
    x = brute.tensor([True, False], dtype=brute.bit1)
    y = x.to(dst)
    assert_device(y, dst)


@pytest.mark.parametrize("dst", DEVICES)
def test_to_device_obj(dst):
    x = brute.tensor([True, False], dtype=brute.bit1)
    y = x.to(torch.device(dst))
    assert_device(y, dst)


@pytest.mark.cuda
def test_pinned_memory():
    x = brute.zeros(128, dtype=brute.bit1)
    pinned = x.bool().as_subclass(torch.Tensor).pin_memory()
    assert pinned.is_pinned()


@pytest.mark.cuda
def test_non_blocking_transfer():
    x = brute.zeros(128, dtype=brute.bit1)
    y = x.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert_device(y, "cuda")


def test_device_inference_from_input(device):
    """When dtype=bit1 with a device-typed input, the result lives on that device."""
    src = torch.tensor([True, False, True], device=device)
    t = brute.tensor(src, dtype=brute.bit1)
    assert_device(t, device)


def test_device_consistency_across_ops(device):
    a = brute.tensor([True, False, True], dtype=brute.bit1, device=device)
    b = brute.tensor([False, True, True], dtype=brute.bit1, device=device)
    out = a & b
    assert_device(out, device)


def test_cpu_to_cpu_noop():
    x = brute.tensor([True, False], dtype=brute.bit1)
    y = x.cpu()
    assert_device(y, "cpu")
    assert torch.equal(x.bool().as_subclass(torch.Tensor),
                       y.bool().as_subclass(torch.Tensor))
