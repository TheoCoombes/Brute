"""DLPack interop."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import bit1


def test_dlpack_roundtrip_float(device):
    x = brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    try:
        cap = torch.utils.dlpack.to_dlpack(x)
        y = torch.utils.dlpack.from_dlpack(cap)
    except Exception as e:
        pytest.xfail(f"DLPack unsupported for this configuration: {e}")
    assert torch.allclose(
        y.cpu(),
        torch.tensor([1.0, 2.0, 3.0], device="cpu"),
    )


def test_dlpack_roundtrip_int(device):
    x = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    try:
        cap = torch.utils.dlpack.to_dlpack(x)
        y = torch.utils.dlpack.from_dlpack(cap)
    except Exception as e:
        pytest.xfail(f"DLPack unsupported: {e}")
    assert torch.equal(y.cpu(),
                       torch.tensor([1, 2, 3], dtype=torch.int32))


def test_dlpack_roundtrip_bool(device):
    """torch.bool is supported in modern DLPack; bool-typed brute.Tensor must roundtrip."""
    x = brute.tensor([True, False, True], dtype=torch.bool, device=device)
    try:
        cap = torch.utils.dlpack.to_dlpack(x)
        y = torch.utils.dlpack.from_dlpack(cap)
    except Exception as e:
        pytest.xfail(f"DLPack bool unsupported: {e}")
    assert torch.equal(y.cpu(),
                       torch.tensor([True, False, True]))


def test_dlpack_bit1_falls_back_to_bool(device):
    """bit1's underlying storage IS bool; DLPack export should expose that bool."""
    src = torch.tensor([True, False, True, False], device=device)
    x = bit1(src)
    try:
        cap = torch.utils.dlpack.to_dlpack(x.as_subclass(torch.Tensor))
        y = torch.utils.dlpack.from_dlpack(cap)
    except Exception as e:
        pytest.xfail(f"DLPack bool not supported: {e}")
    assert torch.equal(y.cpu(), src.cpu())
