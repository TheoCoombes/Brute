"""Pickle and torch.save / torch.load round-trips."""
from __future__ import annotations

import copy
import io
import pickle
import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


def test_pickle_bit1_roundtrip(device):
    x = brute.tensor([True, False, True, False], dtype=brute.bit1, device=device)
    blob = pickle.dumps(x)
    y = pickle.loads(blob)
    assert y.dtype == brute.bit1
    assert_bit1_matches_bool(y, torch.tensor([True, False, True, False], device=device))


def test_pickle_bool_roundtrip(device):
    x = brute.tensor([True, False, True], dtype=torch.bool, device=device)
    y = pickle.loads(pickle.dumps(x))
    assert y.dtype == torch.bool
    assert torch.equal(x.as_subclass(torch.Tensor), y.as_subclass(torch.Tensor))


def test_pickle_non_bool_roundtrip(device):
    x = brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    y = pickle.loads(pickle.dumps(x))
    assert y.dtype == torch.float32
    assert isinstance(y, brute.Tensor)


def test_torch_save_load_bit1(tmp_path, device):
    # Keep src as bool reference for value comparison after save/load.
    src = torch.randint(0, 2, (32, 32), dtype=torch.bool, device=device)
    x = bit1(src)
    path = tmp_path / "bit1.pt"
    torch.save(x, path)
    y = torch.load(path, weights_only=False)
    assert isinstance(y, brute.Tensor)
    assert y.dtype == brute.bit1
    assert_bit1_matches_bool(y, src)


def test_torch_save_load_bool(tmp_path, device):
    src = torch.tensor([True, False, True], device=device)
    x = brute.tensor(src, dtype=torch.bool)
    path = tmp_path / "bool.pt"
    torch.save(x, path)
    y = torch.load(path, weights_only=False)
    assert isinstance(y, brute.Tensor)
    assert torch.equal(x.as_subclass(torch.Tensor), y.as_subclass(torch.Tensor))


def test_deepcopy_bit1(device):
    # Keep src as bool reference to verify independence after fill_.
    src = torch.tensor([True, False, True], device=device)
    x = bit1(src)
    y = copy.deepcopy(x)
    assert y.dtype == brute.bit1
    assert_bit1_matches_bool(y, src)
    x.fill_(False)
    assert_bit1_matches_bool(y, src)


def test_save_load_through_buffer(device):
    x = brute.tensor([True, False, True], dtype=brute.bit1, device=device)
    buf = io.BytesIO()
    torch.save(x, buf)
    buf.seek(0)
    y = torch.load(buf, weights_only=False)
    assert y.dtype == brute.bit1
    assert_bit1_matches_bool(y, torch.tensor([True, False, True], device=device))


@pytest.mark.cuda
def test_cross_device_load_map_to_cpu(tmp_path):
    x = brute.tensor([True, False, True], dtype=brute.bit1, device="cuda")
    path = tmp_path / "x.pt"
    torch.save(x, path)
    y = torch.load(path, map_location="cpu", weights_only=False)
    assert y.device.type == "cpu"
    assert y.dtype == brute.bit1


