"""Layout / metadata APIs (shape, stride, dim, numel, size, layout)."""
from __future__ import annotations

import pytest
import torch

import brute


def test_shape(device):
    x = brute.zeros((3, 4, 5), dtype=brute.bit1, device=device)
    assert x.shape == (3, 4, 5)
    assert tuple(x.size()) == (3, 4, 5)


def test_ndim(device):
    x = brute.zeros((3, 4), dtype=brute.bit1, device=device)
    assert x.ndim == 2


def test_numel(device):
    x = brute.zeros((3, 4), dtype=brute.bit1, device=device)
    assert x.numel() == 12


def test_dim(device):
    x = brute.zeros((3, 4), dtype=brute.bit1, device=device)
    assert x.dim() == 2


def test_layout(device):
    x = brute.zeros((3, 4), dtype=brute.bit1, device=device)
    assert x.layout == torch.strided


def test_size_with_dim(device):
    x = brute.zeros((3, 4, 5), dtype=brute.bit1, device=device)
    assert x.size(0) == 3
    assert x.size(1) == 4
    assert x.size(-1) == 5


def test_stride_with_dim(device):
    x = brute.zeros((3, 4, 5), dtype=brute.bit1, device=device)
    strides = x.stride()
    assert strides == (20, 5, 1)
    assert x.stride(0) == 20
    assert x.stride(-1) == 1


def test_element_size_raises_for_bit1(device):
    x = brute.zeros(4, dtype=brute.bit1, device=device)
    with pytest.raises(TypeError):
        x.element_size()


def test_element_size_for_other_dtypes(device):
    x = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    assert x.element_size() == 4


def test_itemsize_raises_for_bit1(device):
    x = brute.zeros(4, dtype=brute.bit1, device=device)
    with pytest.raises(TypeError):
        _ = x.itemsize


def test_itemsize_for_other_dtypes(device):
    x = brute.tensor([1, 2, 3], dtype=torch.int32, device=device)
    assert x.itemsize == 4


def test_nbytes_bit1(device):
    x = brute.zeros(64, dtype=brute.bit1, device=device)
    assert x.nbytes > 0


def test_type_method_returns_string_for_bit1(device):
    x = brute.zeros(4, dtype=brute.bit1, device=device)
    assert "Bit1" in x.type() or "bit1" in x.type().lower()


def test_type_method_casts(device):
    x = brute.tensor([True, False, True], dtype=brute.bit1, device=device)
    casted = x.type(torch.float32)
    assert casted.dtype == torch.float32


def test_repr_includes_bit1():
    x = brute.tensor([True, False], dtype=brute.bit1)
    s = repr(x)
    assert "bit1" in s
    assert "brute" in s
