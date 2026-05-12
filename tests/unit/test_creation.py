"""Tensor creation: factory functions and constructors should be drop-in for torch."""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, assert_dtype, assert_subclass, bit1


@pytest.mark.parametrize("shape", [(), (0,), (1,), (4,), (3, 5), (2, 3, 4), (2, 3, 4, 5)])
def test_zeros_bool_parity(shape, device):
    t = brute.zeros(*shape, dtype=torch.bool, device=device)
    ref = torch.zeros(shape, dtype=torch.bool, device=device)
    assert_subclass(t)
    assert_dtype(t, torch.bool)
    assert torch.equal(t.as_subclass(torch.Tensor), ref)


@pytest.mark.parametrize("shape", [(0,), (1,), (4,), (3, 5), (2, 3, 4)])
def test_zeros_bit1(shape, device):
    t = brute.zeros(*shape, dtype=brute.bit1, device=device)
    assert_subclass(t)
    assert_dtype(t, brute.bit1)
    ref = torch.zeros(shape, dtype=torch.bool, device=device)
    assert_bit1_matches_bool(t, ref)


@pytest.mark.parametrize("shape", [(0,), (1,), (4,), (3, 5), (2, 3, 4)])
def test_ones_bit1(shape, device):
    t = brute.ones(*shape, dtype=brute.bit1, device=device)
    assert_dtype(t, brute.bit1)
    ref = torch.ones(shape, dtype=torch.bool, device=device)
    assert_bit1_matches_bool(t, ref)


@pytest.mark.parametrize("shape", [(4,), (3, 5)])
def test_empty_bit1(shape, device):
    t = brute.empty(*shape, dtype=brute.bit1, device=device)
    assert_dtype(t, brute.bit1)
    assert t.shape == shape


@pytest.mark.parametrize("fill", [True, False])
def test_full_bit1(fill, device):
    t = brute.full((4, 5), fill, dtype=brute.bit1, device=device)
    ref = torch.full((4, 5), fill, dtype=torch.bool, device=device)
    assert_bit1_matches_bool(t, ref)


def test_tensor_from_list_bool(device):
    data = [[True, False, True], [False, True, False]]
    t = brute.tensor(data, dtype=torch.bool, device=device)
    ref = torch.tensor(data, dtype=torch.bool, device=device)
    assert_dtype(t, torch.bool)
    assert torch.equal(t.as_subclass(torch.Tensor), ref)


def test_tensor_from_list_bit1(device):
    data = [[True, False, True], [False, True, False]]
    t = brute.tensor(data, dtype=brute.bit1, device=device)
    ref = torch.tensor(data, dtype=torch.bool, device=device)
    assert_dtype(t, brute.bit1)
    assert_bit1_matches_bool(t, ref)


def test_tensor_from_tensor(device):
    src = torch.tensor([True, False, True], device=device)
    t = brute.tensor(src, dtype=brute.bit1)
    assert_bit1_matches_bool(t, src)


def test_as_tensor_no_copy_for_bool(device):
    src = torch.tensor([True, False, True], device=device)
    t = brute.as_tensor(src)
    assert isinstance(t, brute.Tensor)
    assert t.dtype == torch.bool


def test_as_tensor_to_bit1(device):
    src = torch.tensor([True, False, True], device=device)
    t = brute.as_tensor(src, dtype=brute.bit1)
    assert_dtype(t, brute.bit1)
    assert_bit1_matches_bool(t, src)


@pytest.mark.parametrize("shape", [(8,), (16, 32), (3, 4, 5)])
def test_randint_bit1(shape, device, seed):
    t = brute.randint(0, 2, shape, dtype=brute.bit1, device=device)
    assert_dtype(t, brute.bit1)
    assert t.shape == shape
    # All values should be valid bools.
    materialised = t.bool().as_subclass(torch.Tensor)
    assert materialised.dtype == torch.bool


@pytest.mark.parametrize("shape", [(8,), (4, 5)])
def test_rand_bit1(shape, device, seed):
    t = brute.rand(*shape, dtype=brute.bit1, device=device)
    assert_dtype(t, brute.bit1)
    assert t.shape == shape


@pytest.mark.parametrize("shape", [(4,), (3, 4)])
def test_randn_bit1(shape, device, seed):
    t = brute.randn(*shape, dtype=brute.bit1, device=device)
    assert_dtype(t, brute.bit1)
    assert t.shape == shape


def test_arange_returns_brute_tensor():
    t = brute.arange(8)
    assert_subclass(t)
    assert torch.equal(t.as_subclass(torch.Tensor), torch.arange(8))


def test_linspace_returns_brute_tensor():
    t = brute.linspace(0.0, 1.0, 5)
    assert_subclass(t)


def test_eye_returns_brute_tensor():
    t = brute.eye(4)
    assert_subclass(t)
    assert torch.equal(t.as_subclass(torch.Tensor), torch.eye(4))


@pytest.mark.parametrize("pack_dtype", [torch.uint8, torch.uint32, torch.uint64])
def test_factory_accepts_pack_dtype(pack_dtype, device):
    t = brute.zeros(64, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
    assert t.pack_dtype == pack_dtype


def test_zeros_default_dtype_is_torch_float32():
    """Default dtype (no override) must match torch's default."""
    t = brute.zeros(4)
    assert t.dtype == torch.get_default_dtype()


def test_creation_preserves_subclass():
    for fn, args in [
        (brute.zeros, (4,)),
        (brute.ones, (4,)),
        (brute.empty, (4,)),
        (brute.arange, (8,)),
    ]:
        t = fn(*args)
        assert isinstance(t, brute.Tensor), f"{fn.__name__} did not return brute.Tensor"


def test_like_factories_inherit_dtype(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    z = brute.zeros_like(a)
    assert_dtype(z, brute.bit1)
    o = brute.ones_like(a)
    assert_dtype(o, brute.bit1)


def test_like_factories_override_dtype(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    z = brute.zeros_like(a, dtype=torch.float32)
    assert_dtype(z, torch.float32)


def test_constructor_with_dtype_torch_bool(device):
    t = brute.Tensor(torch.tensor([True, False]), dtype=torch.bool, device=device)
    assert_dtype(t, torch.bool)


def test_constructor_with_dtype_bit1(device):
    t = brute.Tensor(torch.tensor([True, False]), dtype=brute.bit1, device=device)
    assert_dtype(t, brute.bit1)


@pytest.mark.parametrize("data,expected_dtype", [
    ([1, 2, 3], torch.int64),
    ([1.0, 2.0], torch.get_default_dtype()),
    ([True, False], torch.bool),
])
def test_tensor_dtype_inference(data, expected_dtype):
    t = brute.tensor(data)
    assert t.dtype == expected_dtype
