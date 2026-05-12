"""Reusable assertions for brute tensors."""
from __future__ import annotations

import torch

import brute


def assert_tensors_equal(a, b, msg: str = ""):
    """Strict element-wise equality (shape + values; allows subclass differences)."""
    a_t = a.as_subclass(torch.Tensor) if isinstance(a, brute.Tensor) else a
    b_t = b.as_subclass(torch.Tensor) if isinstance(b, brute.Tensor) else b
    assert a_t.shape == b_t.shape, f"shape mismatch: {a_t.shape} vs {b_t.shape}. {msg}"
    if a_t.dtype != b_t.dtype:
        # Cast to common dtype to compare values.
        if a_t.dtype == torch.bool or b_t.dtype == torch.bool:
            a_t = a_t.bool() if a_t.dtype != torch.bool else a_t
            b_t = b_t.bool() if b_t.dtype != torch.bool else b_t
        else:
            a_t = a_t.float()
            b_t = b_t.float()
    assert torch.equal(a_t.cpu(), b_t.cpu()), f"values differ. {msg}"


def assert_bit1_matches_bool(bit1_tensor, bool_tensor, msg: str = ""):
    """Verify a bit1 tensor materializes to the same bool tensor as a torch.bool reference."""
    assert getattr(bit1_tensor, "_is_bit1", False), (
        f"expected a bit1 tensor, got dtype={bit1_tensor.dtype}. {msg}"
    )
    materialized = bit1_tensor.bool().as_subclass(torch.Tensor)
    ref = bool_tensor.as_subclass(torch.Tensor) if isinstance(bool_tensor, brute.Tensor) else bool_tensor
    assert ref.dtype == torch.bool, f"reference must be torch.bool, got {ref.dtype}. {msg}"
    assert materialized.shape == ref.shape, (
        f"shape mismatch: {materialized.shape} vs {ref.shape}. {msg}"
    )
    assert torch.equal(materialized.cpu(), ref.cpu()), f"bit1 vs bool mismatch. {msg}"


def assert_subclass(t, expected=brute.Tensor, msg: str = ""):
    """Verify a result is a brute.Tensor (subclass preservation)."""
    assert isinstance(t, expected), (
        f"expected {expected.__name__}, got {type(t).__name__}. {msg}"
    )


def assert_device(t, dev, msg: str = ""):
    """Verify a tensor's device matches expectation (compares device.type)."""
    if isinstance(dev, str):
        dev_type = dev
    else:
        dev_type = dev.type
    assert t.device.type == dev_type, (
        f"device mismatch: {t.device.type} vs {dev_type}. {msg}"
    )


def assert_dtype(t, expected_dtype, msg: str = ""):
    """Verify a tensor's dtype (including bit1)."""
    assert t.dtype == expected_dtype, (
        f"dtype mismatch: {t.dtype} vs {expected_dtype}. {msg}"
    )


def assert_same_storage(a, b, msg: str = ""):
    """Assert two tensors share the same underlying storage."""
    a_t = a.as_subclass(torch.Tensor) if isinstance(a, brute.Tensor) else a
    b_t = b.as_subclass(torch.Tensor) if isinstance(b, brute.Tensor) else b
    a_ptr = a_t.untyped_storage().data_ptr()
    b_ptr = b_t.untyped_storage().data_ptr()
    assert a_ptr == b_ptr, f"expected shared storage, got {a_ptr} vs {b_ptr}. {msg}"


def assert_different_storage(a, b, msg: str = ""):
    """Assert two tensors do NOT share storage (e.g. after .clone())."""
    a_t = a.as_subclass(torch.Tensor) if isinstance(a, brute.Tensor) else a
    b_t = b.as_subclass(torch.Tensor) if isinstance(b, brute.Tensor) else b
    a_ptr = a_t.untyped_storage().data_ptr()
    b_ptr = b_t.untyped_storage().data_ptr()
    assert a_ptr != b_ptr, f"expected distinct storage, got {a_ptr}. {msg}"
