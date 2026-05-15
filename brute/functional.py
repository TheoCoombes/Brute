"""
brute.functional — factory and utility functions.

Only overrides a torch function when bit1 requires special handling.
For all other dtypes these are thin wrappers that keep results inside brute.
"""

from __future__ import annotations

import torch

from brute.tensor import Tensor, _Bit1DType
from brute.dtype import _PACK_WIDTH, _PACK_STORAGE_DTYPE


# Internal normaliser

def _norm_size(size) -> tuple:
    """Accept zeros(2,3), zeros((2,3)), or zeros([2,3])."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        return tuple(size[0])
    return size


def _packed_shape(size: tuple) -> tuple:
    """Compute the packed buffer shape for a given logical shape."""
    if len(size) == 0:
        return (1,)
    last = int(size[-1])
    n_words = (last + _PACK_WIDTH - 1) // _PACK_WIDTH if last > 0 else 0
    return tuple(size[:-1]) + (n_words,)


def _make_bit1_eager(packed: torch.Tensor, size: tuple, device) -> Tensor:
    """Wrap a freshly-allocated packed buffer as a bit1 tensor with empty
    (uninitialised) bool storage. The bool view is left dirty — it is unpacked
    lazily on first access.
    """
    bool_t = torch.empty(size, dtype=torch.bool, device=device)
    instance = bool_t.as_subclass(Tensor)
    instance._is_bit1    = True
    instance.__dict__['_packed_buf_cache'] = packed
    instance.__dict__['_packed_ver']       = instance._version
    instance.__dict__['_bool_dirty']       = True
    return instance


# Creators: bit1-aware

def zeros(*size, dtype=None, device=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        packed = torch.zeros(_packed_shape(size), dtype=_PACK_STORAGE_DTYPE, device=device)
        return _make_bit1_eager(packed, size, device)
    return Tensor._make_plain(torch.zeros(size, dtype=dtype, device=device, **kwargs))


def ones(*size, dtype=None, device=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        ps = _packed_shape(size)
        packed = torch.empty(ps, dtype=_PACK_STORAGE_DTYPE, device=device)
        if packed.numel() > 0:
            packed.view(torch.uint8).fill_(0xFF)
            if size and size[-1] > 0 and size[-1] % _PACK_WIDTH != 0:
                valid_bits = size[-1] % _PACK_WIDTH
                mask = (1 << valid_bits) - 1
                packed[..., -1] = mask
        return _make_bit1_eager(packed, size, device)
    return Tensor._make_plain(torch.ones(size, dtype=dtype, device=device, **kwargs))


def empty(*size, dtype=None, device=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        ps = _packed_shape(size)
        packed = torch.empty(ps, dtype=_PACK_STORAGE_DTYPE, device=device)
        if size and size[-1] > 0 and size[-1] % _PACK_WIDTH != 0 and packed.numel() > 0:
            packed[..., -1] = 0
        return _make_bit1_eager(packed, size, device)
    return Tensor._make_plain(torch.empty(size, dtype=dtype, device=device, **kwargs))


def full(size, fill_value, *, dtype=None, device=None, **kwargs) -> Tensor:
    if isinstance(size, int):
        size = (size,)
    if isinstance(dtype, _Bit1DType):
        return ones(*size, dtype=dtype, device=device) \
            if bool(fill_value) else \
            zeros(*size, dtype=dtype, device=device)
    return Tensor._make_plain(torch.full(size, fill_value, dtype=dtype, device=device, **kwargs))


def tensor(data, *, dtype=None, device=None, **kwargs) -> Tensor:
    if isinstance(dtype, _Bit1DType):
        return Tensor._make_bit1(torch.tensor(data, dtype=torch.bool, device=device))
    return Tensor._make_plain(torch.tensor(data, dtype=dtype, device=device, **kwargs))


def as_tensor(data, *, dtype=None, device=None) -> Tensor:
    if isinstance(dtype, _Bit1DType):
        if isinstance(data, torch.Tensor):
            bool_t = data.as_subclass(torch.Tensor) if isinstance(data, Tensor) else data
            bool_t = bool_t if bool_t.dtype == torch.bool else bool_t.bool()
        else:
            bool_t = torch.as_tensor(data, dtype=torch.bool, device=device)
        if device is not None:
            bool_t = bool_t.to(device=device)
        return Tensor._make_bit1(bool_t)
    return Tensor._make_plain(torch.as_tensor(data, dtype=dtype, device=device))


def from_numpy(ndarray) -> Tensor:
    return Tensor._make_plain(torch.from_numpy(ndarray))


# Random creators

def rand(*size, dtype=None, device=None, generator=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.rand(size, device=device, **gen_kw) >= 0.5
        return Tensor._make_bit1(bool_t)
    return Tensor._make_plain(torch.rand(size, dtype=dtype, device=device, **gen_kw, **kwargs))


def randn(*size, dtype=None, device=None, generator=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.randn(size, device=device, **gen_kw) >= 0
        return Tensor._make_bit1(bool_t)
    return Tensor._make_plain(torch.randn(size, dtype=dtype, device=device, **gen_kw, **kwargs))


def randint(low_or_high, high=None, size=None, *, dtype=None, device=None,
            generator=None, **kwargs) -> Tensor:
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        s = size if size is not None else (low_or_high,)
        bool_t = torch.randint(0, 2, s, device=device, **gen_kw).bool()
        return Tensor._make_bit1(bool_t)
    if high is None:
        return Tensor._make_plain(torch.randint(low_or_high, size, dtype=dtype, device=device, **gen_kw, **kwargs))
    return Tensor._make_plain(torch.randint(low_or_high, high, size, dtype=dtype, device=device, **gen_kw, **kwargs))


def rand_like(input: Tensor, *, dtype=None, device=None) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.rand_like(input.as_subclass(torch.Tensor).float()) >= 0.5
        dev = device or input.device
        return Tensor._make_bit1(bool_t.to(device=dev))
    return Tensor._make_plain(torch.rand_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def randn_like(input: Tensor, *, dtype=None, device=None) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.randn_like(input.as_subclass(torch.Tensor).float()) >= 0
        dev = device or input.device
        return Tensor._make_bit1(bool_t.to(device=dev))
    return Tensor._make_plain(torch.randn_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def zeros_like(input: Tensor, *, dtype=None, device=None) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.zeros_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t)
    return Tensor._make_plain(torch.zeros_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def ones_like(input: Tensor, *, dtype=None, device=None) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.ones_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t)
    return Tensor._make_plain(torch.ones_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def full_like(input: Tensor, fill_value, *, dtype=None, device=None, **kwargs) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.full_like(input.as_subclass(torch.Tensor), bool(fill_value),
                                 dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t)
    return Tensor._make_plain(torch.full_like(
        input.as_subclass(torch.Tensor) if isinstance(input, Tensor) else input,
        fill_value, dtype=dtype, device=device, **kwargs,
    ))


def empty_like(input: Tensor, *, dtype=None, device=None, **kwargs) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.empty_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t)
    return Tensor._make_plain(torch.empty_like(
        input.as_subclass(torch.Tensor) if isinstance(input, Tensor) else input,
        dtype=dtype, device=device, **kwargs,
    ))


# Arange / linspace / eye

def arange(start, end=None, step=1, *, dtype=None, device=None, **kwargs) -> Tensor:
    if end is None:
        return Tensor._make_plain(torch.arange(start, dtype=dtype, device=device, **kwargs))
    return Tensor._make_plain(torch.arange(start, end, step, dtype=dtype, device=device, **kwargs))


def linspace(start, end, steps, *, dtype=None, device=None, **kwargs) -> Tensor:
    return Tensor._make_plain(torch.linspace(start, end, steps, dtype=dtype, device=device, **kwargs))


def eye(n, m=None, *, dtype=None, device=None, **kwargs) -> Tensor:
    if m is None:
        return Tensor._make_plain(torch.eye(n, dtype=dtype, device=device, **kwargs))
    return Tensor._make_plain(torch.eye(n, m, dtype=dtype, device=device, **kwargs))
