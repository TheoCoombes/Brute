"""
brute.functional — factory and utility functions.

Only overrides a torch function when bit1 requires special handling.
For all other dtypes these are thin wrappers that keep results inside brute.
"""

from __future__ import annotations

import torch

from brute.tensor import Tensor, _Bit1DType, _resolve_pack_dtype, bit1


# ── Internal normaliser ───────────────────────────────────────────────────────

def _norm_size(size) -> tuple:
    """Accept zeros(2,3), zeros((2,3)), or zeros([2,3])."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        return tuple(size[0])
    return size


# ── Creators: bit1-aware ──────────────────────────────────────────────────────

def zeros(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        return Tensor._make_bit1(torch.zeros(size, dtype=torch.bool, device=device),
                                 _resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.zeros(size, dtype=dtype, device=device, **kwargs))


def ones(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        return Tensor._make_bit1(torch.ones(size, dtype=torch.bool, device=device),
                                 _resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.ones(size, dtype=dtype, device=device, **kwargs))


def empty(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        return Tensor._make_bit1(torch.empty(size, dtype=torch.bool, device=device),
                                 _resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.empty(size, dtype=dtype, device=device, **kwargs))


def full(size, fill_value, *, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    if isinstance(size, int):
        size = (size,)
    if isinstance(dtype, _Bit1DType):
        return Tensor._make_bit1(
            torch.full(size, bool(fill_value), dtype=torch.bool, device=device),
            _resolve_pack_dtype(pack_dtype),
        )
    return Tensor._make_plain(torch.full(size, fill_value, dtype=dtype, device=device, **kwargs))


def tensor(data, *, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    if isinstance(dtype, _Bit1DType):
        return Tensor._make_bit1(torch.tensor(data, dtype=torch.bool, device=device),
                                 _resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.tensor(data, dtype=dtype, device=device, **kwargs))


def as_tensor(data, *, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    if isinstance(dtype, _Bit1DType):
        if isinstance(data, torch.Tensor):
            bool_t = data.as_subclass(torch.Tensor) if isinstance(data, Tensor) else data
            bool_t = bool_t if bool_t.dtype == torch.bool else bool_t.bool()
        else:
            bool_t = torch.as_tensor(data, dtype=torch.bool, device=device)
        if device is not None:
            bool_t = bool_t.to(device=device)
        return Tensor._make_bit1(bool_t, _resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.as_tensor(data, dtype=dtype, device=device))


def from_numpy(ndarray) -> Tensor:
    return Tensor._make_plain(torch.from_numpy(ndarray))


# ── Random creators ───────────────────────────────────────────────────────────

def rand(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8,
         generator=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.rand(size, device=device, **gen_kw) >= 0.5
        return Tensor._make_bit1(bool_t, _resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.rand(size, dtype=dtype, device=device, **gen_kw, **kwargs))


def randn(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8,
          generator=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.randn(size, device=device, **gen_kw) >= 0
        return Tensor._make_bit1(bool_t, _resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.randn(size, dtype=dtype, device=device, **gen_kw, **kwargs))


def randint(low_or_high, high=None, size=None, *, dtype=None, device=None,
            pack_dtype: torch.dtype = torch.uint8, generator=None, **kwargs) -> Tensor:
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        s = size if size is not None else (low_or_high,)
        bool_t = torch.randint(0, 2, s, device=device, **gen_kw).bool()
        return Tensor._make_bit1(bool_t, _resolve_pack_dtype(pack_dtype))
    if high is None:
        return Tensor._make_plain(torch.randint(low_or_high, size, dtype=dtype, device=device, **gen_kw, **kwargs))
    return Tensor._make_plain(torch.randint(low_or_high, high, size, dtype=dtype, device=device, **gen_kw, **kwargs))


def _inherit_pack_dtype(input: Tensor, pack_dtype: torch.dtype) -> torch.dtype:
    """Return the caller-supplied pack_dtype if explicitly set, else inherit from input."""
    if pack_dtype is not torch.uint8:
        return _resolve_pack_dtype(pack_dtype)
    inherited = getattr(input, '_pack_dtype', None)
    return inherited if inherited is not None else torch.uint8


def rand_like(input: Tensor, *, dtype=None, device=None,
              pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.rand_like(input.as_subclass(torch.Tensor).float()) >= 0.5
        dev = device or input.device
        return Tensor._make_bit1(bool_t.to(device=dev), _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.rand_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def randn_like(input: Tensor, *, dtype=None, device=None,
               pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.randn_like(input.as_subclass(torch.Tensor).float()) >= 0
        dev = device or input.device
        return Tensor._make_bit1(bool_t.to(device=dev), _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.randn_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def zeros_like(input: Tensor, *, dtype=None, device=None,
               pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.zeros_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.zeros_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def ones_like(input: Tensor, *, dtype=None, device=None,
              pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.ones_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.ones_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


# ── Arange / linspace / eye ───────────────────────────────────────────────────

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


# ── Re-exports: work via __torch_function__ ────────────────────────────────────

all   = torch.all
any   = torch.any
sum   = torch.sum
max   = torch.max
min   = torch.min
mean  = torch.mean
prod  = torch.prod
cat   = torch.cat
stack = torch.stack
where = torch.where
