"""Tensor + shape generators for unit and fuzz tests.

The blueprint specifies `brute.bit1(x)` as a constructor. The actual brute API
exposes bit1 as a *dtype*, so we provide a thin `bit1(x)` helper here that
constructs a bit1 brute.Tensor from any boolean-coercible input.
"""
from __future__ import annotations

import torch

import brute


def bit1(data, *, pack_dtype=None, device=None):
    """Construct a bit1 brute.Tensor from any boolean-coercible input.

    Mirrors the convention used throughout the blueprint:
        bit_t = bit1(bool_tensor)
    """
    if pack_dtype is None:
        return brute.tensor(data, dtype=brute.bit1, device=device)
    return brute.tensor(data, dtype=brute.bit1, pack_dtype=pack_dtype, device=device)


def bool_tensor(data, *, device=None):
    """Construct a plain torch.bool tensor from raw data on a given device."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=torch.bool, device=device) if device else data.bool()
    return torch.tensor(data, dtype=torch.bool, device=device)


def random_bool(shape, device="cpu", seed: int | None = None) -> torch.Tensor:
    """Random torch.bool tensor of `shape`."""
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        cpu = torch.randint(0, 2, shape, generator=g, dtype=torch.uint8).bool()
        return cpu.to(device=device)
    return torch.randint(0, 2, shape, device=device, dtype=torch.uint8).bool()


def random_bit1(shape, device="cpu", pack_dtype=None, seed: int | None = None):
    """Random bit1 brute.Tensor of `shape`."""
    b = random_bool(shape, device=device, seed=seed)
    return bit1(b, pack_dtype=pack_dtype, device=device)


def random_shape(rng, ndim_max=4, dim_max=16, allow_empty=False) -> tuple[int, ...]:
    """Random tuple shape for property-based tests."""
    n = rng.randint(0, ndim_max)
    lo = 0 if allow_empty else 1
    return tuple(rng.randint(lo, dim_max) for _ in range(n))


# ---------------------------------------------------------------------------
# Hypothesis strategies (optional — only imported when hypothesis is present).
# ---------------------------------------------------------------------------

try:
    from hypothesis import strategies as st

    shape_strategy = st.lists(
        st.integers(min_value=0, max_value=32),
        min_size=0,
        max_size=4,
    ).map(tuple)

    nonempty_shape_strategy = st.lists(
        st.integers(min_value=1, max_value=32),
        min_size=1,
        max_size=4,
    ).map(tuple)

    bool_strategy = st.booleans()
    int_strategy = st.integers(min_value=-1000, max_value=1000)
    float_strategy = st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )

except ImportError:  # hypothesis is optional
    shape_strategy = None
    nonempty_shape_strategy = None
    bool_strategy = None
    int_strategy = None
    float_strategy = None


# ---------------------------------------------------------------------------
# View transforms (used in many parametric tests)
# ---------------------------------------------------------------------------

VIEW_TRANSFORMS_2D = [
    ("identity", lambda x: x),
    ("transpose", lambda x: x.t()),
    ("stride2_cols", lambda x: x[:, ::2]),
    ("stride2_rows", lambda x: x[::2]),
    ("permute", lambda x: x.permute(1, 0)),
    ("flip_rows", lambda x: x.flip(0)),
    ("flip_cols", lambda x: x.flip(1)),
]


VIEW_TRANSFORMS_3D = [
    ("identity", lambda x: x),
    ("permute_120", lambda x: x.permute(1, 2, 0)),
    ("permute_201", lambda x: x.permute(2, 0, 1)),
    ("transpose_01", lambda x: x.transpose(0, 1)),
    ("transpose_12", lambda x: x.transpose(1, 2)),
    ("stride_axis0", lambda x: x[::2]),
    ("stride_axis2", lambda x: x[:, :, ::2]),
    ("flip", lambda x: x.flip(0)),
]
