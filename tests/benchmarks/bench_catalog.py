"""Paired bit1 / bool benchmarks, generated from `tests.ops_catalog`.

For every op spec × device × scale (dispatch/medium/huge), we emit *two*
benchmarks sharing the same `benchmark.group`:

  test_bit1_<op>[<device>-<scale>]
  test_bool_<op>[<device>-<scale>]

`make_report.py` keys on the matching group name and computes the speedup
(bool / bit1). When `spec.bool_cant_run` is True (matmul, popcount, hamming,
unpack_pm1, argmax, argmin) we benchmark `spec.oracle` on the bool side — i.e.
"what a bool user would write" — so the report still pairs cleanly.

Pack-width sweep: bitwise/logical/brute/matmul ops are timed at all three
pack widths (uint8/uint32/uint64); other categories are timed once at the
default uint8 (their throughput is pack-invariant by construction).
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests import ops_catalog
from tests.benchmarks._helpers import (
    VECTOR_SCALES, MATRIX_SCALES, MATMUL_SCALES,
    with_sync, set_throughput, scale_size,
)


# Same pack-sensitive set used by the parity driver.
_PACK_SENSITIVE = frozenset({"bitwise", "logical", "brute", "matmul"})

_PACK_DTYPES = [
    pytest.param(torch.uint8,  id="pw_08"),
    pytest.param(torch.uint32, id="pw_32"),
    pytest.param(torch.uint64, id="pw_64"),
]


def _scales_for(scale: str):
    if scale == "vector":
        return VECTOR_SCALES
    if scale == "matrix":
        return MATRIX_SCALES
    if scale == "matmul":
        return MATMUL_SCALES
    raise ValueError(scale)


def _input_shapes(scale: str, size_param):
    """Return one shape per tensor input, given (scale, size_param)."""
    if scale == "vector":
        return [(int(size_param),)]
    if scale == "matrix":
        return [tuple(size_param)]
    if scale == "matmul":
        M, K, N = size_param
        return [(M, K), (N, K)]
    raise ValueError(scale)


def _rand_bool(shape, device, *, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cpu = torch.randint(0, 2, shape, generator=g, dtype=torch.uint8).bool()
    return cpu.to(device=device)


def _build_pair(spec: ops_catalog.OpSpec, size_param, device, pack_dtype):
    """Build (bit1_args, bool_args) for the given scale & pack width."""
    shapes = _input_shapes(spec.scale, size_param)
    arity = max(spec.arity, 1)
    bool_args = tuple(_rand_bool(shapes[i % len(shapes)], device, seed=i + 1)
                      for i in range(arity))
    bit1_args = tuple(
        brute.tensor(b, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
        for b in bool_args
    )
    return bit1_args, bool_args


def _throughput_elems(spec, size_param):
    """How many logical bits / elements / bit-ops the run touches."""
    if spec.scale == "matmul":
        M, K, N = size_param
        if spec.name.startswith("matmul"):
            return 2 * M * K * N
        return max(M, N) * K
    return scale_size(size_param)


# Build the parametrize matrix.
def _matrix():
    rows = []
    for s in ops_catalog.CATALOG:
        scales = _scales_for(s.scale)
        if s.category in _PACK_SENSITIVE:
            for pd in _PACK_DTYPES:
                for sc in scales:
                    rows.append((s, sc, pd))
        else:
            for sc in scales:
                rows.append((s, sc, _PACK_DTYPES[0]))  # default pw_08
    return rows


_PARAMS = _matrix()


def _id_for(spec, size_param_obj, pack_param):
    """Compose a pytest parameter id."""
    # size_param_obj is a pytest.param wrapping the size; its id is in .id
    sid = size_param_obj.id if hasattr(size_param_obj, "id") else str(size_param_obj)
    if spec.category in _PACK_SENSITIVE:
        return f"{sid}-{pack_param.id}"
    return sid


# pytest can't natively parametrize across heterogeneous shape tuples in one
# go without breaking the (device, scale, pack) cross-product; build the two
# test functions manually, one per (bit1, bool) flavour, sharing the matrix.

def _params_for(category_filter: str | None = None):
    """Yield (spec, size_value, size_id, pack_dtype, pack_id) tuples."""
    for spec, size_param, pack_param in _PARAMS:
        if category_filter and spec.category != category_filter:
            continue
        sv = size_param.values[0] if hasattr(size_param, "values") else size_param
        sid = size_param.id if hasattr(size_param, "id") else str(sv)
        pdv = pack_param.values[0] if hasattr(pack_param, "values") else pack_param
        pdid = pack_param.id if hasattr(pack_param, "id") else str(pdv)
        yield spec, sv, sid, pdv, pdid


_ALL_PARAMS = list(_params_for())


def _pytest_param(spec, size_value, size_id, pack_dtype, pack_id):
    suffix = f"{size_id}-{pack_id}" if spec.category in _PACK_SENSITIVE else size_id
    pid = f"{spec.name}-{suffix}"
    return pytest.param(spec, size_value, pack_dtype, id=pid)


_PARAMETRIZED = [_pytest_param(*p) for p in _ALL_PARAMS]


@pytest.mark.parametrize("spec,size,pack_dtype", _PARAMETRIZED)
def test_bit1(benchmark, device, spec, size, pack_dtype):
    """Time `spec.run(*bit1_inputs)` on `device`."""
    benchmark.group = spec.group
    bit1_args, _ = _build_pair(spec, size, device, pack_dtype)
    set_throughput(benchmark, _throughput_elems(spec, size), "bits")
    try:
        benchmark(with_sync(lambda: spec.run(*bit1_args), device))
    except NotImplementedError as e:
        if "MPS" in str(e) or "not currently implemented" in str(e):
            pytest.skip(f"backend missing op: {e}")
        raise


@pytest.mark.parametrize("spec,size,pack_dtype", _PARAMETRIZED)
def test_bool(benchmark, device, spec, size, pack_dtype):
    """Time the bool oracle on `device`.

    When `spec.bool_cant_run` is True, the oracle is "what a bool user would
    write" (e.g. `b.long().sum()` for popcount). Otherwise the oracle is just
    `spec.run(*bool_inputs)`.
    """
    benchmark.group = spec.group
    _, bool_args = _build_pair(spec, size, device, pack_dtype)
    fn = spec.oracle if spec.oracle is not None else spec.run
    set_throughput(benchmark, _throughput_elems(spec, size), "bits")
    try:
        benchmark(with_sync(lambda: fn(*bool_args), device))
    except NotImplementedError as e:
        if "MPS" in str(e) or "not currently implemented" in str(e):
            pytest.skip(f"backend missing op: {e}")
        raise
