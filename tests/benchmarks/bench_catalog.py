"""Paired bit1 / bool benchmarks, generated from `tests.ops_catalog`.

For every op spec × device × scale (dispatch/medium/huge), we emit *two*
benchmarks sharing the same `benchmark.group`:

  test_bit1_<op>[<device>-<scale>]
  test_bool_<op>[<device>-<scale>]

`make_report.py` keys on the matching group name and computes the speedup
(bool / bit1). When `spec.bool_cant_run` is True (matmul, popcount, hamming,
unpack_pm1, argmax, argmin) we benchmark `spec.oracle` on the bool side — i.e.
"what a bool user would write" — so the report still pairs cleanly.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests import ops_catalog
from tests.benchmarks._helpers import (
    VECTOR_SCALES, MATRIX_SCALES, MATMUL_SCALES, DIAG_VECTOR_SCALES,
    with_sync, set_throughput, scale_size,
)


def _scales_for(scale: str):
    if scale == "vector":
        return VECTOR_SCALES
    if scale == "matrix":
        return MATRIX_SCALES
    if scale == "matmul":
        return MATMUL_SCALES
    if scale == "diag_vector":
        return DIAG_VECTOR_SCALES
    raise ValueError(scale)


def _input_shapes(scale: str, size_param):
    """Return one shape per tensor input, given (scale, size_param)."""
    if scale in ("vector", "diag_vector"):
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


def _build_pair(spec: ops_catalog.OpSpec, size_param, device):
    """Build (bit1_args, bool_args) for the given scale."""
    shapes = _input_shapes(spec.scale, size_param)
    arity = max(spec.arity, 1)
    bool_args = tuple(_rand_bool(shapes[i % len(shapes)], device, seed=i + 1)
                      for i in range(arity))
    bit1_args = tuple(
        brute.tensor(b, dtype=brute.bit1, device=device)
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
        for sc in _scales_for(s.scale):
            rows.append((s, sc))
    return rows


_PARAMS = _matrix()


def _params_for():
    """Yield (spec, size_value, size_id) tuples."""
    for spec, size_param in _PARAMS:
        sv = size_param.values[0] if hasattr(size_param, "values") else size_param
        sid = size_param.id if hasattr(size_param, "id") else str(sv)
        yield spec, sv, sid


_ALL_PARAMS = list(_params_for())


def _pytest_param(spec, size_value, size_id):
    pid = f"{spec.name}-{size_id}"
    return pytest.param(spec, size_value, id=pid)


_PARAMETRIZED = [_pytest_param(*p) for p in _ALL_PARAMS]


@pytest.mark.parametrize("spec,size", _PARAMETRIZED)
def test_bit1(benchmark, device, spec, size):
    """Time `spec.run(*bit1_inputs)` on `device`."""
    benchmark.group = spec.group
    bit1_args, _ = _build_pair(spec, size, device)
    set_throughput(benchmark, _throughput_elems(spec, size), "bits")
    try:
        benchmark(with_sync(lambda: spec.run(*bit1_args), device))
    except NotImplementedError as e:
        if "MPS" in str(e) or "not currently implemented" in str(e):
            pytest.skip(f"backend missing op: {e}")
        raise


@pytest.mark.parametrize("spec,size", _PARAMETRIZED)
def test_bool(benchmark, device, spec, size):
    """Time the bool oracle on `device`.

    When `spec.bool_cant_run` is True, the oracle is "what a bool user would
    write" (e.g. `b.long().sum()` for popcount). Otherwise the oracle is just
    `spec.run(*bool_inputs)`.
    """
    benchmark.group = spec.group
    _, bool_args = _build_pair(spec, size, device)
    fn = spec.oracle if spec.oracle is not None else spec.run
    set_throughput(benchmark, _throughput_elems(spec, size), "bits")
    try:
        benchmark(with_sync(lambda: fn(*bool_args), device))
    except NotImplementedError as e:
        if "MPS" in str(e) or "not currently implemented" in str(e):
            pytest.skip(f"backend missing op: {e}")
        raise
