"""Hypothesis-driven fuzz parity, generated from `tests.ops_catalog`.

Complements `test_edge_cases_catalog.py`: the latter walks a fixed set of
named edges; this driver generates random shapes / strides / pack widths
and exercises every op against random inputs.

Cheap by default (max_examples=20 per op). Bump `BRUTE_FUZZ_EXAMPLES` to
hammer harder when chasing a flake.
"""
from __future__ import annotations

import os

import pytest
import torch

import brute
from tests import ops_catalog
from tests.ops_catalog import default_cmp

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402


MAX_EXAMPLES = int(os.environ.get("BRUTE_FUZZ_EXAMPLES", "20"))


# Strategy for random shapes by scale.
def _shape_strategy(scale: str):
    if scale == "vector":
        return st.lists(
            st.integers(min_value=1, max_value=64),
            min_size=1, max_size=1,
        ).map(tuple)
    if scale == "matrix":
        return st.lists(
            st.integers(min_value=1, max_value=12),
            min_size=2, max_size=2,
        ).map(tuple)
    if scale == "matmul":
        # (M, K, N) with K aligned to 64 (smallest valid pack_width).
        return st.tuples(
            st.integers(min_value=1, max_value=8),
            st.sampled_from([64, 128, 192]),
            st.integers(min_value=1, max_value=8),
        )
    raise ValueError(scale)


def _pack_dtype_strategy():
    return st.sampled_from([torch.uint8, torch.uint32, torch.uint64])


def _rand_bool(shape, device):
    return torch.randint(0, 2, shape, dtype=torch.bool, device=device)


# Build the parametrized matrix outside the test (Hypothesis composes inside).
_SPECS = [s for s in ops_catalog.CATALOG]


@pytest.mark.fuzz
@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
@settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1),
       pack_dtype=_pack_dtype_strategy())
def test_fuzz_parity(spec, device, seed, pack_dtype):
    """Random shapes / values / pack widths must keep bit1 ≡ bool."""
    torch.manual_seed(seed)

    shape_strat = _shape_strategy(spec.scale)
    # Hypothesis doesn't let us depend on a strategy at parametrize time, so
    # we just use a fast deterministic shape from the seed:
    rng = torch.Generator(device="cpu").manual_seed(seed)

    if spec.scale == "vector":
        n = int(torch.randint(1, 65, (), generator=rng))
        shapes = [(n,)] * max(spec.arity, 1)
    elif spec.scale == "matrix":
        r = int(torch.randint(1, 13, (), generator=rng))
        c = int(torch.randint(1, 13, (), generator=rng))
        shapes = [(r, c)] * max(spec.arity, 1)
    else:  # matmul
        M = int(torch.randint(1, 9, (), generator=rng))
        N = int(torch.randint(1, 9, (), generator=rng))
        K = int(torch.tensor([64, 128, 192])[int(torch.randint(0, 3, (), generator=rng))])
        shapes = [(M, K), (N, K)][:max(spec.arity, 1)]

    bool_args = tuple(_rand_bool(s, device) for s in shapes)
    bit1_args = tuple(
        brute.tensor(b, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
        for b in bool_args
    )

    try:
        bit1_out = spec.run(*bit1_args)
    except NotImplementedError as e:
        if "MPS" in str(e) or "not currently implemented" in str(e):
            pytest.skip(f"backend missing op: {e}")
        raise
    except (RuntimeError, IndexError) as e:
        # If bool also raises on the same input, that's parity.
        bool_raised = None
        try:
            spec.run(*bool_args) if spec.oracle is None else spec.oracle(*bool_args)
        except Exception as e2:
            bool_raised = e2
        if bool_raised is not None:
            return
        raise

    if spec.oracle is not None:
        bool_out = spec.oracle(*bool_args)
    else:
        bool_out = spec.run(*bool_args)

    cmp = spec.cmp if spec.cmp is not None else default_cmp
    cmp(bit1_out, bool_out,
        note=f"op={spec.name}, dev={device}, pd={pack_dtype}, seed={seed}")
