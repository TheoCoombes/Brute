"""Op parity tests, generated from `tests.ops_catalog`.

For each op spec × device, build a (bit1, bool) input pair at the spec's
parity shape, run the op on both, and assert the results are deep-equal.

This is the contract that proves brute.Tensor with dtype=bit1 is a drop-in
replacement for torch.bool: every method behaves identically.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests import ops_catalog
from tests.ops_catalog import default_cmp, parity_input_shapes


def _rand_bool(shape, device, *, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if not shape:
        return torch.tensor(bool(seed % 2), device=device)
    cpu = torch.randint(0, 2, shape, generator=g, dtype=torch.uint8).bool()
    return cpu.to(device=device)


def _build_inputs(spec: ops_catalog.OpSpec, device: str):
    """Build (bit1_args, bool_args) for the spec's parity shape."""
    shapes = parity_input_shapes(spec)
    bool_args = tuple(_rand_bool(s, device, seed=i + 1) for i, s in enumerate(shapes))
    bit1_args = tuple(
        brute.tensor(b, dtype=brute.bit1, device=device)
        for b in bool_args
    )
    return bit1_args, bool_args


def _run(spec: ops_catalog.OpSpec, args):
    return spec.run(*args)


def _oracle(spec: ops_catalog.OpSpec, bool_args):
    if spec.oracle is not None:
        return spec.oracle(*bool_args)
    return spec.run(*bool_args)


def _check_dtype(spec, bit1_out):
    """Optional structural check: result dtype is bit1 iff expected."""
    if not spec.expect_bit1_result:
        return
    if not isinstance(bit1_out, torch.Tensor):
        return
    is_bit1 = getattr(bit1_out, "_is_bit1", False)
    assert is_bit1 or bit1_out.dtype == torch.bool, (
        f"{spec.name}: expected bit1 (or bool) result, got dtype={bit1_out.dtype}"
    )


_PARAMS = list(ops_catalog.CATALOG)
_IDS = [s.name for s in _PARAMS]


@pytest.mark.parametrize("spec", _PARAMS, ids=_IDS)
def test_parity(spec, device):
    """For each (op, device): bit1 result ≡ bool result."""
    bit1_args, bool_args = _build_inputs(spec, device)

    try:
        bit1_out = _run(spec, bit1_args)
        bool_out = _oracle(spec, bool_args)
    except NotImplementedError as e:
        if "MPS" in str(e) or "not currently implemented" in str(e):
            pytest.skip(f"backend missing op: {e}")
        raise

    cmp = spec.cmp if spec.cmp is not None else default_cmp
    cmp(bit1_out, bool_out, note=f"op={spec.name}, dev={device}")

    _check_dtype(spec, bit1_out)


def test_catalog_is_non_empty():
    """Sanity check: the catalog wasn't accidentally cleared."""
    assert len(ops_catalog.CATALOG) > 50, \
        f"catalog seems thin: only {len(ops_catalog.CATALOG)} ops"


def test_catalog_names_unique():
    seen = set()
    for s in ops_catalog.CATALOG:
        assert s.name not in seen, f"duplicate op name: {s.name}"
        seen.add(s.name)


def test_catalog_groups_pair_bit1_and_bool():
    """Every group should be referenced by a single op spec (no duplicate
    groups — the report relies on group-name uniqueness for pairing)."""
    seen = set()
    for s in ops_catalog.CATALOG:
        assert s.group not in seen, f"duplicate benchmark group: {s.group}"
        seen.add(s.group)
