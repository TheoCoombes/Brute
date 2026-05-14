"""Edge-case parity, generated from `tests.ops_catalog`.

For each op spec × declared edge × device, build edge-shaped (bit1, bool)
inputs and assert the results match. Each op declares which edges apply
(unary ops skip "broadcast"; many view ops skip "0-dim", etc.).

The five edge classes:
  • empty        — one or more zero-sized dims
  • 0-dim        — scalar input
  • non-contig   — produced via .t() or [..., ::2]
  • broadcast    — mismatched binary shapes that broadcast
  • tail-misalign — last-dim not a multiple of pack_width (catches pad-bit bugs)
"""
from __future__ import annotations

import pytest
import torch

from tests import ops_catalog
from tests.ops_catalog import default_cmp, edge_inputs


# Build (spec, edge) pairs and their IDs.
def _matrix():
    rows = []
    for s in ops_catalog.CATALOG:
        for edge in sorted(s.edges):
            rows.append((s, edge))
    return rows


_PARAMS = _matrix()
_IDS = [f"{s.name}__{edge}" for s, edge in _PARAMS]


@pytest.mark.parametrize("spec,edge", _PARAMS, ids=_IDS)
def test_edge_parity(spec, edge, device):
    """For each (op, edge, device): the bit1 result matches the bool oracle."""
    bit1_args, bool_args = edge_inputs(spec, edge, device)
    if bit1_args is None:
        pytest.skip(f"{spec.name}: edge {edge!r} not applicable")

    try:
        bit1_out = spec.run(*bit1_args)
    except NotImplementedError as e:
        if "MPS" in str(e) or "not currently implemented" in str(e):
            pytest.skip(f"backend missing op: {e}")
        raise
    except (RuntimeError, IndexError) as e:
        # Some edges (e.g. 0-dim transpose, empty argmax) raise on torch.bool
        # too — that's a torch-level invariant. Verify the bool side also
        # raises and accept that as parity.
        bool_raised = None
        try:
            spec.run(*bool_args) if spec.oracle is None else spec.oracle(*bool_args)
        except Exception as e2:
            bool_raised = e2
        if bool_raised is not None:
            return  # both raised → consistent
        raise

    if spec.oracle is not None:
        bool_out = spec.oracle(*bool_args)
    else:
        bool_out = spec.run(*bool_args)

    cmp = spec.cmp if spec.cmp is not None else default_cmp
    cmp(bit1_out, bool_out, note=f"op={spec.name}, edge={edge}, dev={device}")
