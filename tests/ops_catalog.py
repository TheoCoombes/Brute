"""Single source of truth for every torch op the bit1 tensor needs to support.

Each `OpSpec` declares:
  • how to build inputs for a given shape / device
  • the operation (a callable that runs on either a bit1 *or* bool tensor)
  • how to compare the bit1 and bool results (most ops use the default;
    matmul / dot / mv etc. need custom oracles because torch.bool can't run them)
  • which edge variants make sense (empty, 0-dim, non-contig, broadcast,
    tail-misalignment)

Three drivers consume the catalog:
  • tests/unit/test_parity_catalog.py     — asserts 1:1 parity with torch.bool
  • tests/unit/test_edge_cases_catalog.py — asserts parity on edge inputs
  • tests/benchmarks/bench_catalog.py     — paired bit1 / bool benchmarks

Conventions
-----------
- `run(x, y=None)` is what the test/bench calls. It must accept brute.Tensor
  AND torch.Tensor — the same callable runs both code paths.
- `oracle(x_bool, y_bool=None)` is optional. If present, the parity test
  compares `run(bit1)` against `oracle(bool)` rather than `run(bool)`. Use
  this when torch.bool can't execute the op natively (e.g. mm, matmul, dot).
- `cmp(a, b)` is optional. Defaults to a generic deep-equality that handles
  tensors, bit1 tensors, tuples, and Python scalars. Override only when a
  bespoke tolerance is needed.
- `scale` selects the size matrix:
    "vector" → 1-D flat buffer
    "matrix" → 2-D rectangle
    "matmul" → (M, K, N) triple, K aligned to 64
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch


# ─────────────────────────────────────────────────────────────────────────────
# OpSpec
# ─────────────────────────────────────────────────────────────────────────────


# All edge variants the framework knows how to construct.
ALL_EDGES = frozenset([
    "empty",          # one or more zero-sized dims
    "0-dim",          # scalar
    "non-contig",     # produced via .t() or [..., ::2]
    "broadcast",      # mismatched shapes that broadcast (binary ops only)
    "tail-misalign",  # last-dim is not a multiple of 64 (e.g. 13)
])


@dataclass(frozen=True)
class OpSpec:
    """One operation, described once, consumed everywhere."""

    name: str                            # snake_case identifier, unique
    group: str                           # pytest-benchmark group ("bitwise/and")
    category: str                        # high-level bucket ("bitwise", ...)
    arity: int                           # # of tensor inputs (0/1/2)
    scale: str                           # "vector" | "matrix" | "matmul"
    run: Callable                        # the op itself

    # Optional bool oracle. If None, oracle is `run(*bool_args)`.
    oracle: Optional[Callable] = None

    # Comparator: (bit1_result, bool_result) -> None. Raises on mismatch.
    cmp: Optional[Callable] = None

    # Edge cases this op should be tested under (subset of ALL_EDGES).
    edges: frozenset = field(default_factory=lambda: frozenset([
        "empty", "0-dim", "non-contig",
    ]))

    # If True, bool can't execute this natively → oracle is required.
    bool_cant_run: bool = False

    # If True, expect `run(bit1)` to return a bit1-dtype result. False for ops
    # that return non-bool (sum→int, nonzero→long, etc.).
    expect_bit1_result: bool = True

    # Free-form notes — surfaces in assert messages.
    note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Standard input factories — shapes by scale, edge by name
# ─────────────────────────────────────────────────────────────────────────────


def _rand_bool(shape, device, *, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if not shape or any(d == 0 for d in shape):
        return torch.zeros(shape, dtype=torch.bool, device=device)
    cpu = torch.randint(0, 2, shape, generator=g, dtype=torch.uint8).bool()
    return cpu.to(device=device)


def parity_shape(scale: str) -> tuple[int, ...]:
    """A small, multi-dim shape used by the parity tests (NOT the benchmark
    scales — those exist only to stress timing).

    For "matmul" the returned shape is (M, K, N); the input builder splits
    that into a pair of (M, K) and (N, K) tensors.
    """
    if scale in ("vector", "diag_vector"):
        return (16,)
    if scale == "matrix":
        return (4, 8)
    if scale == "matmul":
        return (4, 64, 4)
    raise ValueError(scale)


def parity_input_shapes(spec: "OpSpec") -> list[tuple[int, ...]]:
    """Return one shape per tensor input for the parity test, derived from the
    op's scale. Matmul splits (M,K,N) into [(M,K), (N,K)].
    """
    shape = parity_shape(spec.scale)
    arity = max(spec.arity, 1)
    if spec.scale == "matmul":
        M, K, N = shape
        return [(M, K), (N, K)][:arity]
    return [shape] * arity


def edge_inputs(spec: "OpSpec", edge: str, device: str):
    """Build (bit1_inputs, bool_inputs) for a given edge variant.

    Each is a tuple of tensors matching `spec.arity`. Returns (None, None) if
    the edge isn't applicable to this op.
    """
    from brute import tensor as _brute_tensor, bit1 as _bit1_dtype

    def _bit1(t):
        return _brute_tensor(t, dtype=_bit1_dtype, device=device)

    arity = max(spec.arity, 1)

    if edge == "empty":
        if spec.scale in ("vector", "diag_vector"):
            shapes = [(0,)] * arity
        elif spec.scale == "matrix":
            shapes = [(0, 8)] * arity
        else:  # matmul: empty K
            # (M, K)=(4,0); (N, K)=(4,0)
            shapes = [(4, 0)] * arity
        srcs = [_rand_bool(shapes[i], device, seed=10 + i) for i in range(arity)]
        return tuple(_bit1(s) for s in srcs), tuple(srcs)

    if edge == "0-dim":
        srcs = [_rand_bool((), device, seed=20 + i) for i in range(arity)]
        return tuple(_bit1(s) for s in srcs), tuple(srcs)

    if edge == "non-contig":
        if spec.scale in ("vector", "diag_vector"):
            srcs = [_rand_bool((32,), device, seed=30 + i)[::2] for i in range(arity)]
        else:
            srcs = [_rand_bool((8, 4), device, seed=30 + i).t() for i in range(arity)]
        return tuple(_bit1(s) for s in srcs), tuple(srcs)

    if edge == "broadcast":
        if spec.arity < 2:
            return None, None
        if spec.scale in ("vector", "diag_vector"):
            return None, None
        a_src = _rand_bool((4, 8), device, seed=40)
        b_src = _rand_bool((1, 8), device, seed=41)
        return (_bit1(a_src), _bit1(b_src)), (a_src, b_src)

    if edge == "tail-misalign":
        shape = (3, 13) if spec.scale not in ("vector", "diag_vector") else (13,)
        srcs = [_rand_bool(shape, device, seed=50 + i) for i in range(arity)]
        return tuple(_bit1(s) for s in srcs), tuple(srcs)

    raise ValueError(f"unknown edge: {edge}")


# ─────────────────────────────────────────────────────────────────────────────
# Generic comparator
# ─────────────────────────────────────────────────────────────────────────────


def _materialize(x):
    """Return a plain torch.Tensor whose values are guaranteed up-to-date.

    For a bit1 brute.Tensor with a lazy/dirty bool buffer, `.as_subclass(Tensor)`
    reads stale bytes; we must go via `.bool()` to force a packed→bool sync.
    For everything else (regular brute.Tensor wrappers, plain torch.Tensor),
    `.as_subclass(Tensor)` is fine.
    """
    if not isinstance(x, torch.Tensor):
        return x
    try:
        import brute
        if isinstance(x, brute.Tensor):
            if getattr(x, "_is_bit1", False):
                return x.bool().as_subclass(torch.Tensor)
            return x.as_subclass(torch.Tensor)
    except ImportError:
        pass
    return x


# Backwards-compat alias used inside the catalog.
_to_plain = _materialize


def default_cmp(a, b, *, note: str = ""):
    """Deep equality. Handles tensors (incl. bit1), tuples, lists, scalars."""
    if isinstance(a, tuple) or isinstance(b, tuple):
        assert isinstance(a, tuple) and isinstance(b, tuple), \
            f"shape mismatch: tuple vs non-tuple ({note})"
        assert len(a) == len(b), f"tuple length mismatch ({note})"
        for ai, bi in zip(a, b):
            default_cmp(ai, bi, note=note)
        return
    if isinstance(a, list) or isinstance(b, list):
        assert a == b, f"list mismatch ({note})"
        return
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ap = _materialize(a)
        bp = _materialize(b)
        if ap.dtype != bp.dtype:
            if ap.dtype == torch.bool or bp.dtype == torch.bool:
                int_side = bp if ap.dtype == torch.bool else ap
                if int_side.numel() > 0:
                    coercible = bool(torch.all((int_side == 0) | (int_side == 1)))
                    if not coercible:
                        raise AssertionError(
                            f"dtype mismatch and values aren't bool-coercible: "
                            f"{ap.dtype} vs {bp.dtype} ({note})"
                        )
                ap = ap.bool() if ap.dtype != torch.bool else ap
                bp = bp.bool() if bp.dtype != torch.bool else bp
            else:
                ap, bp = ap.long(), bp.long()
        assert ap.shape == bp.shape, \
            f"shape mismatch: {ap.shape} vs {bp.shape} ({note})"
        assert torch.equal(ap.cpu(), bp.cpu()), \
            f"value mismatch ({note}): bit1={ap.cpu().tolist()[:32]}{'...' if ap.numel()>32 else ''} vs bool={bp.cpu().tolist()[:32]}{'...' if bp.numel()>32 else ''}"
        return
    assert type(a) is type(b) or (isinstance(a, (int, bool)) and isinstance(b, (int, bool))), \
        f"type mismatch: {type(a).__name__} vs {type(b).__name__} ({note})"
    assert a == b, f"scalar mismatch: {a} vs {b} ({note})"


# ─────────────────────────────────────────────────────────────────────────────
# Op catalog
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_BINARY_EDGES = frozenset(["empty", "0-dim", "non-contig", "broadcast", "tail-misalign"])
_DEFAULT_UNARY_EDGES  = frozenset(["empty", "0-dim", "non-contig", "tail-misalign"])


CATALOG: list[OpSpec] = []


def _add(spec: OpSpec) -> None:
    CATALOG.append(spec)


# ── BITWISE ─────────────────────────────────────────────────────────────────

_add(OpSpec("bitwise_and",     "bitwise/and",     "bitwise", 2, "vector", lambda a, b: a & b,                  edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_or",      "bitwise/or",      "bitwise", 2, "vector", lambda a, b: a | b,                  edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_xor",     "bitwise/xor",     "bitwise", 2, "vector", lambda a, b: a ^ b,                  edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_not",     "bitwise/not",     "bitwise", 1, "vector", lambda a: ~a,                        edges=_DEFAULT_UNARY_EDGES))
_add(OpSpec("bitwise_and_fn",  "bitwise/and_fn",  "bitwise", 2, "vector", lambda a, b: torch.bitwise_and(a, b),edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_or_fn",   "bitwise/or_fn",   "bitwise", 2, "vector", lambda a, b: torch.bitwise_or(a, b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_xor_fn",  "bitwise/xor_fn",  "bitwise", 2, "vector", lambda a, b: torch.bitwise_xor(a, b),edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_not_fn",  "bitwise/not_fn",  "bitwise", 1, "vector", lambda a: torch.bitwise_not(a),       edges=_DEFAULT_UNARY_EDGES))

# ── LOGICAL ─────────────────────────────────────────────────────────────────

_add(OpSpec("logical_and",     "logical/and",     "logical", 2, "vector", lambda a, b: torch.logical_and(a, b),edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("logical_or",      "logical/or",      "logical", 2, "vector", lambda a, b: torch.logical_or(a, b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("logical_xor",     "logical/xor",     "logical", 2, "vector", lambda a, b: torch.logical_xor(a, b),edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("logical_not",     "logical/not",     "logical", 1, "vector", lambda a: torch.logical_not(a),      edges=_DEFAULT_UNARY_EDGES))


# ── IN-PLACE BITWISE ────────────────────────────────────────────────────────

def _iand(a, b):
    a = a.clone();  a &= b;  return a
def _ior(a, b):
    a = a.clone();  a |= b;  return a
def _ixor(a, b):
    a = a.clone();  a ^= b;  return a

_add(OpSpec("iand", "bitwise/iand", "bitwise", 2, "vector", _iand, edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ior",  "bitwise/ior",  "bitwise", 2, "vector", _ior,  edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ixor", "bitwise/ixor", "bitwise", 2, "vector", _ixor, edges=_DEFAULT_BINARY_EDGES))


# ── COMPARISON ──────────────────────────────────────────────────────────────

_add(OpSpec("eq",        "compare/eq",      "compare", 2, "vector", lambda a, b: a == b,           edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ne",        "compare/ne",      "compare", 2, "vector", lambda a, b: a != b,           edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("lt",        "compare/lt",      "compare", 2, "vector", lambda a, b: a < b,            edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("le",        "compare/le",      "compare", 2, "vector", lambda a, b: a <= b,           edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("gt",        "compare/gt",      "compare", 2, "vector", lambda a, b: a > b,            edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ge",        "compare/ge",      "compare", 2, "vector", lambda a, b: a >= b,           edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("eq_fn",     "compare/eq_fn",   "compare", 2, "vector", lambda a, b: torch.eq(a, b),   edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ne_fn",     "compare/ne_fn",   "compare", 2, "vector", lambda a, b: torch.ne(a, b),   edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("equal",     "compare/equal",   "compare", 2, "vector",
            run=lambda a, b: torch.equal(a, b),
            edges=frozenset(["empty"]),
            expect_bit1_result=False))


# ── REDUCTIONS (full) ───────────────────────────────────────────────────────

_add(OpSpec("sum_full",       "reduce/sum_full",       "reduce", 1, "vector",
            run=lambda a: torch.sum(a),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("count_nonzero",  "reduce/count_nonzero",  "reduce", 1, "vector",
            run=lambda a: torch.count_nonzero(a),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("all_full",       "reduce/all_full",       "reduce", 1, "vector",
            run=lambda a: torch.all(a),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("any_full",       "reduce/any_full",       "reduce", 1, "vector",
            run=lambda a: torch.any(a),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("argmax_full",    "reduce/argmax_full",    "reduce", 1, "vector",
            run=lambda a: torch.argmax(a),
            oracle=lambda b: torch.argmax(b.long()),
            bool_cant_run=True,
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("argmin_full",    "reduce/argmin_full",    "reduce", 1, "vector",
            run=lambda a: torch.argmin(a),
            oracle=lambda b: torch.argmin(b.long()),
            bool_cant_run=True,
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("max_full",       "reduce/max_full",       "reduce", 1, "vector",
            run=lambda a: torch.max(a),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("min_full",       "reduce/min_full",       "reduce", 1, "vector",
            run=lambda a: torch.min(a),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("nonzero",        "reduce/nonzero",        "reduce", 1, "vector",
            run=lambda a: torch.nonzero(a),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("argwhere",       "reduce/argwhere",       "reduce", 1, "vector",
            run=lambda a: torch.argwhere(a),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))


# ── REDUCTIONS (dim) ────────────────────────────────────────────────────────

_add(OpSpec("sum_dim0",      "reduce_dim/sum_dim0",      "reduce_dim", 1, "matrix",
            run=lambda a: torch.sum(a, dim=0),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("sum_dim1",      "reduce_dim/sum_dim1",      "reduce_dim", 1, "matrix",
            run=lambda a: torch.sum(a, dim=1),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("all_dim0",      "reduce_dim/all_dim0",      "reduce_dim", 1, "matrix",
            run=lambda a: torch.all(a, dim=0),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("any_dim_last",  "reduce_dim/any_dim_last",  "reduce_dim", 1, "matrix",
            run=lambda a: torch.any(a, dim=-1),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("count_nonzero_dim0", "reduce_dim/count_nonzero_dim0", "reduce_dim", 1, "matrix",
            run=lambda a: torch.count_nonzero(a, dim=0),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("cumsum_dim_last", "reduce_dim/cumsum_dim_last", "reduce_dim", 1, "matrix",
            run=lambda a: torch.cumsum(a, dim=-1),
            edges=frozenset(["non-contig"]),
            expect_bit1_result=False))


# ── SHAPE / VIEW ────────────────────────────────────────────────────────────

_add(OpSpec("view_flat",       "view/view_flat",      "view", 1, "matrix",
            run=lambda a: a.contiguous().view(-1),
            edges=frozenset(["tail-misalign"])))
_add(OpSpec("reshape_flat",    "view/reshape_flat",   "view", 1, "matrix",
            run=lambda a: a.reshape(-1),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("flatten",         "view/flatten",        "view", 1, "matrix",
            run=lambda a: torch.flatten(a),
            edges=frozenset(["non-contig", "tail-misalign"])))
# NOTE: bit1.squeeze() deliberately preserves the packed (last) axis even if
# size-1, because the packed buffer's trailing word can't be erased without
# losing alignment. We squeeze a non-packed leading 1-dim to stay within the
# parity envelope.
_add(OpSpec("squeeze",         "view/squeeze",        "view", 1, "matrix",
            run=lambda a: a.unsqueeze(0).squeeze(0),
            edges=frozenset(["non-contig"])))
_add(OpSpec("unsqueeze_0",     "view/unsqueeze_0",    "view", 1, "matrix",
            run=lambda a: a.unsqueeze(0),
            edges=frozenset(["non-contig"])))
_add(OpSpec("unsqueeze_last",  "view/unsqueeze_last", "view", 1, "matrix",
            run=lambda a: a.unsqueeze(-1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("t",               "view/t",              "view", 1, "matrix",
            run=lambda a: a.t(),
            edges=frozenset(["non-contig"])))
_add(OpSpec("transpose_01",    "view/transpose_01",   "view", 1, "matrix",
            run=lambda a: a.transpose(0, 1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("permute",         "view/permute",        "view", 1, "matrix",
            run=lambda a: a.unsqueeze(0).permute(0, 2, 1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("movedim",         "view/movedim",        "view", 1, "matrix",
            run=lambda a: torch.movedim(a, 0, -1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("swapaxes",        "view/swapaxes",       "view", 1, "matrix",
            run=lambda a: torch.swapaxes(a, 0, 1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("contiguous",      "view/contiguous",     "view", 1, "matrix",
            run=lambda a: a.contiguous(),
            edges=frozenset(["non-contig", "tail-misalign"])))


def _expand_run(a):
    if a.dim() >= 2 and a.shape[0] > 0 and a.shape[1] > 0:
        return a[:1].expand(a.shape[0], a.shape[1]).contiguous()
    return a


_add(OpSpec("expand", "view/expand", "view", 1, "matrix", run=_expand_run, edges=frozenset([])))


def _broadcast_to_run(a):
    if a.dim() >= 2 and a.shape[0] > 0:
        return torch.broadcast_to(a[:1], a.shape).contiguous()
    return a


_add(OpSpec("broadcast_to", "view/broadcast_to", "view", 1, "matrix", run=_broadcast_to_run, edges=frozenset([])))


def _narrow_run(a):
    if a.shape[0] > 1:
        return a.narrow(0, 0, a.shape[0] // 2)
    return a


_add(OpSpec("narrow", "view/narrow", "view", 1, "matrix", run=_narrow_run, edges=frozenset(["non-contig"])))


def _select_run(a):
    if a.shape[0] > 0:
        return a.select(0, 0)
    return a


_add(OpSpec("select", "view/select", "view", 1, "matrix", run=_select_run, edges=frozenset(["non-contig"])))

_add(OpSpec("atleast_2d", "view/atleast_2d", "view", 1, "vector",
            run=lambda a: torch.atleast_2d(a),
            edges=frozenset(["empty", "0-dim"])))


# ── COMBINING ───────────────────────────────────────────────────────────────

_add(OpSpec("cat_dim0",   "combine/cat_dim0",  "combine", 2, "matrix",
            run=lambda a, b: torch.cat([a, b], dim=0),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("stack_dim0", "combine/stack_dim0","combine", 2, "matrix",
            run=lambda a, b: torch.stack([a, b], dim=0),
            edges=frozenset(["non-contig", "tail-misalign"])))


def _split_eq(a):
    if a.shape[0] < 2:
        return (a,)
    return torch.split(a, max(1, a.shape[0] // 2), dim=0)

def _chunk_eq(a):
    if a.shape[0] < 2:
        return (a,)
    return torch.chunk(a, 2, dim=0)

def _unbind_eq(a):
    return torch.unbind(a, dim=0)


_add(OpSpec("split",  "combine/split",  "combine", 1, "matrix", run=_split_eq,  edges=frozenset(["non-contig"])))
_add(OpSpec("chunk",  "combine/chunk",  "combine", 1, "matrix", run=_chunk_eq,  edges=frozenset(["non-contig"])))
_add(OpSpec("unbind", "combine/unbind", "combine", 1, "matrix", run=_unbind_eq, edges=frozenset(["non-contig"])))


# ── INDEXING ────────────────────────────────────────────────────────────────

def _getitem_row(a):     return a[0]                    if a.shape[0] > 0 else a
def _getitem_slice(a):   return a[:max(1, a.shape[0] // 2)] if a.shape[0] > 0 else a
def _getitem_col(a):
    if a.dim() < 2 or a.shape[-1] == 0:
        return a
    return a[:, 0]
def _getitem_neg(a):     return a[-1] if a.shape[0] > 0 else a
def _getitem_ellipsis(a):
    if a.dim() < 2 or a.shape[-1] == 0:
        return a
    return a[..., 0]


_add(OpSpec("getitem_row",       "index/getitem_row",     "index", 1, "matrix", run=_getitem_row,     edges=frozenset(["non-contig"])))
_add(OpSpec("getitem_slice",     "index/getitem_slice",   "index", 1, "matrix", run=_getitem_slice,   edges=frozenset(["non-contig"])))
_add(OpSpec("getitem_col",       "index/getitem_col",     "index", 1, "matrix", run=_getitem_col,     edges=frozenset(["non-contig"])))
_add(OpSpec("getitem_neg",       "index/getitem_neg",     "index", 1, "matrix", run=_getitem_neg,     edges=frozenset([])))
_add(OpSpec("getitem_ellipsis",  "index/getitem_ellipsis","index", 1, "matrix", run=_getitem_ellipsis,edges=frozenset([])))


def _gather_run(a):
    if a.dim() < 2 or a.shape[1] == 0:
        return a
    idx = torch.zeros((a.shape[0], 1), dtype=torch.long, device=a.device)
    return torch.gather(a, 1, idx)


def _index_select_run(a):
    if a.shape[0] == 0:
        return a
    idx = torch.tensor([0], device=a.device)
    return torch.index_select(a, 0, idx)


def _take_run(a):
    if a.numel() == 0:
        return a
    idx = torch.tensor([0, a.numel() - 1], device=a.device)
    return torch.take(a, idx)


def _scatter_run(a):
    if a.dim() < 2 or a.shape[1] == 0:
        return a.clone()
    idx = torch.zeros((a.shape[0], 1), dtype=torch.long, device=a.device)
    src = a[:, :1]
    return a.scatter(1, idx, src)


def _build_bool_mask(a):
    """Build a deterministic bool mask matching `a.shape` via plain torch
    (avoids brute's bit1 zeros_like override which returns a bit1 tensor).
    """
    flat = torch.zeros(a.numel(), dtype=torch.bool, device=a.device)
    if a.numel() > 0:
        flat[::2] = True
    return flat.view(a.shape)


def _masked_select_run(a):
    mask = _build_bool_mask(a)
    return torch.masked_select(a, mask)


def _masked_fill_run(a):
    a = a.clone()
    mask = torch.zeros(a.numel(), dtype=torch.bool, device=a.device)
    if a.numel() > 0:
        mask[0] = True
    return a.masked_fill(mask.view(a.shape), True)


_add(OpSpec("gather",        "index/gather",        "index", 1, "matrix", run=_gather_run,        edges=frozenset(["non-contig"])))
_add(OpSpec("index_select",  "index/index_select",  "index", 1, "matrix", run=_index_select_run,  edges=frozenset(["non-contig"])))
_add(OpSpec("take",          "index/take",          "index", 1, "matrix", run=_take_run,          edges=frozenset(["non-contig"])))
_add(OpSpec("scatter",       "index/scatter",       "index", 1, "matrix", run=_scatter_run,       edges=frozenset(["non-contig"])))
_add(OpSpec("masked_select", "index/masked_select", "index", 1, "matrix", run=_masked_select_run, edges=frozenset(["non-contig"])))
_add(OpSpec("masked_fill",   "index/masked_fill",   "index", 1, "matrix", run=_masked_fill_run,   edges=frozenset(["non-contig", "tail-misalign"])))


def _where_run(a):
    # Build constant value tensors via plain torch to avoid brute's bit1
    # promotion (bit1 + bit1 + bit1 in where would itself be tested, not what
    # we want here — we want to test bit1 as the *condition*).
    t = torch.ones(a.shape, dtype=torch.bool, device=a.device)
    f = torch.zeros(a.shape, dtype=torch.bool, device=a.device)
    return torch.where(a, t, f)


_add(OpSpec("where", "index/where", "index", 1, "matrix", run=_where_run,
            edges=frozenset(["non-contig"])))


# ── CLONE / DETACH / FILL ───────────────────────────────────────────────────

_add(OpSpec("clone",  "clone/clone",  "clone", 1, "matrix",
            run=lambda a: a.clone(),
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("detach", "clone/detach", "clone", 1, "matrix",
            run=lambda a: a.detach(),
            edges=frozenset(["non-contig"])))


def _fill_true(a):
    a = a.clone();  a.fill_(True);  return a

def _fill_false(a):
    a = a.clone();  a.fill_(False);  return a

def _zero_(a):
    a = a.clone();  a.zero_();  return a


_add(OpSpec("fill_true",  "fill/fill_true",  "fill", 1, "matrix", run=_fill_true,  edges=frozenset(["tail-misalign"])))
_add(OpSpec("fill_false", "fill/fill_false", "fill", 1, "matrix", run=_fill_false, edges=frozenset(["tail-misalign"])))
_add(OpSpec("zero_",      "fill/zero_",      "fill", 1, "matrix", run=_zero_,      edges=frozenset(["tail-misalign"])))


# ── REARRANGEMENT ───────────────────────────────────────────────────────────

_add(OpSpec("roll",       "rearr/roll",       "rearr", 1, "matrix",
            run=lambda a: torch.roll(a, 1, dims=0) if a.shape[0] > 0 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("flip_dim0",  "rearr/flip_dim0",  "rearr", 1, "matrix",
            run=lambda a: torch.flip(a, [0]),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("flip_dim1",  "rearr/flip_dim1",  "rearr", 1, "matrix",
            run=lambda a: torch.flip(a, [-1]),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("fliplr",     "rearr/fliplr",     "rearr", 1, "matrix",
            run=lambda a: torch.fliplr(a),
            edges=frozenset(["non-contig"])))
_add(OpSpec("flipud",     "rearr/flipud",     "rearr", 1, "matrix",
            run=lambda a: torch.flipud(a),
            edges=frozenset(["non-contig"])))
_add(OpSpec("rot90",      "rearr/rot90",      "rearr", 1, "matrix",
            run=lambda a: torch.rot90(a),
            edges=frozenset(["non-contig"])))


def _tile_run(a):
    if a.dim() >= 2:
        return torch.tile(a, (1, 2))
    return torch.tile(a, (2,))

def _repeat_run(a):
    if a.dim() >= 2:
        return a.repeat(1, 2)
    return a.repeat(2)


_add(OpSpec("tile",   "rearr/tile",   "rearr", 1, "matrix", run=_tile_run,   edges=frozenset(["non-contig"])))
_add(OpSpec("repeat", "rearr/repeat", "rearr", 1, "matrix", run=_repeat_run, edges=frozenset(["non-contig"])))
_add(OpSpec("repeat_interleave", "rearr/repeat_interleave", "rearr", 1, "matrix",
            run=lambda a: torch.repeat_interleave(a, 2, dim=-1),
            edges=frozenset(["non-contig"])))


# ── DIAGONAL / TRIANGULAR ───────────────────────────────────────────────────

_add(OpSpec("diagonal",   "diag/diagonal",   "diag", 1, "matrix",
            run=lambda a: torch.diagonal(a) if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("diag",       "diag/diag",       "diag", 1, "matrix",
            run=lambda a: torch.diag(a) if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("diag_embed", "diag/diag_embed", "diag", 1, "diag_vector",
            run=lambda a: torch.diag_embed(a),
            edges=frozenset(["non-contig"])))
_add(OpSpec("diagflat",   "diag/diagflat",   "diag", 1, "diag_vector",
            run=lambda a: torch.diagflat(a),
            edges=frozenset(["non-contig"])))
_add(OpSpec("tril",       "diag/tril",       "diag", 1, "matrix",
            run=lambda a: torch.tril(a) if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("triu",       "diag/triu",       "diag", 1, "matrix",
            run=lambda a: torch.triu(a) if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))


# ── DTYPE / DEVICE CONVERSION ───────────────────────────────────────────────

_add(OpSpec("to_float32", "cast/to_float32", "cast", 1, "vector",
            run=lambda a: a.float(),  expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_int32",   "cast/to_int32",   "cast", 1, "vector",
            run=lambda a: a.to(torch.int32),  expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_int64",   "cast/to_int64",   "cast", 1, "vector",
            run=lambda a: a.long(),  expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_uint8",   "cast/to_uint8",   "cast", 1, "vector",
            run=lambda a: a.byte(),  expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_bool",    "cast/to_bool",    "cast", 1, "vector",
            run=lambda a: a.bool(),  expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))


# ── MATMUL (xnor-popcount semantics) ────────────────────────────────────────

def _bool_pm1_matmul(a_bool, b_bool):
    a_pm1 = a_bool.float() * 2 - 1
    b_pm1 = b_bool.float() * 2 - 1
    return a_pm1 @ b_pm1.t()


def _matmul_run(a, b):
    return a @ b


def _matmul_cmp(bit1_out, bool_out, *, note=""):
    a = _to_plain(bit1_out).float().cpu()
    b = _to_plain(bool_out).float().cpu()
    assert a.shape == b.shape, f"matmul shape mismatch: {a.shape} vs {b.shape} ({note})"
    assert torch.allclose(a, b, atol=1e-3), f"matmul value mismatch ({note})"


_add(OpSpec("matmul_a_at_bt", "matmul/a_at_bt", "matmul", 2, "matmul",
            run=_matmul_run,
            oracle=_bool_pm1_matmul,
            cmp=_matmul_cmp,
            bool_cant_run=True,
            expect_bit1_result=False,
            edges=frozenset([])))


# ── BRUTE-SPECIFIC OPS ──────────────────────────────────────────────────────

def _popcount_oracle(b_bool):
    return b_bool.long().sum()


def _hamming_oracle(a_bool, b_bool):
    return (a_bool ^ b_bool).long().sum()


def _unpack_pm1_oracle(b_bool):
    return b_bool.float() * 2 - 1


def _popcount_cmp(a, b, *, note=""):
    assert int(_to_plain(a)) == int(_to_plain(b)), f"popcount mismatch ({note})"

def _hamming_cmp(a, b, *, note=""):
    assert int(_to_plain(a)) == int(_to_plain(b)), f"hamming mismatch ({note})"

def _unpack_pm1_cmp(a, b, *, note=""):
    ap = _to_plain(a).float().cpu()
    bp = _to_plain(b).float().cpu()
    assert ap.shape == bp.shape, f"unpack_pm1 shape mismatch ({note})"
    assert torch.allclose(ap, bp), f"unpack_pm1 value mismatch ({note})"


_add(OpSpec("popcount", "brute/popcount", "brute", 1, "vector",
            run=lambda a: a.popcount(),
            oracle=_popcount_oracle,
            cmp=_popcount_cmp,
            bool_cant_run=True,
            expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("hamming", "brute/hamming", "brute", 2, "vector",
            run=lambda a, b: a.hamming(b),
            oracle=_hamming_oracle,
            cmp=_hamming_cmp,
            bool_cant_run=True,
            expect_bit1_result=False,
            edges=frozenset(["empty", "tail-misalign"])))
_add(OpSpec("unpack_pm1", "brute/unpack_pm1", "brute", 1, "vector",
            run=lambda a: a.unpack_pm1(),
            oracle=_unpack_pm1_oracle,
            cmp=_unpack_pm1_cmp,
            bool_cant_run=True,
            expect_bit1_result=False,
            edges=frozenset(["tail-misalign"])))


# ─────────────────────────────────────────────────────────────────────────────
# FORM PARITY — Tensor.<method>() and torch.<fn>() variants
#
# Every op above is exercised via one of: operator (`a & b`, `a == b`),
# free function (`torch.bitwise_and(a, b)`), or method (`a.float()`).
# This section adds the *other* forms — so each op is covered as torch.fn
# AND Tensor.method where both exist, ensuring dispatch through both
# torch_function entry points is identical.
#
# Naming: `<base>_method` for the bound-method form, `<base>_fn` for the
# free-function form when the original spec was method-only.
# Groups are kept unique so the bench report can pair bit1 / bool cleanly.
# ─────────────────────────────────────────────────────────────────────────────

# ── BITWISE: method forms ───────────────────────────────────────────────────
_add(OpSpec("bitwise_and_method", "bitwise/and_method", "bitwise", 2, "vector",
            run=lambda a, b: a.bitwise_and(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_or_method",  "bitwise/or_method",  "bitwise", 2, "vector",
            run=lambda a, b: a.bitwise_or(b),  edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_xor_method", "bitwise/xor_method", "bitwise", 2, "vector",
            run=lambda a, b: a.bitwise_xor(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("bitwise_not_method", "bitwise/not_method", "bitwise", 1, "vector",
            run=lambda a: a.bitwise_not(),     edges=_DEFAULT_UNARY_EDGES))

# ── LOGICAL: method forms ───────────────────────────────────────────────────
_add(OpSpec("logical_and_method", "logical/and_method", "logical", 2, "vector",
            run=lambda a, b: a.logical_and(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("logical_or_method",  "logical/or_method",  "logical", 2, "vector",
            run=lambda a, b: a.logical_or(b),  edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("logical_xor_method", "logical/xor_method", "logical", 2, "vector",
            run=lambda a, b: a.logical_xor(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("logical_not_method", "logical/not_method", "logical", 1, "vector",
            run=lambda a: a.logical_not(),     edges=_DEFAULT_UNARY_EDGES))

# ── COMPARISON: method + fn forms ───────────────────────────────────────────
_add(OpSpec("eq_method", "compare/eq_method", "compare", 2, "vector",
            run=lambda a, b: a.eq(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ne_method", "compare/ne_method", "compare", 2, "vector",
            run=lambda a, b: a.ne(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("lt_fn",     "compare/lt_fn",     "compare", 2, "vector",
            run=lambda a, b: torch.lt(a, b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("lt_method", "compare/lt_method", "compare", 2, "vector",
            run=lambda a, b: a.lt(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("le_fn",     "compare/le_fn",     "compare", 2, "vector",
            run=lambda a, b: torch.le(a, b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("le_method", "compare/le_method", "compare", 2, "vector",
            run=lambda a, b: a.le(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("gt_fn",     "compare/gt_fn",     "compare", 2, "vector",
            run=lambda a, b: torch.gt(a, b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("gt_method", "compare/gt_method", "compare", 2, "vector",
            run=lambda a, b: a.gt(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ge_fn",     "compare/ge_fn",     "compare", 2, "vector",
            run=lambda a, b: torch.ge(a, b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("ge_method", "compare/ge_method", "compare", 2, "vector",
            run=lambda a, b: a.ge(b), edges=_DEFAULT_BINARY_EDGES))
_add(OpSpec("equal_method", "compare/equal_method", "compare", 2, "vector",
            run=lambda a, b: a.equal(b),
            edges=frozenset(["empty"]),
            expect_bit1_result=False))

# ── REDUCTIONS (full): method forms ─────────────────────────────────────────
_add(OpSpec("sum_full_method", "reduce/sum_full_method", "reduce", 1, "vector",
            run=lambda a: a.sum(),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("count_nonzero_method", "reduce/count_nonzero_method", "reduce", 1, "vector",
            run=lambda a: a.count_nonzero(),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("all_full_method", "reduce/all_full_method", "reduce", 1, "vector",
            run=lambda a: a.all(),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("any_full_method", "reduce/any_full_method", "reduce", 1, "vector",
            run=lambda a: a.any(),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("argmax_full_method", "reduce/argmax_full_method", "reduce", 1, "vector",
            run=lambda a: a.argmax(),
            oracle=lambda b: b.long().argmax(),
            bool_cant_run=True,
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("argmin_full_method", "reduce/argmin_full_method", "reduce", 1, "vector",
            run=lambda a: a.argmin(),
            oracle=lambda b: b.long().argmin(),
            bool_cant_run=True,
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("max_full_method", "reduce/max_full_method", "reduce", 1, "vector",
            run=lambda a: a.max(),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("min_full_method", "reduce/min_full_method", "reduce", 1, "vector",
            run=lambda a: a.min(),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("nonzero_method", "reduce/nonzero_method", "reduce", 1, "vector",
            run=lambda a: a.nonzero(),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("argwhere_method", "reduce/argwhere_method", "reduce", 1, "vector",
            run=lambda a: a.argwhere(),
            edges=frozenset(["empty", "non-contig", "tail-misalign"]),
            expect_bit1_result=False))

# ── REDUCTIONS (dim): method forms ──────────────────────────────────────────
_add(OpSpec("sum_dim0_method", "reduce_dim/sum_dim0_method", "reduce_dim", 1, "matrix",
            run=lambda a: a.sum(dim=0),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("sum_dim1_method", "reduce_dim/sum_dim1_method", "reduce_dim", 1, "matrix",
            run=lambda a: a.sum(dim=1),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("all_dim0_method", "reduce_dim/all_dim0_method", "reduce_dim", 1, "matrix",
            run=lambda a: a.all(dim=0),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("any_dim_last_method", "reduce_dim/any_dim_last_method", "reduce_dim", 1, "matrix",
            run=lambda a: a.any(dim=-1),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("count_nonzero_dim0_method", "reduce_dim/count_nonzero_dim0_method",
            "reduce_dim", 1, "matrix",
            run=lambda a: a.count_nonzero(dim=0),
            edges=frozenset(["non-contig", "tail-misalign"]),
            expect_bit1_result=False))
_add(OpSpec("cumsum_dim_last_method", "reduce_dim/cumsum_dim_last_method",
            "reduce_dim", 1, "matrix",
            run=lambda a: a.cumsum(dim=-1),
            edges=frozenset(["non-contig"]),
            expect_bit1_result=False))


# ── SHAPE / VIEW: missing fn / method forms ─────────────────────────────────
_add(OpSpec("reshape_flat_fn", "view/reshape_flat_fn", "view", 1, "matrix",
            run=lambda a: torch.reshape(a, (-1,)),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("flatten_method", "view/flatten_method", "view", 1, "matrix",
            run=lambda a: a.flatten(),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("squeeze_fn", "view/squeeze_fn", "view", 1, "matrix",
            run=lambda a: torch.squeeze(a.unsqueeze(0), 0),
            edges=frozenset(["non-contig"])))
_add(OpSpec("unsqueeze_fn_0", "view/unsqueeze_fn_0", "view", 1, "matrix",
            run=lambda a: torch.unsqueeze(a, 0),
            edges=frozenset(["non-contig"])))
_add(OpSpec("unsqueeze_fn_last", "view/unsqueeze_fn_last", "view", 1, "matrix",
            run=lambda a: torch.unsqueeze(a, -1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("t_fn", "view/t_fn", "view", 1, "matrix",
            run=lambda a: torch.t(a),
            edges=frozenset(["non-contig"])))
_add(OpSpec("transpose_01_fn", "view/transpose_01_fn", "view", 1, "matrix",
            run=lambda a: torch.transpose(a, 0, 1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("permute_fn", "view/permute_fn", "view", 1, "matrix",
            run=lambda a: torch.permute(a.unsqueeze(0), (0, 2, 1)),
            edges=frozenset(["non-contig"])))
_add(OpSpec("movedim_method", "view/movedim_method", "view", 1, "matrix",
            run=lambda a: a.movedim(0, -1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("swapaxes_method", "view/swapaxes_method", "view", 1, "matrix",
            run=lambda a: a.swapaxes(0, 1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("swapdims_fn", "view/swapdims_fn", "view", 1, "matrix",
            run=lambda a: torch.swapdims(a, 0, 1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("swapdims_method", "view/swapdims_method", "view", 1, "matrix",
            run=lambda a: a.swapdims(0, 1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("moveaxis_fn", "view/moveaxis_fn", "view", 1, "matrix",
            run=lambda a: torch.moveaxis(a, 0, -1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("moveaxis_method", "view/moveaxis_method", "view", 1, "matrix",
            run=lambda a: a.moveaxis(0, -1),
            edges=frozenset(["non-contig"])))
_add(OpSpec("ravel_fn", "view/ravel_fn", "view", 1, "matrix",
            run=lambda a: torch.ravel(a),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("ravel_method", "view/ravel_method", "view", 1, "matrix",
            run=lambda a: a.ravel(),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("broadcast_to_method", "view/broadcast_to_method", "view", 1, "matrix",
            run=lambda a: (a[:1].broadcast_to(a.shape).contiguous()
                           if (a.dim() >= 2 and a.shape[0] > 0) else a),
            edges=frozenset([])))
_add(OpSpec("expand_as_method", "view/expand_as_method", "view", 1, "matrix",
            run=lambda a: (a[:1].expand_as(a).contiguous()
                           if (a.dim() >= 2 and a.shape[0] > 0) else a),
            edges=frozenset([])))
_add(OpSpec("reshape_as_method", "view/reshape_as_method", "view", 1, "matrix",
            run=lambda a: a.reshape_as(a),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("view_as_method", "view/view_as_method", "view", 1, "matrix",
            run=lambda a: a.contiguous().view_as(a.contiguous()),
            edges=frozenset(["tail-misalign"])))
_add(OpSpec("narrow_fn", "view/narrow_fn", "view", 1, "matrix",
            run=lambda a: torch.narrow(a, 0, 0, a.shape[0] // 2) if a.shape[0] > 1 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("select_fn", "view/select_fn", "view", 1, "matrix",
            run=lambda a: torch.select(a, 0, 0) if a.shape[0] > 0 else a,
            edges=frozenset(["non-contig"])))


# ── COMBINING: method forms ─────────────────────────────────────────────────
def _split_eq_method(a):
    if a.shape[0] < 2:
        return (a,)
    return a.split(max(1, a.shape[0] // 2), dim=0)

def _chunk_eq_method(a):
    if a.shape[0] < 2:
        return (a,)
    return a.chunk(2, dim=0)

def _unbind_eq_method(a):
    return a.unbind(dim=0)


_add(OpSpec("split_method",  "combine/split_method",  "combine", 1, "matrix",
            run=_split_eq_method,  edges=frozenset(["non-contig"])))
_add(OpSpec("chunk_method",  "combine/chunk_method",  "combine", 1, "matrix",
            run=_chunk_eq_method,  edges=frozenset(["non-contig"])))
_add(OpSpec("unbind_method", "combine/unbind_method", "combine", 1, "matrix",
            run=_unbind_eq_method, edges=frozenset(["non-contig"])))


# ── INDEXING: method / fn forms ─────────────────────────────────────────────
def _gather_method(a):
    if a.dim() < 2 or a.shape[1] == 0:
        return a
    idx = torch.zeros((a.shape[0], 1), dtype=torch.long, device=a.device)
    return a.gather(1, idx)


def _index_select_method(a):
    if a.shape[0] == 0:
        return a
    idx = torch.tensor([0], device=a.device)
    return a.index_select(0, idx)


def _take_method(a):
    if a.numel() == 0:
        return a
    idx = torch.tensor([0, a.numel() - 1], device=a.device)
    return a.take(idx)


def _scatter_fn(a):
    if a.dim() < 2 or a.shape[1] == 0:
        return a.clone()
    idx = torch.zeros((a.shape[0], 1), dtype=torch.long, device=a.device)
    src = a[:, :1]
    return torch.scatter(a, 1, idx, src)


def _masked_select_method(a):
    mask = _build_bool_mask(a)
    return a.masked_select(mask)


def _masked_fill_fn(a):
    a = a.clone()
    mask = torch.zeros(a.numel(), dtype=torch.bool, device=a.device)
    if a.numel() > 0:
        mask[0] = True
    return torch.masked_fill(a, mask.view(a.shape), True)


def _where_method(a):
    t = torch.ones(a.shape, dtype=torch.bool, device=a.device)
    f = torch.zeros(a.shape, dtype=torch.bool, device=a.device)
    return a.where(t, f)


_add(OpSpec("gather_method",        "index/gather_method",        "index", 1, "matrix",
            run=_gather_method,        edges=frozenset(["non-contig"])))
_add(OpSpec("index_select_method",  "index/index_select_method",  "index", 1, "matrix",
            run=_index_select_method,  edges=frozenset(["non-contig"])))
_add(OpSpec("take_method",          "index/take_method",          "index", 1, "matrix",
            run=_take_method,          edges=frozenset(["non-contig"])))
_add(OpSpec("scatter_fn",           "index/scatter_fn",           "index", 1, "matrix",
            run=_scatter_fn,           edges=frozenset(["non-contig"])))
_add(OpSpec("masked_select_method", "index/masked_select_method", "index", 1, "matrix",
            run=_masked_select_method, edges=frozenset(["non-contig"])))
_add(OpSpec("masked_fill_fn",       "index/masked_fill_fn",       "index", 1, "matrix",
            run=_masked_fill_fn,
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("where_method",         "index/where_method",         "index", 1, "matrix",
            run=_where_method,         edges=frozenset(["non-contig"])))


# ── CLONE / DETACH: missing fn forms ────────────────────────────────────────
_add(OpSpec("clone_fn",  "clone/clone_fn",  "clone", 1, "matrix",
            run=lambda a: torch.clone(a),
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("detach_fn", "clone/detach_fn", "clone", 1, "matrix",
            run=lambda a: torch.detach(a),
            edges=frozenset(["non-contig"])))


# ── REARRANGEMENT: method forms ─────────────────────────────────────────────
_add(OpSpec("roll_method", "rearr/roll_method", "rearr", 1, "matrix",
            run=lambda a: a.roll(1, dims=0) if a.shape[0] > 0 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("flip_dim0_method", "rearr/flip_dim0_method", "rearr", 1, "matrix",
            run=lambda a: a.flip([0]),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("flip_dim1_method", "rearr/flip_dim1_method", "rearr", 1, "matrix",
            run=lambda a: a.flip([-1]),
            edges=frozenset(["non-contig", "tail-misalign"])))
_add(OpSpec("fliplr_method", "rearr/fliplr_method", "rearr", 1, "matrix",
            run=lambda a: a.fliplr(),
            edges=frozenset(["non-contig"])))
_add(OpSpec("flipud_method", "rearr/flipud_method", "rearr", 1, "matrix",
            run=lambda a: a.flipud(),
            edges=frozenset(["non-contig"])))
_add(OpSpec("rot90_method", "rearr/rot90_method", "rearr", 1, "matrix",
            run=lambda a: a.rot90(),
            edges=frozenset(["non-contig"])))


def _tile_method(a):
    if a.dim() >= 2:
        return a.tile((1, 2))
    return a.tile((2,))


_add(OpSpec("tile_method", "rearr/tile_method", "rearr", 1, "matrix",
            run=_tile_method, edges=frozenset(["non-contig"])))
_add(OpSpec("repeat_interleave_method", "rearr/repeat_interleave_method",
            "rearr", 1, "matrix",
            run=lambda a: a.repeat_interleave(2, dim=-1),
            edges=frozenset(["non-contig"])))


# ── DIAGONAL / TRIANGULAR: method forms ─────────────────────────────────────
_add(OpSpec("diagonal_method", "diag/diagonal_method", "diag", 1, "matrix",
            run=lambda a: a.diagonal() if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("diag_method", "diag/diag_method", "diag", 1, "matrix",
            run=lambda a: a.diag() if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("diag_embed_method", "diag/diag_embed_method", "diag", 1, "diag_vector",
            run=lambda a: a.diag_embed(),
            edges=frozenset(["non-contig"])))
_add(OpSpec("diagflat_method", "diag/diagflat_method", "diag", 1, "diag_vector",
            run=lambda a: a.diagflat(),
            edges=frozenset(["non-contig"])))
_add(OpSpec("tril_method", "diag/tril_method", "diag", 1, "matrix",
            run=lambda a: a.tril() if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))
_add(OpSpec("triu_method", "diag/triu_method", "diag", 1, "matrix",
            run=lambda a: a.triu() if a.dim() >= 2 else a,
            edges=frozenset(["non-contig"])))


# ── CAST: torch.Tensor.to(dtype) and torch.<dtype>(...) parity ──────────────
# The existing `to_*` specs use the convenience methods (.float(), .long(),
# .byte(), .bool()). Add explicit `.to(dtype=...)` and torch.fn equivalents
# (where they exist as free functions) so the dispatch path through
# torch.Tensor.to is exercised separately.
_add(OpSpec("to_float32_to",  "cast/to_float32_to",  "cast", 1, "vector",
            run=lambda a: a.to(torch.float32), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_int64_to",    "cast/to_int64_to",    "cast", 1, "vector",
            run=lambda a: a.to(torch.int64), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_uint8_to",    "cast/to_uint8_to",    "cast", 1, "vector",
            run=lambda a: a.to(torch.uint8), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_bool_to",     "cast/to_bool_to",     "cast", 1, "vector",
            run=lambda a: a.to(torch.bool), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_int_method",  "cast/to_int_method",  "cast", 1, "vector",
            run=lambda a: a.int(), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_short_method","cast/to_short_method","cast", 1, "vector",
            run=lambda a: a.short(), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_char_method", "cast/to_char_method", "cast", 1, "vector",
            run=lambda a: a.char(), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
_add(OpSpec("to_half_method", "cast/to_half_method", "cast", 1, "vector",
            run=lambda a: a.half(), expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))
# NOTE: to_double / .double() is intentionally omitted — MPS does not support
# float64 and would TypeError. Float casting is already covered by .float() and
# .to(torch.float32).
_add(OpSpec("type_as_method", "cast/type_as_method", "cast", 1, "vector",
            run=lambda a: a.type_as(torch.empty(0, dtype=torch.float32, device=a.device)),
            expect_bit1_result=False,
            edges=frozenset(["empty", "non-contig", "tail-misalign"])))


# ── MATMUL: method and fn forms ─────────────────────────────────────────────
def _matmul_method(a, b):
    return a.matmul(b.t())


def _matmul_fn(a, b):
    return torch.matmul(a, b.t())


def _mm_method(a, b):
    return a.mm(b.t())


def _mm_fn(a, b):
    return torch.mm(a, b.t())


_add(OpSpec("matmul_method", "matmul/matmul_method", "matmul", 2, "matmul",
            run=_matmul_method, oracle=_bool_pm1_matmul,
            cmp=_matmul_cmp, bool_cant_run=True,
            expect_bit1_result=False, edges=frozenset([])))
_add(OpSpec("matmul_fn",     "matmul/matmul_fn",     "matmul", 2, "matmul",
            run=_matmul_fn, oracle=_bool_pm1_matmul,
            cmp=_matmul_cmp, bool_cant_run=True,
            expect_bit1_result=False, edges=frozenset([])))
_add(OpSpec("mm_method",     "matmul/mm_method",     "matmul", 2, "matmul",
            run=_mm_method, oracle=_bool_pm1_matmul,
            cmp=_matmul_cmp, bool_cant_run=True,
            expect_bit1_result=False, edges=frozenset([])))
_add(OpSpec("mm_fn",         "matmul/mm_fn",         "matmul", 2, "matmul",
            run=_mm_fn, oracle=_bool_pm1_matmul,
            cmp=_matmul_cmp, bool_cant_run=True,
            expect_bit1_result=False, edges=frozenset([])))


# ─────────────────────────────────────────────────────────────────────────────
# Accessor helpers
# ─────────────────────────────────────────────────────────────────────────────

def by_category() -> dict[str, list[OpSpec]]:
    out: dict[str, list[OpSpec]] = {}
    for s in CATALOG:
        out.setdefault(s.category, []).append(s)
    return out


def get(name: str) -> OpSpec:
    for s in CATALOG:
        if s.name == name:
            return s
    raise KeyError(name)


def names() -> list[str]:
    return [s.name for s in CATALOG]


__all__ = [
    "OpSpec",
    "CATALOG",
    "ALL_EDGES",
    "parity_shape",
    "parity_input_shapes",
    "edge_inputs",
    "default_cmp",
    "by_category",
    "get",
    "names",
]
