import torch as _t

try:
    from . import _cbrute
except ImportError:
    # Try site-packages location (i.e. for editable installs).
    try:
        import importlib.util
        spec = importlib.util.find_spec('_cbrute')
        if spec and spec.origin:
            _cbrute = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_cbrute)
    except Exception:
        raise ImportError(
            "brute requires the compiled _cbrute extension. "
            "Install with: pip install --no-build-isolation -ve ."
        )

from brute.tensor import Tensor, bit1

# ── dtypes ────────────────────────────────────────────────────────────────────
bool            = _t.bool
float32         = _t.float32
float64         = _t.float64
float16         = _t.float16
bfloat16        = _t.bfloat16
int8            = _t.int8
int16           = _t.int16
int32           = _t.int32
int64           = _t.int64
uint8           = _t.uint8
uint16          = _t.uint16
uint32          = _t.uint32
uint64          = _t.uint64
complex64       = _t.complex64
complex128      = _t.complex128
float8_e4m3fn   = _t.float8_e4m3fn
float8_e5m2     = _t.float8_e5m2

# ── Factory / creator functions ───────────────────────────────────────────────
from brute.functional import (
    # Bit1-aware creators
    zeros, ones, empty, full,
    tensor, as_tensor, from_numpy,
    # Random creators
    rand, randn, randint,
    rand_like, randn_like, zeros_like, ones_like, full_like, empty_like,
    # Range / grid
    arange, linspace, eye,
)

# ── Reductions ────────────────────────────────────────────────────────────────
from brute.functional import (
    all, any, sum, max, min, mean, prod,
    amax, amin, aminmax,
    argmax, argmin,
    count_nonzero, nonzero, argwhere,
    nansum, nanmean, logsumexp,
)

# ── Combining / stacking ──────────────────────────────────────────────────────
from brute.functional import cat, stack

# ── Shape manipulation ────────────────────────────────────────────────────────
from brute.functional import (
    reshape, flatten, squeeze, unsqueeze,
    permute, transpose, t,
    movedim, moveaxis, swapaxes, swapdims,
    broadcast_to, broadcast_tensors,
    narrow, select,
    atleast_1d, atleast_2d, atleast_3d,
)

# ── Splitting ─────────────────────────────────────────────────────────────────
from brute.functional import (
    split, chunk, unbind, tensor_split,
    hsplit, vsplit, dsplit,
)

# ── Clone / detach ────────────────────────────────────────────────────────────
from brute.functional import clone, detach

# ── Logical (bit1-relevant) ───────────────────────────────────────────────────
from brute.functional import (
    where,
    logical_and, logical_or, logical_xor, logical_not,
)

# ── Bitwise ───────────────────────────────────────────────────────────────────
from brute.functional import (
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
    bitwise_left_shift, bitwise_right_shift,
)

# ── Comparison ────────────────────────────────────────────────────────────────
from brute.functional import (
    eq, ne, lt, le, gt, ge,
    equal, allclose, isclose,
    isnan, isinf, isfinite, isneginf, isposinf,
)

# ── Arithmetic ────────────────────────────────────────────────────────────────
from brute.functional import (
    add, sub, mul, div, divide,
    neg, negative, abs, absolute,
    sign, sgn,
    clamp, clip,
    pow, sqrt, rsqrt,
)

# ── Cumulative ────────────────────────────────────────────────────────────────
from brute.functional import cumsum, cumprod, cummax, cummin

# ── Sorting / selection ───────────────────────────────────────────────────────
from brute.functional import (
    sort, argsort, topk, kthvalue, msort,
    unique, unique_consecutive,
)

# ── Indexing / masking ────────────────────────────────────────────────────────
from brute.functional import (
    gather, index_select, masked_select, take, scatter,
)

# ── Rearrangement / replication ───────────────────────────────────────────────
from brute.functional import (
    roll, flip, fliplr, flipud, rot90,
    tile, repeat_interleave,
)

# ── Matrix / linear algebra ───────────────────────────────────────────────────
from brute.functional import (
    mm, bmm, matmul, mv, dot, inner, outer, vdot, cross,
)

# ── Diagonal / triangular ─────────────────────────────────────────────────────
from brute.functional import (
    diagonal, diag, diag_embed, diagflat,
    tril, triu, trace,
)

# ── Misc ──────────────────────────────────────────────────────────────────────
from brute.functional import (
    nan_to_num, diff, block_diag, cartesian_prod,
)
