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

# Re-export torch dtypes
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

# Factory and utility functions
from brute.functional import (
    # creators
    zeros, ones, empty, full,
    tensor, as_tensor, from_numpy,
    # random
    rand, randn, randint,
    rand_like, randn_like, zeros_like, ones_like,
    # range / grid
    arange, linspace, eye,
    # reductions & ops (re-exported from torch, work via __torch_function__)
    all, any, sum, max, min, mean, prod,
    cat, stack, where,
)
