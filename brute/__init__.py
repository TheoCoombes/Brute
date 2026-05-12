import brute._C  # triggers TORCH_LIBRARY static-init registration

# ── Core tensor type & bit1 dtype ─────────────────────────────────────────────
from brute.tensor import Tensor, bit1

# ── All torch dtypes, re-exported for convenience ─────────────────────────────
# Scalar / float
from brute.dtypes import (
    float32, float64, float16, bfloat16,
    complex64, complex128,
    float8_e4m3fn, float8_e5m2,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
)
import torch as _torch
bool = _torch.bool   # brute.bool is torch.bool (regular bool tensor dtype)

# ── Factory and utility functions ─────────────────────────────────────────────
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
