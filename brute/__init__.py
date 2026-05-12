import brute._C  # triggers TORCH_LIBRARY static-init registration

# ── Core tensor type & dtypes ─────────────────────────────────────────────────
from brute.tensor import (
    Tensor,
    bit1,
    float32, float16, bfloat16,
    int8, int32, int64,
)

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
