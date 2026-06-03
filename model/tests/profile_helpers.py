"""Small instrumentation helpers for model efficiency tests."""

from __future__ import annotations

import contextlib

import brute
import brute.fast as bfast
from brute.tensor import Tensor as BruteTensor


@contextlib.contextmanager
def count_hot_ops():
    counts = {
        "fast_matmul": 0,
        "fast_matmul_sign": 0,
        "fast_majority": 0,
        "pack_sign": 0,
        "unpack_pm1": 0,
        "as_tensor_bit1_pack": 0,
        "threshold_pack": 0,
    }
    orig_matmul      = bfast.matmul
    orig_matmul_sign = bfast.matmul_sign
    orig_majority    = bfast.majority
    orig_sign        = bfast.sign
    orig_unpack      = BruteTensor.unpack_pm1
    orig_as_tensor   = brute.as_tensor

    def matmul_wrap(*args, **kwargs):
        counts["fast_matmul"] += 1
        return orig_matmul(*args, **kwargs)

    def matmul_sign_wrap(*args, **kwargs):
        counts["fast_matmul_sign"] += 1
        return orig_matmul_sign(*args, **kwargs)

    def majority_wrap(*args, **kwargs):
        counts["fast_majority"] += 1
        return orig_majority(*args, **kwargs)

    def sign_wrap(*args, **kwargs):
        counts["threshold_pack"] += 1
        return orig_sign(*args, **kwargs)

    def unpack_wrap(self, *args, **kwargs):
        counts["unpack_pm1"] += 1
        return orig_unpack(self, *args, **kwargs)

    def as_tensor_wrap(data, *, dtype=None, device=None):
        if dtype is brute.bit1:
            counts["as_tensor_bit1_pack"] += 1
        return orig_as_tensor(data, dtype=dtype, device=device)

    bfast.matmul      = matmul_wrap
    brute.fast.matmul = matmul_wrap
    bfast.matmul_sign      = matmul_sign_wrap
    brute.fast.matmul_sign = matmul_sign_wrap
    bfast.majority         = majority_wrap
    brute.fast.majority    = majority_wrap
    bfast.sign      = sign_wrap
    brute.fast.sign = sign_wrap
    BruteTensor.unpack_pm1 = unpack_wrap
    brute.as_tensor = as_tensor_wrap
    try:
        yield counts
    finally:
        bfast.matmul      = orig_matmul
        brute.fast.matmul = orig_matmul
        bfast.matmul_sign      = orig_matmul_sign
        brute.fast.matmul_sign = orig_matmul_sign
        bfast.majority         = orig_majority
        brute.fast.majority    = orig_majority
        bfast.sign      = orig_sign
        brute.fast.sign = orig_sign
        BruteTensor.unpack_pm1 = orig_unpack
        brute.as_tensor = orig_as_tensor
