"""Helpers for the brute test suite."""
from tests.helpers.assertions import (
    assert_tensors_equal,
    assert_bit1_matches_bool,
    assert_subclass,
    assert_device,
    assert_dtype,
    assert_same_storage,
    assert_different_storage,
)
from tests.helpers.generators import (
    bit1,
    bool_tensor,
    random_bit1,
    random_bool,
    random_shape,
    shape_strategy,
    bool_strategy,
    int_strategy,
    float_strategy,
)
from tests.helpers.reference_ops import (
    ref_xnor_popcount_matmul,
    ref_popcount,
    ref_matmul_bool,
)

__all__ = [
    "assert_tensors_equal",
    "assert_bit1_matches_bool",
    "assert_subclass",
    "assert_device",
    "assert_dtype",
    "assert_same_storage",
    "assert_different_storage",
    "bit1",
    "bool_tensor",
    "random_bit1",
    "random_bool",
    "random_shape",
    "shape_strategy",
    "bool_strategy",
    "int_strategy",
    "float_strategy",
    "ref_xnor_popcount_matmul",
    "ref_popcount",
    "ref_matmul_bool",
]
