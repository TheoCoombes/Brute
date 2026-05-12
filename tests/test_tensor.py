"""
Comprehensive test suite for brute.Tensor.

Design rules
────────────
1. Every bit1 op has a matching bool-tensor reference computed with plain torch.
   The bit1 result must equal the bool reference 1:1 (values), and remain a
   brute.Tensor (return-type consistency).
2. bool (brute.bool / torch.bool) and bit1 (brute.bit1) are strictly separate
   dtypes — bit1 + bit1 may stay bit1, but bit1 + plain-bool must NOT promote
   the plain-bool side to bit1.
3. pack_dtype is a plain torch.dtype (brute.uint8 / brute.uint32 / brute.uint64).
4. In-place ops (fill_, copy_, logical_*_, __setitem__, …) on a bit1 tensor
   must leave the packed buffer correct on the next access — verified by
   re-packing from scratch and bytewise-equal.
5. All matmul results are verified against a float {-1,+1} reference.
6. Cross-dtype pairs (bit1 + bool, bit1 + float32, bit1 + int64) are tested
   end-to-end for every binary op category.
"""

from __future__ import annotations

import copy
import pickle
import itertools

import pytest
import torch

import brute


# ── Helpers ────────────────────────────────────────────────────────────────────

def base(t) -> torch.Tensor:
    """Underlying torch.Tensor (the bool base for bit1, plain tensor otherwise)."""
    return t.as_subclass(torch.Tensor) if isinstance(t, brute.Tensor) else t


def _ref_pm1(a_bool: torch.Tensor, b_bool: torch.Tensor) -> torch.Tensor:
    """Float {-1,+1} reference matmul for XNOR-popcount semantics."""
    return (a_bool.float() * 2 - 1) @ (b_bool.float() * 2 - 1).t()


def assert_brute(result, dtype=None):
    """Assert *result* is a brute.Tensor (optionally with a specific dtype)."""
    assert isinstance(result, brute.Tensor), \
        f"Expected brute.Tensor, got {type(result).__name__}"
    if dtype is not None:
        assert result.dtype == dtype, \
            f"Expected dtype={dtype}, got {result.dtype}"


def assert_bit1(result):
    """Assert *result* is a brute.Tensor with dtype=bit1."""
    assert isinstance(result, brute.Tensor), \
        f"Expected brute.Tensor, got {type(result).__name__}"
    assert result.dtype == brute.bit1, \
        f"Expected dtype=bit1, got {result.dtype}"
    assert getattr(result, '_is_bit1', False), "Expected _is_bit1=True"


def assert_plain_bool(result):
    """Assert *result* is a brute.Tensor with dtype=torch.bool (NOT bit1)."""
    assert isinstance(result, brute.Tensor), \
        f"Expected brute.Tensor, got {type(result).__name__}"
    assert result.dtype == torch.bool, \
        f"Expected dtype=torch.bool, got {result.dtype}"
    assert not getattr(result, '_is_bit1', False), \
        "Expected _is_bit1=False (plain bool, not bit1)"


def assert_values_eq(result, ref):
    """Assert *result* has the same values as *ref* (bool comparison)."""
    rv = base(result)
    rv_bool = rv.bool() if rv.dtype != ref.dtype else rv
    assert torch.equal(rv_bool, ref), \
        f"Values differ:\n  got: {rv_bool.tolist()}\n  ref: {ref.tolist()}"


def assert_parity(bit1_result, bool_result, *, expect_bit1=True):
    """Assert a bit1 op result matches its bool reference and has the right dtype."""
    assert_brute(bit1_result)
    if expect_bit1:
        assert bit1_result.dtype == brute.bit1, \
            f"Expected bit1 result, got {bit1_result.dtype}"
    rv = base(bit1_result)
    if rv.dtype == bool_result.dtype:
        assert torch.equal(rv, bool_result), \
            f"Values differ:\n  got: {rv}\n  ref: {bool_result}"
    else:
        assert torch.equal(rv.to(bool_result.dtype), bool_result), \
            f"Values differ:\n  got: {rv}\n  ref: {bool_result}"


def make_pair(data):
    """Return (bit1_tensor, bool_tensor) constructed from the same data."""
    bool_t = torch.tensor(data, dtype=torch.bool)
    bit1_t = brute.tensor(data, dtype=brute.bit1)
    return bit1_t, bool_t


# ── Test data ──────────────────────────────────────────────────────────────────

DATA_1D_A = [True, False, True, False, True, False, True, False]
DATA_1D_B = [False, True, False, True, False, True, False, True]
DATA_1D_C = [True,  True,  False, False, True,  True,  False, False]

DATA_2D_A = [[True, False, True, False],
             [False, True, False, True]]
DATA_2D_B = [[True,  True,  False, False],
             [False, False, True,  True]]

DATA_3D_A = [[[True, False], [True, True]],
             [[False, True], [False, False]]]

PACK_DTYPES = [brute.uint8, brute.uint32, brute.uint64]


# ════════════════════════════════════════════════════════════════════════════════
#   Section 1: Dtype identity, pack_dtype, and dtype separation
# ════════════════════════════════════════════════════════════════════════════════

class TestPackDtype:
    """pack_dtype is a plain torch.dtype — brute.uint8/uint32/uint64 == torch.uint8/uint32/uint64."""

    def test_pack_dtypes_are_torch_dtypes(self):
        assert brute.uint8  is torch.uint8
        assert brute.uint32 is torch.uint32
        assert brute.uint64 is torch.uint64

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_tensor_pack_dtype(self, pw):
        assert brute.zeros(4, dtype=brute.bit1, pack_dtype=pw).pack_dtype is pw

    def test_default_pack_dtype_is_uint8(self):
        # User-facing default in functional creators is uint8.
        assert brute.zeros(4, dtype=brute.bit1).pack_dtype is torch.uint8

    def test_pack_dtype_none_for_non_bit1(self):
        assert brute.zeros(4).pack_dtype is None

    def test_pack_dtype_none_for_bool_tensor(self):
        assert brute.zeros(4, dtype=torch.bool).pack_dtype is None

    @pytest.mark.parametrize("invalid", [torch.float32, torch.int16, torch.int64, torch.float64])
    def test_invalid_pack_dtype_raises(self, invalid):
        with pytest.raises(TypeError):
            brute.zeros(4, dtype=brute.bit1, pack_dtype=invalid)


class TestAllDtypes:
    """brute re-exports all torch dtypes."""

    @pytest.mark.parametrize("name,expected", [
        ('float32',       torch.float32),
        ('float64',       torch.float64),
        ('float16',       torch.float16),
        ('bfloat16',      torch.bfloat16),
        ('int8',          torch.int8),
        ('int16',         torch.int16),
        ('int32',         torch.int32),
        ('int64',         torch.int64),
        ('uint8',         torch.uint8),
        ('uint16',        torch.uint16),
        ('uint32',        torch.uint32),
        ('uint64',        torch.uint64),
        ('bool',          torch.bool),
        ('complex64',     torch.complex64),
        ('complex128',    torch.complex128),
        ('float8_e4m3fn', torch.float8_e4m3fn),
        ('float8_e5m2',   torch.float8_e5m2),
    ])
    def test_dtype_exported(self, name, expected):
        assert getattr(brute, name) is expected


class TestDtypeSeparation:
    """bit1 ≠ plain bool — promotion to bit1 only happens when ALL bool inputs are bit1."""

    def test_brute_bool_is_torch_bool(self):
        assert brute.bool is torch.bool

    def test_bit1_is_not_torch_bool(self):
        assert brute.bit1 != torch.bool
        assert torch.bool != brute.bit1

    def test_bit1_equals_itself(self):
        assert brute.bit1 == brute.bit1

    def test_bit1_dtype_property(self):
        t = brute.zeros(3, dtype=brute.bit1)
        assert t.dtype == brute.bit1
        assert t.dtype != torch.bool

    def test_bool_tensor_dtype_is_torch_bool(self):
        t = brute.tensor([True, False], dtype=torch.bool)
        assert t.dtype == torch.bool
        assert t.dtype != brute.bit1

    def test_list_bool_defaults_to_torch_bool(self):
        t = brute.tensor([True, False])
        assert t.dtype == torch.bool
        assert not getattr(t, '_is_bit1', False)

    def test_is_bit1_flag(self):
        b = brute.tensor([True, False], dtype=torch.bool)
        q = brute.tensor([True, False], dtype=brute.bit1)
        assert not b._is_bit1 and q._is_bit1

    def test_cat_bit1_and_bool_does_not_produce_bit1(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=torch.bool)
        r = torch.cat([a, b])
        assert not getattr(r, '_is_bit1', False)
        assert r.dtype == torch.bool

    def test_and_bit1_and_plain_torch_bool_no_promotion(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1)
        b = torch.tensor([True, True, False])
        r = torch.logical_and(a, b)
        assert not getattr(r, '_is_bit1', False)

    def test_cat_two_bit1_preserves_bit1(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=brute.bit1)
        r = torch.cat([a, b])
        assert_bit1(r)

    def test_and_two_bit1_preserves_bit1(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1)
        b = brute.tensor([True, True, False], dtype=brute.bit1)
        assert_bit1(a & b)

    def test_or_two_bit1_preserves_bit1(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=brute.bit1)
        assert_bit1(a | b)

    def test_xor_two_bit1_preserves_bit1(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=brute.bit1)
        assert_bit1(a ^ b)

    def test_bool_ops_on_plain_bool_stay_bool(self):
        a = brute.tensor([True, False], dtype=torch.bool)
        b = brute.tensor([False, True], dtype=torch.bool)
        assert_plain_bool(a & b)

    def test_brute_bool_and_bit1_no_promotion(self):
        # brute.Tensor(bool) is non-bit1; mixing with bit1 falls back to plain bool.
        a = brute.tensor([True, False, True], dtype=brute.bit1)
        b = brute.tensor([True, True, False], dtype=torch.bool)
        assert_plain_bool(torch.logical_and(a, b))


# ════════════════════════════════════════════════════════════════════════════════
#   Section 2: Properties — dtype, shape, nbytes, itemsize, type(), etc.
# ════════════════════════════════════════════════════════════════════════════════

class TestProperties:
    """All Python-level Tensor attributes that may be affected by bit1 packing."""

    def test_dtype_bit1(self):
        assert brute.zeros(3, dtype=brute.bit1).dtype == brute.bit1

    def test_dtype_bit1_not_torch_bool(self):
        assert brute.zeros(3, dtype=brute.bit1).dtype != torch.bool

    def test_dtype_float(self):
        assert brute.zeros(3).dtype == torch.float32

    def test_dtype_bool_tensor(self):
        assert brute.tensor([True, False], dtype=torch.bool).dtype == torch.bool

    def test_shape(self):
        assert brute.zeros(3, 4, 5, dtype=brute.bit1).shape == torch.Size([3, 4, 5])

    def test_size_method(self):
        t = brute.zeros(3, 4, 5, dtype=brute.bit1)
        assert t.size() == torch.Size([3, 4, 5])
        assert t.size(0) == 3 and t.size(-1) == 5

    def test_numel(self):
        assert brute.zeros(3, 4, dtype=brute.bit1).numel() == 12

    def test_nelement(self):
        assert brute.zeros(3, 4, dtype=brute.bit1).nelement() == 12

    def test_ndim(self):
        assert brute.zeros(2, 3, dtype=brute.bit1).ndim == 2

    def test_dim_method(self):
        assert brute.zeros(2, 3, dtype=brute.bit1).dim() == 2

    def test_device_cpu(self):
        assert brute.zeros(3, dtype=brute.bit1).device.type == 'cpu'

    def test_is_cuda_false(self):
        assert brute.zeros(3, dtype=brute.bit1).is_cuda is False

    def test_isinstance_torch_tensor(self):
        t = brute.zeros(3, dtype=brute.bit1)
        assert isinstance(t, torch.Tensor) and isinstance(t, brute.Tensor)

    def test_is_floating_point_bit1(self):
        assert brute.zeros(3, dtype=brute.bit1).is_floating_point() is False

    def test_is_floating_point_float(self):
        assert brute.zeros(3, dtype=torch.float32).is_floating_point() is True

    def test_is_complex_bit1(self):
        assert brute.zeros(3, dtype=brute.bit1).is_complex() is False

    def test_is_signed_bit1(self):
        # bool storage is unsigned (False/True ∈ {0,1}).
        assert brute.zeros(3, dtype=brute.bit1).is_signed() is False

    def test_is_contiguous(self):
        assert brute.zeros(3, 4, dtype=brute.bit1).is_contiguous() is True

    def test_is_leaf(self):
        assert brute.zeros(3, dtype=brute.bit1).is_leaf is True

    def test_requires_grad_false_for_bit1(self):
        assert not brute.zeros(3, dtype=brute.bit1).requires_grad

    def test_stride(self):
        t = brute.zeros(3, 4, dtype=brute.bit1)
        assert t.stride() == (4, 1)

    def test_storage_offset(self):
        assert brute.zeros(4, dtype=brute.bit1).storage_offset() == 0

    def test_pack_dtype_attribute(self):
        t = brute.zeros(3, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t.pack_dtype is torch.uint8

    # ── nbytes ────────────────────────────────────────────────────────────────

    def test_nbytes_bit1_uint8(self):
        # 8 logical bits → 1 byte
        assert brute.zeros(8, dtype=brute.bit1, pack_dtype=brute.uint8).nbytes == 1

    def test_nbytes_bit1_uint32(self):
        # 32 bits → 4 bytes
        assert brute.zeros(32, dtype=brute.bit1, pack_dtype=brute.uint32).nbytes == 4

    def test_nbytes_bit1_uint64(self):
        # 64 bits → 8 bytes
        assert brute.zeros(64, dtype=brute.bit1, pack_dtype=brute.uint64).nbytes == 8

    def test_nbytes_bit1_non_multiple(self):
        # 9 bits → ceil(9/8) = 2 bytes
        assert brute.zeros(9, dtype=brute.bit1, pack_dtype=brute.uint8).nbytes == 2

    def test_nbytes_bit1_2d(self):
        # 3 rows × 8 cols (uint8) → 3 bytes
        assert brute.zeros(3, 8, dtype=brute.bit1, pack_dtype=brute.uint8).nbytes == 3

    def test_nbytes_float32(self):
        # numel × 4 bytes
        assert brute.zeros(4, dtype=torch.float32).nbytes == 16

    def test_nbytes_int64(self):
        assert brute.zeros(4, dtype=torch.int64).nbytes == 32

    def test_nbytes_plain_bool(self):
        assert brute.zeros(4, dtype=torch.bool).nbytes == 4

    # ── itemsize ──────────────────────────────────────────────────────────────

    def test_itemsize_bit1_raises(self):
        with pytest.raises(TypeError, match="itemsize"):
            _ = brute.zeros(8, dtype=brute.bit1).itemsize

    def test_itemsize_float32(self):
        assert brute.zeros(4, dtype=torch.float32).itemsize == 4

    def test_itemsize_int64(self):
        assert brute.zeros(4, dtype=torch.int64).itemsize == 8

    def test_itemsize_bool(self):
        assert brute.zeros(4, dtype=torch.bool).itemsize == 1

    # ── type() ────────────────────────────────────────────────────────────────

    def test_type_bit1_returns_brute_string(self):
        assert brute.zeros(3, dtype=brute.bit1).type() == 'brute.Bit1Tensor'

    def test_type_float32(self):
        # Plain torch type string for non-bit1.
        assert brute.zeros(3, dtype=torch.float32).type() == 'torch.FloatTensor'

    def test_type_int64(self):
        assert brute.zeros(3, dtype=torch.int64).type() == 'torch.LongTensor'

    def test_type_with_dtype_arg_casts(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        r = t.type('torch.FloatTensor')
        assert_brute(r, dtype=torch.float32)
        assert torch.equal(base(r), torch.tensor([1.0, 0.0, 1.0]))


# ════════════════════════════════════════════════════════════════════════════════
#   Section 3: In-place op staleness — the version-counter contract
# ════════════════════════════════════════════════════════════════════════════════

class TestInPlaceStaleness:
    """After any in-place modification to a bit1 tensor, _packed_buf must reflect
    the new bool state on next access. We verify this by repacking from scratch
    and comparing byte-for-byte."""

    @staticmethod
    def _packed_matches_bool(t: brute.Tensor):
        """Repack from current bool storage; assert it matches the lazy cache."""
        fresh = brute.tensor(base(t), dtype=brute.bit1, pack_dtype=t.pack_dtype)
        assert torch.equal(t._packed_buf, fresh._packed_buf), \
            f"packed_buf stale.\n  got: {t._packed_buf.tolist()}\n  ref: {fresh._packed_buf.tolist()}"

    def test_fill_true(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t.fill_(True)
        assert base(t).all()
        self._packed_matches_bool(t)

    def test_fill_false(self):
        t = brute.ones(8, dtype=brute.bit1)
        t.fill_(False)
        assert not base(t).any()
        self._packed_matches_bool(t)

    def test_zero_inplace(self):
        t = brute.ones(8, dtype=brute.bit1)
        t.zero_()
        assert not base(t).any()
        self._packed_matches_bool(t)

    def test_copy_from_bool(self):
        t = brute.zeros(8, dtype=brute.bit1)
        src = torch.tensor([True, False, True, True, False, True, False, True])
        t.copy_(src)
        assert torch.equal(base(t), src)
        self._packed_matches_bool(t)

    def test_copy_from_bit1(self):
        t = brute.zeros(8, dtype=brute.bit1)
        src = brute.ones(8, dtype=brute.bit1)
        t.copy_(src)
        assert base(t).all()
        self._packed_matches_bool(t)

    def test_setitem_scalar(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t[3] = True
        assert base(t)[3].item() is True
        assert not base(t)[0].item()
        self._packed_matches_bool(t)

    def test_setitem_slice(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t[2:6] = True
        expected = torch.tensor([False, False, True, True, True, True, False, False])
        assert torch.equal(base(t), expected)
        self._packed_matches_bool(t)

    def test_setitem_advanced(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t[[0, 2, 4]] = True
        expected = torch.tensor([True, False, True, False, True, False, False, False])
        assert torch.equal(base(t), expected)
        self._packed_matches_bool(t)

    def test_setitem_mask(self):
        t = brute.zeros(8, dtype=brute.bit1)
        mask = torch.tensor([True, False] * 4)
        t[mask] = True
        assert torch.equal(base(t), mask)
        self._packed_matches_bool(t)

    def test_logical_not_inplace(self):
        t = brute.tensor([True, False, True, False, True, False, True, False], dtype=brute.bit1)
        t.logical_not_()
        expected = torch.tensor([False, True, False, True, False, True, False, True])
        assert torch.equal(base(t), expected)
        self._packed_matches_bool(t)

    def test_logical_and_inplace(self):
        t = brute.tensor([True, True, False, False], dtype=brute.bit1)
        mask = torch.tensor([True, False, True, False])
        t.logical_and_(mask)
        assert torch.equal(base(t), torch.tensor([True, False, False, False]))
        self._packed_matches_bool(t)

    def test_logical_or_inplace(self):
        t = brute.zeros(4, dtype=brute.bit1)
        mask = torch.tensor([True, False, True, False])
        t.logical_or_(mask)
        assert torch.equal(base(t), mask)
        self._packed_matches_bool(t)

    def test_logical_xor_inplace(self):
        t = brute.tensor([True, True, False, False], dtype=brute.bit1)
        mask = torch.tensor([True, False, True, False])
        t.logical_xor_(mask)
        assert torch.equal(base(t), torch.tensor([False, True, True, False]))
        self._packed_matches_bool(t)

    def test_bitwise_not_inplace(self):
        t = brute.ones(8, dtype=brute.bit1)
        t.bitwise_not_()
        assert not base(t).any()
        self._packed_matches_bool(t)

    def test_bitwise_and_inplace(self):
        t = brute.tensor([True, True, False, False], dtype=brute.bit1)
        t.bitwise_and_(torch.tensor([True, False, True, False]))
        assert torch.equal(base(t), torch.tensor([True, False, False, False]))
        self._packed_matches_bool(t)

    def test_bitwise_or_inplace(self):
        t = brute.tensor([True, False, False, False], dtype=brute.bit1)
        t.bitwise_or_(torch.tensor([False, True, False, True]))
        assert torch.equal(base(t), torch.tensor([True, True, False, True]))
        self._packed_matches_bool(t)

    def test_bitwise_xor_inplace(self):
        t = brute.tensor([True, False, True, False], dtype=brute.bit1)
        t.bitwise_xor_(torch.tensor([True, True, False, False]))
        assert torch.equal(base(t), torch.tensor([False, True, True, False]))
        self._packed_matches_bool(t)

    def test_masked_fill_inplace(self):
        t = brute.zeros(8, dtype=brute.bit1)
        mask = torch.tensor([True, False] * 4)
        t.masked_fill_(mask, True)
        assert torch.equal(base(t), mask)
        self._packed_matches_bool(t)

    def test_index_fill_inplace(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t.index_fill_(0, torch.tensor([1, 3, 5]), True)
        expected = torch.tensor([False, True, False, True, False, True, False, False])
        assert torch.equal(base(t), expected)
        self._packed_matches_bool(t)

    def test_index_put_inplace(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t.index_put_((torch.tensor([0, 2, 4]),), torch.tensor(True))
        assert base(t)[0].item() and base(t)[2].item() and base(t)[4].item()
        self._packed_matches_bool(t)

    def test_scatter_inplace(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t.scatter_(0, torch.tensor([0, 2, 4]), torch.ones(3, dtype=torch.bool))
        self._packed_matches_bool(t)

    def test_multiple_inplace_cumulative(self):
        t = brute.zeros(8, dtype=brute.bit1)
        t.fill_(True)
        t.logical_not_()
        t.fill_(True)
        t[0] = False
        assert torch.equal(base(t), torch.tensor([False, True, True, True, True, True, True, True]))
        self._packed_matches_bool(t)

    def test_matmul_after_fill_uses_fresh_packed_buf(self):
        """Critical: matmul must read the post-fill packed buffer, not a stale one."""
        K = 16
        t = brute.zeros(2, K, dtype=brute.bit1)
        t.fill_(True)
        b = brute.ones(3, K, dtype=brute.bit1)
        # all True XNOR all True → +1 contribution × K → result == K
        result = t @ b
        assert (result == K).all(), f"Expected all {K}, got {result}"

    def test_matmul_after_setitem_uses_fresh_packed_buf(self):
        K = 16
        a = brute.ones(2, K, dtype=brute.bit1)
        b = brute.ones(3, K, dtype=brute.bit1)
        a[0, 0] = False  # flip a single bit
        # Before: (True XNOR True)*K = +K for all rows.
        # After: row 0 differs in 1 position → +1 fewer matches, +1 mismatch → K - 2.
        r = a @ b
        assert (r[0] == K - 2).all()
        assert (r[1] == K).all()

    def test_popcount_after_fill(self):
        t = brute.zeros(32, dtype=brute.bit1)
        t.fill_(True)
        assert t.popcount().item() == 32

    def test_popcount_after_setitem(self):
        t = brute.zeros(16, dtype=brute.bit1)
        t[0] = True
        t[7] = True
        t[15] = True
        assert t.popcount().item() == 3

    def test_unpack_pm1_after_inplace(self):
        t = brute.zeros(4, dtype=brute.bit1)
        t.fill_(True)
        assert torch.equal(t.unpack_pm1(), torch.ones(4))

    def test_packed_buf_version_increments_on_inplace(self):
        t = brute.zeros(8, dtype=brute.bit1)
        pb_before = t._packed_buf.clone()
        t[0] = True
        pb_after = t._packed_buf
        assert not torch.equal(pb_before, pb_after)


# ════════════════════════════════════════════════════════════════════════════════
#   Section 4: Tensor.new_* factory methods
# ════════════════════════════════════════════════════════════════════════════════

class TestNewFactoryMethods:
    """Tensor.new_zeros / new_ones / new_empty / new_full / new_tensor preserve bit1."""

    @pytest.fixture
    def t_bit1(self):
        return brute.tensor([True, False, True], dtype=brute.bit1, pack_dtype=brute.uint32)

    @pytest.fixture
    def t_float(self):
        return brute.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    @pytest.fixture
    def t_bool(self):
        return brute.tensor([True, False, True], dtype=torch.bool)

    # ── new_zeros ─────────────────────────────────────────────────────────────

    def test_new_zeros_from_bit1(self, t_bit1):
        r = t_bit1.new_zeros([5])
        assert_bit1(r)
        assert r.shape == torch.Size([5])
        assert not base(r).any()

    def test_new_zeros_inherits_pack_dtype(self, t_bit1):
        r = t_bit1.new_zeros([8])
        assert r.pack_dtype is torch.uint32

    def test_new_zeros_from_float(self, t_float):
        r = t_float.new_zeros([4])
        assert_brute(r, dtype=torch.float32)
        assert torch.equal(base(r), torch.zeros(4))

    def test_new_zeros_explicit_bit1(self, t_float):
        r = t_float.new_zeros([4], dtype=brute.bit1)
        assert_bit1(r)
        assert not base(r).any()

    def test_new_zeros_explicit_float_from_bit1(self, t_bit1):
        r = t_bit1.new_zeros([4], dtype=torch.float32)
        assert_brute(r, dtype=torch.float32)

    def test_new_zeros_2d(self, t_bit1):
        r = t_bit1.new_zeros([3, 4])
        assert_bit1(r)
        assert r.shape == torch.Size([3, 4])

    # ── new_ones ──────────────────────────────────────────────────────────────

    def test_new_ones_from_bit1(self, t_bit1):
        r = t_bit1.new_ones([5])
        assert_bit1(r)
        assert base(r).all()

    def test_new_ones_inherits_pack_dtype(self, t_bit1):
        assert t_bit1.new_ones([4]).pack_dtype is torch.uint32

    def test_new_ones_from_float(self, t_float):
        r = t_float.new_ones([4])
        assert_brute(r, dtype=torch.float32)
        assert torch.equal(base(r), torch.ones(4))

    def test_new_ones_explicit_bit1(self, t_float):
        r = t_float.new_ones([4], dtype=brute.bit1)
        assert_bit1(r)
        assert base(r).all()

    # ── new_empty ─────────────────────────────────────────────────────────────

    def test_new_empty_from_bit1_shape(self, t_bit1):
        r = t_bit1.new_empty([5])
        assert_bit1(r)
        assert r.shape == torch.Size([5])

    def test_new_empty_inherits_pack_dtype(self, t_bit1):
        assert t_bit1.new_empty([4]).pack_dtype is torch.uint32

    def test_new_empty_from_float(self, t_float):
        r = t_float.new_empty([4])
        assert_brute(r, dtype=torch.float32)
        assert r.shape == torch.Size([4])

    def test_new_empty_explicit_bit1(self, t_float):
        r = t_float.new_empty([4], dtype=brute.bit1)
        assert_bit1(r)

    # ── new_full ──────────────────────────────────────────────────────────────

    def test_new_full_from_bit1_true(self, t_bit1):
        r = t_bit1.new_full([4], True)
        assert_bit1(r)
        assert base(r).all()

    def test_new_full_from_bit1_false(self, t_bit1):
        r = t_bit1.new_full([4], False)
        assert_bit1(r)
        assert not base(r).any()

    def test_new_full_from_float(self, t_float):
        r = t_float.new_full([4], 3.5)
        assert_brute(r, dtype=torch.float32)
        assert (base(r) == 3.5).all()

    def test_new_full_explicit_bit1(self, t_float):
        r = t_float.new_full([4], True, dtype=brute.bit1)
        assert_bit1(r)
        assert base(r).all()

    # ── new_tensor ────────────────────────────────────────────────────────────

    def test_new_tensor_from_bit1(self, t_bit1):
        r = t_bit1.new_tensor([True, False, True, False])
        assert_bit1(r)
        assert torch.equal(base(r), torch.tensor([True, False, True, False]))

    def test_new_tensor_inherits_pack_dtype(self, t_bit1):
        assert t_bit1.new_tensor([True, False]).pack_dtype is torch.uint32

    def test_new_tensor_from_float(self, t_float):
        r = t_float.new_tensor([1.0, 2.0, 3.0])
        assert_brute(r, dtype=torch.float32)
        assert torch.equal(base(r), torch.tensor([1.0, 2.0, 3.0]))

    def test_new_tensor_explicit_bit1(self, t_float):
        r = t_float.new_tensor([True, False, True], dtype=brute.bit1)
        assert_bit1(r)


# ════════════════════════════════════════════════════════════════════════════════
#   Section 5: *_like factories (zeros_like, ones_like, full_like, empty_like, *_like rand)
# ════════════════════════════════════════════════════════════════════════════════

class TestLikeFactories:
    """*_like creators preserve dtype/pack_dtype of the source tensor by default."""

    def test_zeros_like_bit1(self):
        src = brute.ones(3, 4, dtype=brute.bit1)
        r = brute.zeros_like(src)
        assert_bit1(r)
        assert not base(r).any()

    def test_zeros_like_preserves_pack_dtype(self):
        src = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint32)
        assert brute.zeros_like(src).pack_dtype is torch.uint32

    def test_zeros_like_float(self):
        src = brute.randn(3, 4)
        r = brute.zeros_like(src)
        assert_brute(r, dtype=torch.float32)

    def test_ones_like_bit1(self):
        src = brute.zeros(3, 4, dtype=brute.bit1)
        r = brute.ones_like(src)
        assert_bit1(r)
        assert base(r).all()

    def test_ones_like_preserves_pack_dtype(self):
        src = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint64)
        assert brute.ones_like(src).pack_dtype is torch.uint64

    def test_full_like_bit1_true(self):
        src = brute.zeros(3, 4, dtype=brute.bit1)
        r = brute.full_like(src, True)
        assert_bit1(r)
        assert base(r).all()

    def test_full_like_bit1_false(self):
        src = brute.ones(3, 4, dtype=brute.bit1)
        r = brute.full_like(src, False)
        assert_bit1(r)
        assert not base(r).any()

    def test_full_like_float(self):
        src = brute.zeros(3, 4)
        r = brute.full_like(src, 2.5)
        assert_brute(r, dtype=torch.float32)
        assert (base(r) == 2.5).all()

    def test_empty_like_bit1(self):
        src = brute.zeros(3, 4, dtype=brute.bit1)
        r = brute.empty_like(src)
        assert_bit1(r)
        assert r.shape == src.shape

    def test_empty_like_float(self):
        src = brute.randn(3, 4)
        r = brute.empty_like(src)
        assert_brute(r, dtype=torch.float32)

    def test_rand_like_bit1(self):
        src = brute.zeros(100, dtype=brute.bit1)
        r = brute.rand_like(src)
        assert_bit1(r)

    def test_rand_like_float(self):
        src = brute.zeros(10)
        r = brute.rand_like(src)
        assert_brute(r, dtype=torch.float32)
        assert (base(r) >= 0).all() and (base(r) <= 1).all()

    def test_randn_like_bit1(self):
        src = brute.zeros(100, dtype=brute.bit1)
        r = brute.randn_like(src)
        assert_bit1(r)

    def test_torch_zeros_like_bit1_via_dispatch(self):
        # torch.zeros_like(bit1_t) goes through __torch_function__.
        src = brute.ones(8, dtype=brute.bit1)
        r = torch.zeros_like(src)
        assert_bit1(r)
        assert not base(r).any()

    def test_torch_ones_like_bit1_via_dispatch(self):
        src = brute.zeros(8, dtype=brute.bit1)
        r = torch.ones_like(src)
        assert_bit1(r)
        assert base(r).all()

    def test_torch_empty_like_bit1_via_dispatch(self):
        src = brute.zeros(8, dtype=brute.bit1)
        r = torch.empty_like(src)
        assert_bit1(r)


# ════════════════════════════════════════════════════════════════════════════════
#   Section 6: Module-level factories
# ════════════════════════════════════════════════════════════════════════════════

class TestFactories:

    def test_zeros_bit1_values(self):
        t   = brute.zeros(3, 4, dtype=brute.bit1)
        ref = torch.zeros(3, 4, dtype=torch.bool)
        assert_bit1(t)
        assert t.shape == torch.Size([3, 4])
        assert torch.equal(base(t), ref)

    def test_zeros_bit1_tuple_size(self):
        assert brute.zeros((2, 5), dtype=brute.bit1).shape == torch.Size([2, 5])

    def test_zeros_bit1_list_size(self):
        assert brute.zeros([2, 5], dtype=brute.bit1).shape == torch.Size([2, 5])

    def test_zeros_float(self):
        assert torch.equal(base(brute.zeros(3, 4)), torch.zeros(3, 4))

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_zeros_bit1_pack_dtype(self, pw):
        assert brute.zeros(4, dtype=brute.bit1, pack_dtype=pw).pack_dtype is pw

    def test_ones_bit1_values(self):
        assert torch.equal(base(brute.ones(3, 4, dtype=brute.bit1)),
                           torch.ones(3, 4, dtype=torch.bool))

    def test_ones_float(self):
        assert torch.equal(base(brute.ones(3, 4)), torch.ones(3, 4))

    def test_full_bit1_true(self):
        assert torch.equal(base(brute.full((2, 3), True, dtype=brute.bit1)),
                           torch.full((2, 3), True, dtype=torch.bool))

    def test_full_bit1_false(self):
        assert torch.equal(base(brute.full((2, 3), False, dtype=brute.bit1)),
                           torch.full((2, 3), False, dtype=torch.bool))

    def test_full_bit1_int_truthy(self):
        # bool(5) = True
        assert base(brute.full((2,), 5, dtype=brute.bit1)).all()

    def test_full_bit1_int_zero(self):
        assert not base(brute.full((2,), 0, dtype=brute.bit1)).any()

    def test_tensor_bit1(self):
        data = [True, False, True, True, False]
        t = brute.tensor(data, dtype=brute.bit1)
        assert_bit1(t)
        assert torch.equal(base(t), torch.tensor(data, dtype=torch.bool))

    def test_tensor_float(self):
        data = [1.0, 2.0, 3.0]
        assert torch.equal(base(brute.tensor(data)), torch.tensor(data))

    def test_tensor_bool_list_defaults_to_bool(self):
        t = brute.tensor([True, False])
        assert t.dtype == torch.bool and not t._is_bit1

    def test_tensor_from_tensor_bit1(self):
        src = torch.tensor([True, False, True])
        t = brute.tensor(src, dtype=brute.bit1)
        assert_bit1(t)
        assert torch.equal(base(t), src)

    def test_as_tensor_bool_to_bit1(self):
        raw = torch.tensor([True, False, True])
        t = brute.as_tensor(raw, dtype=brute.bit1)
        assert_bit1(t)
        assert torch.equal(base(t), raw)

    def test_as_tensor_list_bit1(self):
        t = brute.as_tensor([True, False], dtype=brute.bit1)
        assert torch.equal(base(t), torch.tensor([True, False], dtype=torch.bool))

    def test_rand_bit1_shape(self):
        t = brute.rand(4, 5, dtype=brute.bit1)
        assert_bit1(t)
        assert t.shape == torch.Size([4, 5])

    def test_randn_bit1(self):
        assert_bit1(brute.randn(4, 5, dtype=brute.bit1))

    def test_rand_bit1_has_both_values(self):
        torch.manual_seed(0)
        t = brute.rand(1000, dtype=brute.bit1)
        assert base(t).any().item() and (~base(t)).any().item()

    def test_rand_float_range(self):
        t = brute.rand(100)
        assert (base(t) >= 0).all() and (base(t) <= 1).all()

    def test_randn_float(self):
        assert brute.randn(50).dtype == torch.float32

    def test_randint_bit1(self):
        t = brute.randint(0, 2, size=(4, 5), dtype=brute.bit1)
        assert_bit1(t)
        assert t.shape == torch.Size([4, 5])

    def test_arange(self):
        assert torch.equal(base(brute.arange(5)), torch.arange(5))

    def test_arange_returns_brute(self):
        assert_brute(brute.arange(5))

    def test_linspace(self):
        assert torch.allclose(base(brute.linspace(0.0, 1.0, 5)),
                              torch.linspace(0.0, 1.0, 5))

    def test_linspace_returns_brute(self):
        assert_brute(brute.linspace(0.0, 1.0, 5))

    def test_eye(self):
        assert torch.equal(base(brute.eye(4)), torch.eye(4))

    def test_eye_returns_brute(self):
        assert_brute(brute.eye(4))

    def test_from_numpy(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = brute.from_numpy(arr)
        assert_brute(t)
        assert torch.equal(base(t), torch.from_numpy(arr))


# ════════════════════════════════════════════════════════════════════════════════
#   Section 7: Conversion — bool(), to(), float(), int(), long(), etc.
# ════════════════════════════════════════════════════════════════════════════════

class TestBoolMethod:
    """.bool() returns a brute.Tensor with dtype=torch.bool (NOT bit1)."""

    def test_bit1_bool_returns_plain_bool(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        b = t.bool()
        assert_plain_bool(b)

    def test_bit1_bool_values(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        assert torch.equal(base(t.bool()), torch.tensor([True, False, True]))

    def test_float_bool(self):
        b = brute.tensor([1.5, -0.5, 0.1]).bool()
        assert_brute(b, dtype=torch.bool)

    def test_bool_tensor_bool_is_noop(self):
        t = brute.tensor([True, False], dtype=torch.bool)
        b = t.bool()
        assert_brute(b, dtype=torch.bool)
        assert torch.equal(base(b), torch.tensor([True, False]))

    def test_bit1_bool_then_bit1_roundtrip(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        t2 = brute.as_tensor(t.bool(), dtype=brute.bit1)
        assert_bit1(t2)
        assert torch.equal(base(t2), base(t))


class TestElementSize:
    def test_bit1_raises(self):
        with pytest.raises(TypeError, match="element_size"):
            brute.zeros(4, dtype=brute.bit1).element_size()

    @pytest.mark.parametrize("dtype,size", [
        (torch.float32, 4), (torch.float64, 8), (torch.int32, 4), (torch.int64, 8),
        (torch.bool, 1), (torch.int8, 1), (torch.int16, 2),
    ])
    def test_non_bit1(self, dtype, size):
        assert brute.zeros(4, dtype=dtype).element_size() == size

    def test_packed_buf_nbytes_uint8(self):
        assert brute.zeros(8, dtype=brute.bit1, pack_dtype=brute.uint8)._packed_buf.nbytes == 1

    def test_packed_buf_nbytes_uint32(self):
        assert brute.zeros(32, dtype=brute.bit1, pack_dtype=brute.uint32)._packed_buf.nbytes == 4

    def test_packed_buf_nbytes_non_multiple(self):
        assert brute.zeros(9, dtype=brute.bit1, pack_dtype=brute.uint8)._packed_buf.nbytes == 2


class TestConversion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.t = brute.tensor([True, False, True], dtype=brute.bit1)

    def test_bool_method_returns_plain_bool(self):
        assert_plain_bool(self.t.bool())

    def test_to_float32(self):
        r = self.t.float()
        assert_brute(r, dtype=torch.float32)
        assert torch.equal(base(r), torch.tensor([1.0, 0.0, 1.0]))

    def test_to_float64(self):
        r = self.t.double()
        assert_brute(r, dtype=torch.float64)
        assert torch.equal(base(r), torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))

    def test_to_int(self):
        r = self.t.int()
        assert_brute(r, dtype=torch.int32)
        assert torch.equal(base(r), torch.tensor([1, 0, 1], dtype=torch.int32))

    def test_to_long(self):
        r = self.t.long()
        assert_brute(r, dtype=torch.int64)
        assert torch.equal(base(r), torch.tensor([1, 0, 1], dtype=torch.int64))

    def test_to_short(self):
        r = self.t.short()
        assert_brute(r, dtype=torch.int16)

    def test_to_byte(self):
        r = self.t.byte()
        assert_brute(r, dtype=torch.uint8)

    def test_to_char(self):
        r = self.t.char()
        assert_brute(r, dtype=torch.int8)

    def test_to_half(self):
        r = self.t.half()
        assert_brute(r, dtype=torch.float16)

    def test_to_bfloat16(self):
        r = self.t.bfloat16()
        assert_brute(r, dtype=torch.bfloat16)

    def test_unpack_pm1(self):
        assert torch.equal(self.t.unpack_pm1(), torch.tensor([1.0, -1.0, 1.0]))

    def test_float_to_bit1_via_to(self):
        t = brute.tensor([1.5, -0.5, 0.1], dtype=torch.float32).to(brute.bit1)
        assert_bit1(t)
        assert torch.equal(base(t), torch.tensor([True, False, True]))

    def test_zero_float_to_bit1_is_false(self):
        assert base(brute.tensor([0.0]).to(brute.bit1))[0].item() is False

    def test_negative_float_to_bit1_is_false(self):
        assert not base(brute.tensor([-1.0, -1e6, -0.001]).to(brute.bit1)).any()

    def test_positive_float_to_bit1_is_true(self):
        assert base(brute.tensor([0.001, 1.0, 1e6]).to(brute.bit1)).all()

    def test_int_nonzero_to_bit1_is_true(self):
        assert base(brute.tensor([1, 2, -1, 100], dtype=torch.int32).to(brute.bit1)).all()

    def test_int_zero_to_bit1_is_false(self):
        assert base(brute.tensor([0], dtype=torch.int32).to(brute.bit1))[0].item() is False

    def test_to_device_preserves_bit1(self):
        t2 = self.t.to('cpu')
        assert_bit1(t2)
        assert torch.equal(base(t2), base(self.t))

    def test_to_pack_dtype_uint32(self):
        t = brute.tensor([True, False, True, False], dtype=brute.bit1, pack_dtype=brute.uint8)
        t2 = t.to(brute.bit1, pack_dtype=brute.uint32)
        assert t2.pack_dtype is torch.uint32
        assert torch.equal(base(t2), base(t))

    def test_float_round_trip(self):
        t_bit = brute.tensor([True, False, True], dtype=brute.bit1)
        t_back = t_bit.float().to(brute.bit1)
        assert torch.equal(base(t_back), base(t_bit))

    def test_to_kwargs_only(self):
        r = self.t.to(dtype=torch.float32)
        assert_brute(r, dtype=torch.float32)


class TestTypeMethod:
    """Tensor.type() — returns string or casts to dtype."""

    def test_bit1_returns_brute_bit1_string(self):
        assert brute.zeros(3, dtype=brute.bit1).type() == 'brute.Bit1Tensor'

    def test_float32_returns_torch_string(self):
        assert brute.zeros(3, dtype=torch.float32).type() == 'torch.FloatTensor'

    def test_int64_returns_torch_string(self):
        assert brute.zeros(3, dtype=torch.int64).type() == 'torch.LongTensor'

    def test_bool_returns_torch_string(self):
        assert brute.zeros(3, dtype=torch.bool).type() == 'torch.BoolTensor'

    def test_cast_to_float_via_type(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        r = t.type('torch.FloatTensor')
        assert_brute(r, dtype=torch.float32)
        assert torch.equal(base(r), torch.tensor([1.0, 0.0, 1.0]))

    def test_cast_to_int_via_type(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        r = t.type('torch.LongTensor')
        assert_brute(r, dtype=torch.int64)


# ════════════════════════════════════════════════════════════════════════════════
#   Section 8: Popcount and padding edges
# ════════════════════════════════════════════════════════════════════════════════

class TestPopcount:

    def test_bit1_all_true(self):
        assert brute.ones(8, dtype=brute.bit1).popcount().item() == 8

    def test_bit1_all_false(self):
        assert brute.zeros(8, dtype=brute.bit1).popcount().item() == 0

    def test_bit1_mixed(self):
        assert brute.tensor([True, False, True, False], dtype=brute.bit1).popcount().item() == 2

    def test_bit1_2d(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert t.popcount().item() == 2

    def test_bit1_returns_brute_tensor(self):
        assert_brute(brute.ones(4, dtype=brute.bit1).popcount())

    def test_bit1_returns_int64_scalar(self):
        r = brute.ones(4, dtype=brute.bit1).popcount()
        assert r.dtype == torch.int64 and r.dim() == 0

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_bit1_all_packs(self, pw):
        data = [True, False, True, True, False, True, False, True]
        t = brute.tensor(data, dtype=brute.bit1, pack_dtype=pw)
        assert t.popcount().item() == sum(data)

    def test_bit1_non_multiple_k(self):
        t = brute.tensor([True] * 9, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t.popcount().item() == 9

    def test_bit1_k65_no_padding_inflation(self):
        t = brute.tensor([True] * 65, dtype=brute.bit1, pack_dtype=brute.uint64)
        assert t.popcount().item() == 65

    def test_bool_all_true(self):
        assert brute.ones(6, dtype=torch.bool).popcount().item() == 6

    def test_bool_all_false(self):
        assert brute.zeros(6, dtype=torch.bool).popcount().item() == 0

    def test_bool_mixed(self):
        assert brute.tensor([True, False, True], dtype=torch.bool).popcount().item() == 2

    def test_bool_returns_brute_tensor(self):
        assert_brute(brute.ones(4, dtype=torch.bool).popcount())

    def test_float_raises(self):
        with pytest.raises(TypeError):
            brute.zeros(4, dtype=torch.float32).popcount()

    def test_popcount_matches_sum(self):
        torch.manual_seed(42)
        data = torch.randint(0, 2, (64,)).bool()
        t = brute.tensor(data, dtype=brute.bit1)
        assert t.popcount().item() == data.sum().item()

    def test_empty_bit1_popcount_is_zero(self):
        assert brute.zeros(0, dtype=brute.bit1).popcount().item() == 0


class TestPaddingEdgeCases:

    @pytest.mark.parametrize("K,pw", [
        (1, brute.uint8), (7, brute.uint8), (8, brute.uint8), (9, brute.uint8),
        (31, brute.uint32), (32, brute.uint32), (33, brute.uint32),
        (63, brute.uint64), (64, brute.uint64), (65, brute.uint64),
    ])
    def test_matmul_all_ones(self, K, pw):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=pw)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=pw)
        assert (a @ b == K).all()

    @pytest.mark.parametrize("K,pw", [
        (7, brute.uint8), (9, brute.uint8),
        (31, brute.uint32), (33, brute.uint32),
        (63, brute.uint64), (65, brute.uint64),
    ])
    def test_matmul_vs_float_reference(self, K, pw):
        torch.manual_seed(K)
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=pw)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=pw)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

    @pytest.mark.parametrize("K,pw", [
        (7, brute.uint8), (9, brute.uint8),
        (31, brute.uint32), (33, brute.uint32),
        (63, brute.uint64), (65, brute.uint64),
    ])
    def test_unpack_pm1_non_multiple_k(self, K, pw):
        t = brute.ones(K, dtype=brute.bit1, pack_dtype=pw)
        r = t.unpack_pm1()
        assert r.shape == torch.Size([K]) and (r == 1.0).all()

    @pytest.mark.parametrize("K,pw", [
        (7, brute.uint8), (9, brute.uint8),
        (31, brute.uint32), (33, brute.uint32),
        (63, brute.uint64), (65, brute.uint64),
    ])
    def test_popcount_non_multiple_k_all_true(self, K, pw):
        assert brute.ones(K, dtype=brute.bit1, pack_dtype=pw).popcount().item() == K

    def test_packed_buf_trailing_zeros(self):
        t = brute.tensor([True, True, True], dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t._packed_buf[0].item() == 7

    def test_packed_buf_all_false_trailing_zero(self):
        t = brute.tensor([False, False, False], dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t._packed_buf[0].item() == 0


# ════════════════════════════════════════════════════════════════════════════════
#   Section 9: Systematic bit1↔bool parity for every op category
# ════════════════════════════════════════════════════════════════════════════════

class TestParityLogical:
    """Logical and bitwise ops on bit1 must produce identical values to bool."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a_bit1, self.a_bool = make_pair(DATA_1D_A)
        self.b_bit1, self.b_bool = make_pair(DATA_1D_B)

    def test_logical_and(self):
        assert_parity(torch.logical_and(self.a_bit1, self.b_bit1),
                      torch.logical_and(self.a_bool, self.b_bool))

    def test_logical_or(self):
        assert_parity(torch.logical_or(self.a_bit1, self.b_bit1),
                      torch.logical_or(self.a_bool, self.b_bool))

    def test_logical_xor(self):
        assert_parity(torch.logical_xor(self.a_bit1, self.b_bit1),
                      torch.logical_xor(self.a_bool, self.b_bool))

    def test_logical_not(self):
        assert_parity(torch.logical_not(self.a_bit1),
                      torch.logical_not(self.a_bool))

    def test_bitwise_and(self):
        assert_parity(torch.bitwise_and(self.a_bit1, self.b_bit1),
                      torch.bitwise_and(self.a_bool, self.b_bool))

    def test_bitwise_or(self):
        assert_parity(torch.bitwise_or(self.a_bit1, self.b_bit1),
                      torch.bitwise_or(self.a_bool, self.b_bool))

    def test_bitwise_xor(self):
        assert_parity(torch.bitwise_xor(self.a_bit1, self.b_bit1),
                      torch.bitwise_xor(self.a_bool, self.b_bool))

    def test_bitwise_not(self):
        assert_parity(torch.bitwise_not(self.a_bit1),
                      torch.bitwise_not(self.a_bool))

    def test_operator_and(self):
        assert_parity(self.a_bit1 & self.b_bit1, self.a_bool & self.b_bool)

    def test_operator_or(self):
        assert_parity(self.a_bit1 | self.b_bit1, self.a_bool | self.b_bool)

    def test_operator_xor(self):
        assert_parity(self.a_bit1 ^ self.b_bit1, self.a_bool ^ self.b_bool)

    def test_operator_invert(self):
        assert_parity(~self.a_bit1, ~self.a_bool)

    # Method-form on the tensor
    def test_method_logical_and(self):
        assert_parity(self.a_bit1.logical_and(self.b_bit1),
                      self.a_bool.logical_and(self.b_bool))

    def test_method_logical_or(self):
        assert_parity(self.a_bit1.logical_or(self.b_bit1),
                      self.a_bool.logical_or(self.b_bool))

    def test_method_logical_xor(self):
        assert_parity(self.a_bit1.logical_xor(self.b_bit1),
                      self.a_bool.logical_xor(self.b_bool))

    def test_method_logical_not(self):
        assert_parity(self.a_bit1.logical_not(), self.a_bool.logical_not())


class TestParityComparison:
    """Comparison ops on bit1 produce bit1 results matching bool comparisons."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a_bit1, self.a_bool = make_pair(DATA_1D_A)
        self.b_bit1, self.b_bool = make_pair(DATA_1D_B)

    def test_eq(self):
        assert_parity(torch.eq(self.a_bit1, self.b_bit1),
                      torch.eq(self.a_bool, self.b_bool))

    def test_ne(self):
        assert_parity(torch.ne(self.a_bit1, self.b_bit1),
                      torch.ne(self.a_bool, self.b_bool))

    def test_lt(self):
        assert_parity(torch.lt(self.a_bit1, self.b_bit1),
                      torch.lt(self.a_bool, self.b_bool))

    def test_le(self):
        assert_parity(torch.le(self.a_bit1, self.b_bit1),
                      torch.le(self.a_bool, self.b_bool))

    def test_gt(self):
        assert_parity(torch.gt(self.a_bit1, self.b_bit1),
                      torch.gt(self.a_bool, self.b_bool))

    def test_ge(self):
        assert_parity(torch.ge(self.a_bit1, self.b_bit1),
                      torch.ge(self.a_bool, self.b_bool))

    def test_eq_operator(self):
        assert_parity(self.a_bit1 == self.b_bit1, self.a_bool == self.b_bool)

    def test_ne_operator(self):
        assert_parity(self.a_bit1 != self.b_bit1, self.a_bool != self.b_bool)

    def test_lt_operator(self):
        assert_parity(self.a_bit1 < self.b_bit1, self.a_bool < self.b_bool)

    def test_gt_operator(self):
        assert_parity(self.a_bit1 > self.b_bit1, self.a_bool > self.b_bool)

    def test_le_operator(self):
        assert_parity(self.a_bit1 <= self.b_bit1, self.a_bool <= self.b_bool)

    def test_ge_operator(self):
        assert_parity(self.a_bit1 >= self.b_bit1, self.a_bool >= self.b_bool)

    def test_equal_scalar(self):
        # torch.equal returns a Python bool, not a tensor.
        assert torch.equal(self.a_bit1, self.a_bit1) is True
        assert torch.equal(self.a_bit1, self.b_bit1) is False


class TestParityReductions:
    """Reductions on bit1 match bool reductions in value and shape."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t1_bit1, self.t1_bool = make_pair(DATA_1D_A)
        self.t2_bit1, self.t2_bool = make_pair(DATA_2D_A)

    def test_all_global(self):
        r_bit1 = torch.all(self.t1_bit1)
        r_bool = torch.all(self.t1_bool)
        assert_brute(r_bit1)
        assert r_bit1.item() == r_bool.item()

    def test_any_global(self):
        r_bit1 = torch.any(self.t1_bit1)
        r_bool = torch.any(self.t1_bool)
        assert_brute(r_bit1)
        assert r_bit1.item() == r_bool.item()

    def test_all_dim(self):
        r_bit1 = torch.all(self.t2_bit1, dim=0)
        r_bool = torch.all(self.t2_bool, dim=0)
        assert_brute(r_bit1)
        assert torch.equal(base(r_bit1), r_bool)

    def test_any_dim(self):
        r_bit1 = torch.any(self.t2_bit1, dim=1)
        r_bool = torch.any(self.t2_bool, dim=1)
        assert_brute(r_bit1)
        assert torch.equal(base(r_bit1), r_bool)

    def test_sum(self):
        assert torch.sum(self.t1_bit1).item() == torch.sum(self.t1_bool).item()

    def test_sum_dim(self):
        r_bit1 = torch.sum(self.t2_bit1, dim=0)
        r_bool = torch.sum(self.t2_bool, dim=0)
        assert_brute(r_bit1)
        assert torch.equal(base(r_bit1), r_bool)

    def test_count_nonzero(self):
        assert torch.count_nonzero(self.t1_bit1).item() == torch.count_nonzero(self.t1_bool).item()

    def test_count_nonzero_dim(self):
        r_bit1 = torch.count_nonzero(self.t2_bit1, dim=0)
        r_bool = torch.count_nonzero(self.t2_bool, dim=0)
        assert torch.equal(base(r_bit1), r_bool)

    def test_nonzero(self):
        r_bit1 = torch.nonzero(self.t1_bit1)
        r_bool = torch.nonzero(self.t1_bool)
        assert_brute(r_bit1)
        assert torch.equal(base(r_bit1), r_bool)

    def test_argwhere(self):
        r_bit1 = torch.argwhere(self.t1_bit1)
        r_bool = torch.argwhere(self.t1_bool)
        assert torch.equal(base(r_bit1), r_bool)

    def test_max_global(self):
        # max of bool returns bool
        assert torch.max(self.t1_bit1).item() == torch.max(self.t1_bool).item()

    def test_min_global(self):
        assert torch.min(self.t1_bit1).item() == torch.min(self.t1_bool).item()

    def test_argmax_raises_on_bool(self):
        # PyTorch doesn't support argmax on bool; bit1 follows the same rule.
        with pytest.raises(RuntimeError):
            torch.argmax(self.t1_bit1)

    def test_argmin_raises_on_bool(self):
        with pytest.raises(RuntimeError):
            torch.argmin(self.t1_bit1)

    def test_argmax_after_cast(self):
        # Cast to int first → argmax works and stays in brute.
        r = torch.argmax(self.t1_bit1.long())
        assert_brute(r)
        assert r.item() == torch.argmax(self.t1_bool.long()).item()

    def test_amax(self):
        # bool tensors → amax requires non-bool typically; use a comparison
        # torch.amax does support bool
        assert torch.amax(self.t1_bit1).item() == torch.amax(self.t1_bool).item()

    def test_amin(self):
        assert torch.amin(self.t1_bit1).item() == torch.amin(self.t1_bool).item()


class TestParityShape:
    """Shape ops on bit1 produce bit1 results structurally identical to bool ops."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t1_bit1, self.t1_bool = make_pair(DATA_1D_A)
        self.t2_bit1, self.t2_bool = make_pair(DATA_2D_A)
        self.t3_bit1, self.t3_bool = make_pair(DATA_3D_A)

    def test_view(self):
        assert_parity(self.t1_bit1.view(2, 4), self.t1_bool.view(2, 4))

    def test_reshape(self):
        assert_parity(self.t1_bit1.reshape(2, 4), self.t1_bool.reshape(2, 4))

    def test_reshape_neg1(self):
        assert_parity(self.t1_bit1.reshape(-1, 2), self.t1_bool.reshape(-1, 2))

    def test_flatten(self):
        assert_parity(self.t2_bit1.flatten(), self.t2_bool.flatten())

    def test_flatten_partial(self):
        assert_parity(self.t3_bit1.flatten(0, 1), self.t3_bool.flatten(0, 1))

    def test_squeeze_all(self):
        a_bit1 = brute.zeros(1, 4, 1, dtype=brute.bit1)
        a_bool = torch.zeros(1, 4, 1, dtype=torch.bool)
        assert_parity(a_bit1.squeeze(), a_bool.squeeze())

    def test_squeeze_dim(self):
        a_bit1 = brute.zeros(1, 4, 1, dtype=brute.bit1)
        a_bool = torch.zeros(1, 4, 1, dtype=torch.bool)
        assert_parity(a_bit1.squeeze(0), a_bool.squeeze(0))

    def test_unsqueeze(self):
        assert_parity(self.t1_bit1.unsqueeze(0), self.t1_bool.unsqueeze(0))

    def test_unsqueeze_neg(self):
        assert_parity(self.t1_bit1.unsqueeze(-1), self.t1_bool.unsqueeze(-1))

    def test_t_2d(self):
        assert_parity(self.t2_bit1.t(), self.t2_bool.t())

    def test_transpose(self):
        assert_parity(self.t2_bit1.transpose(0, 1), self.t2_bool.transpose(0, 1))

    def test_permute_2d(self):
        assert_parity(self.t2_bit1.permute(1, 0), self.t2_bool.permute(1, 0))

    def test_permute_3d(self):
        assert_parity(self.t3_bit1.permute(2, 0, 1), self.t3_bool.permute(2, 0, 1))

    def test_movedim(self):
        assert_parity(self.t3_bit1.movedim(0, -1), self.t3_bool.movedim(0, -1))

    def test_swapaxes(self):
        assert_parity(self.t2_bit1.swapaxes(0, 1), self.t2_bool.swapaxes(0, 1))

    def test_swapdims(self):
        assert_parity(self.t2_bit1.swapdims(0, 1), self.t2_bool.swapdims(0, 1))

    def test_contiguous(self):
        assert_parity(self.t2_bit1.t().contiguous(), self.t2_bool.t().contiguous())

    def test_expand(self):
        a_bit1 = brute.tensor([[True], [False]], dtype=brute.bit1)
        a_bool = torch.tensor([[True], [False]])
        assert_parity(a_bit1.expand(2, 3), a_bool.expand(2, 3))

    def test_expand_as(self):
        a_bit1 = brute.tensor([[True], [False]], dtype=brute.bit1)
        a_bool = torch.tensor([[True], [False]])
        ref = torch.zeros(2, 3, dtype=torch.bool)
        ref_bit1 = brute.zeros(2, 3, dtype=brute.bit1)
        assert_parity(a_bit1.expand_as(ref_bit1), a_bool.expand_as(ref))

    def test_repeat(self):
        assert_parity(self.t1_bit1.repeat(2), self.t1_bool.repeat(2))

    def test_repeat_2d(self):
        assert_parity(self.t2_bit1.repeat(2, 3), self.t2_bool.repeat(2, 3))

    def test_tile(self):
        assert_parity(self.t1_bit1.tile(2), self.t1_bool.tile(2))

    def test_broadcast_to(self):
        a_bit1 = brute.tensor([[True], [False]], dtype=brute.bit1)
        a_bool = torch.tensor([[True], [False]])
        assert_parity(torch.broadcast_to(a_bit1, (2, 3)),
                      torch.broadcast_to(a_bool, (2, 3)))

    def test_narrow(self):
        assert_parity(self.t1_bit1.narrow(0, 1, 4), self.t1_bool.narrow(0, 1, 4))

    def test_select(self):
        assert_parity(self.t2_bit1.select(0, 1), self.t2_bool.select(0, 1))


class TestParityRearrangement:
    """roll, flip, rot90, etc. preserve bit1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t1_bit1, self.t1_bool = make_pair(DATA_1D_A)
        self.t2_bit1, self.t2_bool = make_pair(DATA_2D_A)

    def test_roll_1d(self):
        assert_parity(torch.roll(self.t1_bit1, 2), torch.roll(self.t1_bool, 2))

    def test_roll_2d(self):
        assert_parity(torch.roll(self.t2_bit1, shifts=(1, 1), dims=(0, 1)),
                      torch.roll(self.t2_bool, shifts=(1, 1), dims=(0, 1)))

    def test_flip(self):
        assert_parity(torch.flip(self.t1_bit1, [0]),
                      torch.flip(self.t1_bool, [0]))

    def test_flip_2d_both(self):
        assert_parity(torch.flip(self.t2_bit1, [0, 1]),
                      torch.flip(self.t2_bool, [0, 1]))

    def test_fliplr(self):
        assert_parity(torch.fliplr(self.t2_bit1), torch.fliplr(self.t2_bool))

    def test_flipud(self):
        assert_parity(torch.flipud(self.t2_bit1), torch.flipud(self.t2_bool))

    def test_rot90(self):
        assert_parity(torch.rot90(self.t2_bit1), torch.rot90(self.t2_bool))

    def test_rot90_k2(self):
        assert_parity(torch.rot90(self.t2_bit1, k=2),
                      torch.rot90(self.t2_bool, k=2))


class TestParityIndexing:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t1_bit1, self.t1_bool = make_pair(DATA_1D_A)
        self.t2_bit1, self.t2_bool = make_pair(DATA_2D_A)

    def test_getitem_int(self):
        # 0-dim result → plain bool brute.Tensor (not bit1)
        r_bit1 = self.t1_bit1[0]
        r_bool = self.t1_bool[0]
        assert_brute(r_bit1)
        assert r_bit1.item() == r_bool.item()

    def test_getitem_slice(self):
        assert_parity(self.t1_bit1[2:6], self.t1_bool[2:6])

    def test_getitem_neg_slice(self):
        assert_parity(self.t1_bit1[-4:], self.t1_bool[-4:])

    def test_getitem_step(self):
        assert_parity(self.t1_bit1[::2], self.t1_bool[::2])

    def test_getitem_row(self):
        assert_parity(self.t2_bit1[0], self.t2_bool[0])

    def test_getitem_col_slice(self):
        assert_parity(self.t2_bit1[:, 1:3], self.t2_bool[:, 1:3])

    def test_getitem_plain_bool_mask_no_bit1_promotion(self):
        # Plain bool mask is a non-bit1 bool input → result is plain brute bool.
        mask = torch.tensor([True, False, True, False, True, False, True, False])
        r_bit1 = self.t1_bit1[mask]
        r_bool = self.t1_bool[mask]
        assert_plain_bool(r_bit1)
        assert torch.equal(base(r_bit1), r_bool)

    def test_getitem_bit1_mask_keeps_bit1(self):
        # Mask is bit1 → all bool inputs are bit1, so promotion fires.
        mask = brute.tensor([True, False, True, False, True, False, True, False],
                            dtype=brute.bit1)
        bool_mask = torch.tensor([True, False, True, False, True, False, True, False])
        r_bit1 = self.t1_bit1[mask]
        r_bool = self.t1_bool[bool_mask]
        assert_parity(r_bit1, r_bool)

    def test_getitem_advanced(self):
        assert_parity(self.t2_bit1[[0, 1], [2, 0]],
                      self.t2_bool[[0, 1], [2, 0]])

    def test_index_select(self):
        idx = torch.tensor([0, 2, 4, 6])
        assert_parity(torch.index_select(self.t1_bit1, 0, idx),
                      torch.index_select(self.t1_bool, 0, idx))

    def test_gather(self):
        idx = torch.tensor([0, 2, 4, 6])
        assert_parity(torch.gather(self.t1_bit1, 0, idx),
                      torch.gather(self.t1_bool, 0, idx))

    def test_gather_2d(self):
        idx = torch.tensor([[0, 1, 0], [1, 0, 1]])
        assert_parity(torch.gather(self.t2_bit1, 1, idx),
                      torch.gather(self.t2_bool, 1, idx))

    def test_masked_select(self):
        mask_bit1 = brute.tensor([True, False] * 4, dtype=brute.bit1)
        mask_bool = torch.tensor([True, False] * 4)
        # masked_select needs a bool mask; bit1 mask interacts via __torch_function__
        assert_parity(torch.masked_select(self.t1_bit1, mask_bit1),
                      torch.masked_select(self.t1_bool, mask_bool))

    def test_take(self):
        idx = torch.tensor([0, 3, 5])
        assert_parity(torch.take(self.t1_bit1, idx), torch.take(self.t1_bool, idx))


class TestParityCombining:

    def test_cat_dim0(self):
        a_bit1, a_bool = make_pair([True, False])
        b_bit1, b_bool = make_pair([False, True])
        assert_parity(torch.cat([a_bit1, b_bit1]), torch.cat([a_bool, b_bool]))

    def test_cat_dim1(self):
        a_bit1 = brute.zeros(3, 2, dtype=brute.bit1)
        b_bit1 = brute.ones(3, 2, dtype=brute.bit1)
        a_bool = torch.zeros(3, 2, dtype=torch.bool)
        b_bool = torch.ones(3, 2, dtype=torch.bool)
        assert_parity(torch.cat([a_bit1, b_bit1], dim=1),
                      torch.cat([a_bool, b_bool], dim=1))

    def test_cat_three_tensors(self):
        x_bit1 = brute.zeros(2, dtype=brute.bit1)
        y_bit1 = brute.ones(3, dtype=brute.bit1)
        z_bit1 = brute.zeros(1, dtype=brute.bit1)
        x_bool = torch.zeros(2, dtype=torch.bool)
        y_bool = torch.ones(3, dtype=torch.bool)
        z_bool = torch.zeros(1, dtype=torch.bool)
        assert_parity(torch.cat([x_bit1, y_bit1, z_bit1]),
                      torch.cat([x_bool, y_bool, z_bool]))

    def test_stack(self):
        a_bit1, a_bool = make_pair([True, False])
        b_bit1, b_bool = make_pair([False, True])
        assert_parity(torch.stack([a_bit1, b_bit1]), torch.stack([a_bool, b_bool]))

    def test_stack_dim1(self):
        a_bit1, a_bool = make_pair([True, False, True])
        b_bit1, b_bool = make_pair([False, True, False])
        assert_parity(torch.stack([a_bit1, b_bit1], dim=1),
                      torch.stack([a_bool, b_bool], dim=1))


class TestParitySplitting:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t_bit1, self.t_bool = make_pair(DATA_1D_A)

    def test_split_chunks_of_2(self):
        out_bit1 = self.t_bit1.split(2)
        out_bool = self.t_bool.split(2)
        assert len(out_bit1) == len(out_bool)
        for b1, b in zip(out_bit1, out_bool):
            assert_parity(b1, b)

    def test_split_with_sizes(self):
        out_bit1 = self.t_bit1.split([3, 5])
        out_bool = self.t_bool.split([3, 5])
        for b1, b in zip(out_bit1, out_bool):
            assert_parity(b1, b)

    def test_chunk(self):
        out_bit1 = self.t_bit1.chunk(4)
        out_bool = self.t_bool.chunk(4)
        assert len(out_bit1) == len(out_bool) == 4
        for b1, b in zip(out_bit1, out_bool):
            assert_parity(b1, b)

    def test_unbind(self):
        t2_bit1, t2_bool = make_pair(DATA_2D_A)
        out_bit1 = t2_bit1.unbind(0)
        out_bool = t2_bool.unbind(0)
        for b1, b in zip(out_bit1, out_bool):
            assert_parity(b1, b)

    def test_tensor_split(self):
        out_bit1 = torch.tensor_split(self.t_bit1, 4)
        out_bool = torch.tensor_split(self.t_bool, 4)
        for b1, b in zip(out_bit1, out_bool):
            assert_parity(b1, b)

    def test_hsplit_2d(self):
        t2_bit1 = brute.tensor([[True, False, True, False],
                                [False, True, False, True]], dtype=brute.bit1)
        t2_bool = torch.tensor([[True, False, True, False],
                                [False, True, False, True]])
        out_bit1 = torch.hsplit(t2_bit1, 2)
        out_bool = torch.hsplit(t2_bool, 2)
        for b1, b in zip(out_bit1, out_bool):
            assert_parity(b1, b)


class TestParitySorting:

    def test_sort_1d_returns_brute(self):
        t_bit1, t_bool = make_pair(DATA_1D_A)
        v_bit1, i_bit1 = torch.sort(t_bit1)
        v_bool, i_bool = torch.sort(t_bool)
        assert_parity(v_bit1, v_bool)
        assert_brute(i_bit1)
        assert torch.equal(base(i_bit1), i_bool)

    def test_argsort(self):
        t_bit1, t_bool = make_pair(DATA_1D_A)
        r_bit1 = torch.argsort(t_bit1)
        r_bool = torch.argsort(t_bool)
        assert_brute(r_bit1)
        assert torch.equal(base(r_bit1), r_bool)

    def test_topk_raises_on_bool(self):
        # PyTorch CPU doesn't support topk on bool — bit1 follows.
        t_bit1, _ = make_pair(DATA_1D_A)
        with pytest.raises(RuntimeError):
            torch.topk(t_bit1, 3)

    def test_topk_after_cast(self):
        # Cast to int → topk works and stays brute.
        t_bit1, t_bool = make_pair(DATA_1D_A)
        v_bit1, i_bit1 = torch.topk(t_bit1.long(), 3)
        v_bool, i_bool = torch.topk(t_bool.long(), 3)
        assert_brute(v_bit1)
        assert_brute(i_bit1)
        assert torch.equal(base(v_bit1), v_bool)
        assert torch.equal(base(i_bit1), i_bool)

    def test_unique(self):
        t_bit1, t_bool = make_pair(DATA_1D_A)
        u_bit1 = torch.unique(t_bit1)
        u_bool = torch.unique(t_bool)
        assert_parity(u_bit1, u_bool)

    def test_unique_consecutive(self):
        t_bit1, t_bool = make_pair([True, True, False, False, True, False])
        u_bit1 = torch.unique_consecutive(t_bit1)
        u_bool = torch.unique_consecutive(t_bool)
        assert_parity(u_bit1, u_bool)

    def test_msort(self):
        t2_bit1, t2_bool = make_pair(DATA_2D_A)
        assert_parity(torch.msort(t2_bit1), torch.msort(t2_bool))


# ════════════════════════════════════════════════════════════════════════════════
#   Section 10: Return-type consistency — every op returns brute.Tensor
# ════════════════════════════════════════════════════════════════════════════════

class TestReturnTypeConsistency:
    """Every operation on a bit1 tensor returns a brute.Tensor, never a bare torch.Tensor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t = brute.tensor(DATA_1D_A, dtype=brute.bit1)
        self.t2 = brute.tensor(DATA_2D_A, dtype=brute.bit1)

    # Shape ops
    def test_view(self):       assert_bit1(self.t.view(2, 4))
    def test_reshape(self):    assert_bit1(self.t.reshape(2, 4))
    def test_flatten(self):    assert_bit1(self.t2.flatten())
    def test_unsqueeze(self):  assert_bit1(self.t.unsqueeze(0))
    def test_squeeze(self):    assert_bit1(self.t.unsqueeze(0).squeeze())
    def test_t(self):          assert_bit1(self.t2.t())
    def test_transpose(self):  assert_bit1(self.t2.transpose(0, 1))
    def test_permute(self):    assert_bit1(self.t2.permute(1, 0))
    def test_contiguous(self): assert_bit1(self.t2.t().contiguous())
    def test_movedim(self):    assert_bit1(self.t2.movedim(0, 1))
    def test_swapaxes(self):   assert_bit1(self.t2.swapaxes(0, 1))
    def test_swapdims(self):   assert_bit1(self.t2.swapdims(0, 1))

    # Copy/clone
    def test_clone(self):      assert_bit1(self.t.clone())
    def test_detach(self):     assert_bit1(self.t.detach())

    # Expand/repeat
    def test_expand(self):     assert_bit1(self.t.unsqueeze(0).expand(3, -1))
    def test_repeat(self):     assert_bit1(self.t.repeat(2))
    def test_tile(self):       assert_bit1(self.t.tile(2))

    # Rearrangement
    def test_roll(self):       assert_bit1(self.t.roll(2))
    def test_flip(self):       assert_bit1(self.t.flip(0))
    def test_fliplr(self):     assert_bit1(self.t2.fliplr())
    def test_flipud(self):     assert_bit1(self.t2.flipud())
    def test_rot90(self):      assert_bit1(torch.rot90(self.t2))

    # Logical
    def test_logical_not(self):  assert_bit1(~self.t)
    def test_logical_and(self):  assert_bit1(self.t & self.t)
    def test_logical_or(self):   assert_bit1(self.t | self.t)
    def test_logical_xor(self):  assert_bit1(self.t ^ self.t)

    # Comparison (bit1 vs bit1 stays bit1)
    def test_eq(self):  assert_bit1(self.t == self.t)
    def test_ne(self):  assert_bit1(self.t != self.t)
    def test_lt(self):  assert_bit1(self.t < self.t)
    def test_le(self):  assert_bit1(self.t <= self.t)
    def test_gt(self):  assert_bit1(self.t > self.t)
    def test_ge(self):  assert_bit1(self.t >= self.t)

    # Reductions (non-bool results — still brute.Tensor)
    def test_sum_brute(self):           assert_brute(self.t.sum())
    def test_count_nonzero_brute(self): assert_brute(torch.count_nonzero(self.t))
    def test_nonzero_brute(self):       assert_brute(torch.nonzero(self.t))
    def test_argmax_after_cast_brute(self): assert_brute(torch.argmax(self.t.long()))
    def test_argmin_after_cast_brute(self): assert_brute(torch.argmin(self.t.long()))

    # Reductions on bit1 → 0-dim bool stays plain brute (can't pack 0-dim)
    def test_all_brute(self):           assert_brute(self.t.all())
    def test_any_brute(self):           assert_brute(self.t.any())

    # Combining
    def test_cat(self):   assert_bit1(torch.cat([self.t, self.t]))
    def test_stack(self): assert_bit1(torch.stack([self.t, self.t]))

    # Splitting (each chunk is bit1)
    def test_split_chunks(self):
        for c in self.t.split(2):
            assert_bit1(c)
    def test_chunk_chunks(self):
        for c in self.t.chunk(2):
            assert_bit1(c)
    def test_unbind_chunks(self):
        for c in self.t2.unbind(0):
            assert_bit1(c)

    # Indexing
    def test_slice(self):        assert_bit1(self.t[1:5])
    def test_index_select(self): assert_bit1(torch.index_select(self.t, 0, torch.tensor([0, 2, 4])))
    def test_gather(self):       assert_bit1(torch.gather(self.t, 0, torch.tensor([0, 2, 4])))
    def test_masked_select(self):
        m = brute.tensor([True, False] * 4, dtype=brute.bit1)
        assert_bit1(torch.masked_select(self.t, m))
    def test_narrow(self):       assert_bit1(self.t.narrow(0, 1, 4))
    def test_select_returns_brute(self):
        # 1-dim → 0-dim slice; can't pack 0-dim, stays plain brute
        assert_brute(self.t2.select(0, 0))

    # Factory
    def test_new_zeros(self): assert_bit1(self.t.new_zeros([4]))
    def test_new_ones(self):  assert_bit1(self.t.new_ones([4]))
    def test_new_full(self):  assert_bit1(self.t.new_full([4], True))
    def test_new_empty(self): assert_bit1(self.t.new_empty([4]))
    def test_new_tensor(self): assert_bit1(self.t.new_tensor([True, False]))

    # brute.* namespace
    def test_brute_clone(self):       assert_bit1(brute.clone(self.t))
    def test_brute_detach(self):      assert_bit1(brute.detach(self.t))
    def test_brute_logical_not(self): assert_bit1(brute.logical_not(self.t))
    def test_brute_logical_and(self): assert_bit1(brute.logical_and(self.t, self.t))
    def test_brute_roll(self):        assert_bit1(brute.roll(self.t, 1))
    def test_brute_flip(self):        assert_bit1(brute.flip(self.t, [0]))
    def test_brute_reshape(self):     assert_bit1(brute.reshape(self.t, (2, 4)))
    def test_brute_flatten(self):     assert_bit1(brute.flatten(self.t))
    def test_brute_squeeze(self):     assert_bit1(brute.squeeze(self.t.unsqueeze(0)))
    def test_brute_unsqueeze(self):   assert_bit1(brute.unsqueeze(self.t, 0))
    def test_brute_broadcast_to(self):
        assert_bit1(brute.broadcast_to(self.t.unsqueeze(0), (3, 8)))
    def test_brute_eq(self):  assert_bit1(brute.eq(self.t, self.t))
    def test_brute_ne(self):  assert_bit1(brute.ne(self.t, self.t))
    def test_brute_lt(self):  assert_bit1(brute.lt(self.t, self.t))
    def test_brute_transpose(self): assert_bit1(brute.transpose(self.t2, 0, 1))
    def test_brute_t(self):         assert_bit1(brute.t(self.t2))
    def test_brute_permute(self):   assert_bit1(brute.permute(self.t2, (1, 0)))
    def test_brute_nonzero(self):   assert_brute(brute.nonzero(self.t))
    def test_brute_unique(self):    assert_brute(brute.unique(self.t))
    def test_brute_argmax_after_cast(self): assert_brute(brute.argmax(self.t.long()))


# ════════════════════════════════════════════════════════════════════════════════
#   Section 11: Cross-dtype interactions
# ════════════════════════════════════════════════════════════════════════════════

class TestCrossDtypeBoolPair:
    """bit1 paired with plain torch.bool: must NOT promote to bit1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a_bit1 = brute.tensor([True, False, True, False], dtype=brute.bit1)
        self.b_bool = torch.tensor([True, True, False, False])

    def test_logical_and(self):
        r = torch.logical_and(self.a_bit1, self.b_bool)
        assert_plain_bool(r)
        assert torch.equal(base(r), torch.tensor([True, False, False, False]))

    def test_logical_or(self):
        r = torch.logical_or(self.a_bit1, self.b_bool)
        assert_plain_bool(r)

    def test_logical_xor(self):
        r = torch.logical_xor(self.a_bit1, self.b_bool)
        assert_plain_bool(r)

    def test_bitwise_and(self):
        r = self.a_bit1 & self.b_bool
        assert_plain_bool(r)

    def test_eq(self):
        r = self.a_bit1 == self.b_bool
        assert_plain_bool(r)

    def test_cat(self):
        r = torch.cat([self.a_bit1, brute.tensor([True, False], dtype=torch.bool)])
        assert_plain_bool(r)

    def test_stack(self):
        c_bool = brute.tensor([True, False, True, False], dtype=torch.bool)
        r = torch.stack([self.a_bit1, c_bool])
        assert_plain_bool(r)

    def test_brute_bool_does_not_promote(self):
        b = brute.tensor([True, True, False, False], dtype=torch.bool)
        r = self.a_bit1 & b
        assert_plain_bool(r)


class TestCrossDtypeFloat32Pair:
    """bit1 paired with float32: float-promoting ops yield float results; bool-returning ops yield bit1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a_bit1 = brute.tensor([True, False, True, False], dtype=brute.bit1)
        self.f = torch.tensor([1.0, 0.0, 0.5, 1.5])

    def test_add_to_float(self):
        # True + 1.0 = 2.0, False + 0.0 = 0.0, etc.
        r = self.a_bit1 + self.f
        assert_brute(r, dtype=torch.float32)
        assert torch.equal(base(r), torch.tensor([2.0, 0.0, 1.5, 1.5]))

    def test_mul_to_float(self):
        r = self.a_bit1 * self.f
        assert_brute(r, dtype=torch.float32)

    def test_eq_with_float(self):
        # bit1 ∈ {True,False} ≅ {1,0}; eq promotes bool→float internally
        r = self.a_bit1 == self.f
        # promote_to_bit1 = True (no non-bit1 bool inputs), result.dtype=bool → bit1
        assert_bit1(r)

    def test_lt_with_float(self):
        r = self.a_bit1 < self.f
        assert_bit1(r)

    def test_where_with_float(self):
        out = torch.where(self.a_bit1.bool(), self.f, torch.zeros_like(self.f))
        # torch.where with bool cond: returns float; here it's plain torch
        assert torch.equal(out, torch.tensor([1.0, 0.0, 0.5, 0.0]))


class TestCrossDtypeInt64Pair:
    """bit1 paired with int64: arithmetic returns int64 brute.Tensor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a_bit1 = brute.tensor([True, False, True, False], dtype=brute.bit1)
        self.i = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    def test_add_to_int(self):
        r = self.a_bit1 + self.i
        assert_brute(r, dtype=torch.int64)
        assert torch.equal(base(r), torch.tensor([2, 2, 4, 4], dtype=torch.int64))

    def test_mul_to_int(self):
        r = self.a_bit1 * self.i
        assert_brute(r, dtype=torch.int64)
        assert torch.equal(base(r), torch.tensor([1, 0, 3, 0], dtype=torch.int64))

    def test_eq_with_int(self):
        # bool == int: bool result; promote_to_bit1=True → bit1
        r = self.a_bit1 == self.i
        assert_bit1(r)

    def test_lt_with_int(self):
        r = self.a_bit1 < self.i
        assert_bit1(r)


class TestCrossDtypeBit1Pair:
    """bit1 with bit1 of different pack_dtypes: result is bit1; pack_dtype comes from first operand."""

    def test_uint8_with_uint32_and(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1, pack_dtype=brute.uint8)
        b = brute.tensor([True, True, False], dtype=brute.bit1, pack_dtype=brute.uint32)
        r = a & b
        assert_bit1(r)
        assert torch.equal(base(r), torch.tensor([True, False, False]))

    def test_uint8_with_uint64_or(self):
        a = brute.tensor([True, False], dtype=brute.bit1, pack_dtype=brute.uint8)
        b = brute.tensor([False, True], dtype=brute.bit1, pack_dtype=brute.uint64)
        assert_bit1(a | b)

    def test_uint32_with_uint64_xor(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1, pack_dtype=brute.uint32)
        b = brute.tensor([True, True, False], dtype=brute.bit1, pack_dtype=brute.uint64)
        assert_bit1(a ^ b)


class TestCrossDtypeMatrixMixed:
    """Mixed-dtype matmul: bit1 matmul with float requires an explicit float cast."""

    def test_bit1_to_float_then_matmul(self):
        a = brute.ones(2, 4, dtype=brute.bit1).float()
        b = torch.ones(4, 3)
        r = a @ b
        assert torch.equal(base(r), 4 * torch.ones(2, 3))

    def test_unpack_pm1_matmul_float_reference(self):
        # The bit1 XNOR matmul should equal a {-1,+1} float matmul.
        torch.manual_seed(123)
        K = 16
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1)
        b = brute.tensor(b_bool, dtype=brute.bit1)
        r_bit1 = (a @ b).float()
        r_float = (a.unpack_pm1()) @ (b.unpack_pm1().t())
        assert torch.equal(r_bit1, r_float)


# ════════════════════════════════════════════════════════════════════════════════
#   Section 12: Indexing tests
# ════════════════════════════════════════════════════════════════════════════════

class TestIndexing:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t = brute.tensor(DATA_2D_A, dtype=brute.bit1)
        self.ref = torch.tensor(DATA_2D_A)

    def test_row_index_is_bit1(self):
        assert_bit1(self.t[0])

    def test_row_values(self):
        assert torch.equal(base(self.t[0]), self.ref[0])

    def test_col_slice_is_bit1(self):
        assert_bit1(self.t[:, 1:])

    def test_col_slice_values(self):
        assert torch.equal(base(self.t[:, 1:]), self.ref[:, 1:])

    def test_scalar_index_returns_brute(self):
        assert_brute(self.t[0, 0])

    def test_scalar_index_values(self):
        assert self.t[0, 0].item() == self.ref[0, 0].item()
        assert self.t[0, 1].item() == self.ref[0, 1].item()

    def test_bool_mask(self):
        mask = torch.tensor([True, False])
        assert torch.equal(base(self.t[mask]).bool(), self.ref[mask])

    def test_advanced_index(self):
        assert torch.equal(base(self.t[[0, 1], [2, 0]]), self.ref[[0, 1], [2, 0]])

    def test_setitem_scalar(self):
        t = brute.zeros(4, dtype=brute.bit1)
        t[0] = True
        assert base(t)[0].item() is True

    def test_setitem_slice(self):
        t = brute.zeros(4, dtype=brute.bit1)
        t[1:3] = True
        assert torch.equal(base(t), torch.tensor([False, True, True, False]))

    def test_setitem_mask(self):
        t = brute.zeros(4, dtype=brute.bit1)
        t[torch.tensor([True, False, True, False])] = True
        assert torch.equal(base(t), torch.tensor([True, False, True, False]))


# ════════════════════════════════════════════════════════════════════════════════
#   Section 13: Repr, len, iter, pickle
# ════════════════════════════════════════════════════════════════════════════════

class TestRepr:

    def test_small_bit1_has_dtype(self):
        assert 'bit1' in repr(brute.tensor([True, False], dtype=brute.bit1))

    def test_large_bit1_has_dtype(self):
        assert 'bit1' in repr(brute.ones(200, dtype=brute.bit1))

    def test_repr_has_brute_tensor(self):
        t = brute.zeros(4, dtype=brute.bit1)
        r = repr(t)
        assert 'brute.Tensor' in r and 'tensor(' not in r

    def test_no_torch_bool_in_bit1_repr(self):
        assert 'torch.bool' not in repr(brute.tensor([True, False, True], dtype=brute.bit1))

    def test_float_repr_no_bit1(self):
        r = repr(brute.randn(4))
        assert 'brute.Tensor' in r and 'bit1' not in r

    def test_bool_tensor_repr_no_bit1(self):
        assert 'bit1' not in repr(brute.tensor([True, False], dtype=torch.bool))

    def test_str_falls_back(self):
        # str(t) should also work
        s = str(brute.tensor([True, False], dtype=brute.bit1))
        assert 'bit1' in s


class TestSpecialMethods:

    def test_len(self):
        assert len(brute.zeros(5, dtype=brute.bit1)) == 5

    def test_iter(self):
        data = [True, False, True]
        assert [e.item() for e in brute.tensor(data, dtype=brute.bit1)] == data

    def test_bool_scalar_true(self):
        t = brute.tensor([True], dtype=brute.bit1)
        assert bool(t[0]) is True

    def test_bool_scalar_false(self):
        t = brute.tensor([False], dtype=brute.bit1)
        assert bool(t[0]) is False

    def test_tolist_1d(self):
        data = [True, False, True]
        assert brute.tensor(data, dtype=brute.bit1).bool().tolist() == data

    def test_tolist_2d(self):
        data = [[True, False], [False, True]]
        assert brute.tensor(data, dtype=brute.bit1).bool().tolist() == data


# ════════════════════════════════════════════════════════════════════════════════
#   Section 14: Deepcopy, clone, pickling
# ════════════════════════════════════════════════════════════════════════════════

class TestDeepCopy:

    def test_deepcopy_bit1_values(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        t2 = copy.deepcopy(t)
        assert_bit1(t2)
        assert torch.equal(base(t2), base(t))

    def test_deepcopy_independent_storage(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        t2 = copy.deepcopy(t)
        assert t2.data_ptr() != t.data_ptr()

    def test_deepcopy_preserves_pack_dtype(self):
        t = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint64)
        t2 = copy.deepcopy(t)
        assert t2.pack_dtype is torch.uint64

    def test_deepcopy_packed_buf_correct(self):
        t = brute.tensor([True, False, True, False, True, False, True, False],
                         dtype=brute.bit1)
        t2 = copy.deepcopy(t)
        # Packed bufs should be value-equal even though storage is independent
        assert torch.equal(t._packed_buf, t2._packed_buf)

    def test_deepcopy_float(self):
        t = brute.randn(4)
        t2 = copy.deepcopy(t)
        assert torch.equal(base(t2), base(t))

    def test_deepcopy_then_modify_independent(self):
        t = brute.zeros(4, dtype=brute.bit1)
        t2 = copy.deepcopy(t)
        t2.fill_(True)
        assert not base(t).any()  # original untouched
        assert base(t2).all()

    def test_clone_values(self):
        t = brute.tensor([True, False], dtype=brute.bit1)
        t2 = t.clone()
        assert_bit1(t2)
        assert torch.equal(base(t2), base(t))

    def test_clone_independent_storage(self):
        t = brute.tensor([True, False], dtype=brute.bit1)
        t2 = t.clone()
        assert t2.data_ptr() != t.data_ptr()

    def test_detach_returns_bit1(self):
        t = brute.tensor([True, False], dtype=brute.bit1)
        assert_bit1(t.detach())


# ════════════════════════════════════════════════════════════════════════════════
#   Section 15: Matmul
# ════════════════════════════════════════════════════════════════════════════════

class TestMatmul:
    def test_bit1_x_bit1_manual(self):
        a = brute.tensor([[True, True, False, True]], dtype=brute.bit1)
        b = brute.tensor([[True, False, False, True],
                          [False, True, True, False]], dtype=brute.bit1)
        r = a @ b
        assert r.shape == torch.Size([1, 2])
        assert r[0, 0].item() == 2 and r[0, 1].item() == -2

    def test_bit1_x_bit1_vs_float_reference(self):
        torch.manual_seed(0)
        a_bool = torch.randint(0, 2, (8, 16)).bool()
        b_bool = torch.randint(0, 2, (12, 16)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1)
        b = brute.tensor(b_bool, dtype=brute.bit1)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

    def test_torch_matmul_dispatches(self):
        a = brute.ones(4, 8, dtype=brute.bit1)
        b = brute.ones(6, 8, dtype=brute.bit1)
        assert torch.matmul(a, b).shape == torch.Size([4, 6])

    def test_brute_matmul(self):
        a = brute.ones(4, 8, dtype=brute.bit1)
        b = brute.ones(6, 8, dtype=brute.bit1)
        assert brute.matmul(a, b).shape == torch.Size([4, 6])

    def test_brute_mm(self):
        a = brute.ones(4, 8, dtype=brute.bit1)
        b = brute.ones(6, 8, dtype=brute.bit1)
        assert brute.mm(a, b).shape == torch.Size([4, 6])

    def test_float_x_float(self):
        assert (brute.randn(3, 4) @ brute.randn(4, 5)).shape == torch.Size([3, 5])

    def test_all_true_x_all_true(self):
        K = 8
        assert (brute.ones(2, K, dtype=brute.bit1) @ brute.ones(3, K, dtype=brute.bit1) == K).all()

    def test_all_false_x_all_false(self):
        K = 8
        assert (brute.zeros(2, K, dtype=brute.bit1) @ brute.zeros(2, K, dtype=brute.bit1) == K).all()

    def test_all_true_x_all_false(self):
        K = 8
        assert (brute.ones(2, K, dtype=brute.bit1) @ brute.zeros(3, K, dtype=brute.bit1) == -K).all()

    def test_matmul_result_is_not_bit1(self):
        a = brute.ones(2, 4, dtype=brute.bit1)
        b = brute.ones(3, 4, dtype=brute.bit1)
        assert not getattr(a @ b, '_is_bit1', False)


class TestUnpackPm1:

    def test_all_true(self):
        assert torch.equal(brute.ones(4, dtype=brute.bit1).unpack_pm1(), torch.ones(4))

    def test_all_false(self):
        assert torch.equal(brute.zeros(4, dtype=brute.bit1).unpack_pm1(), -torch.ones(4))

    def test_alternating(self):
        t = brute.tensor([True, False, True, False], dtype=brute.bit1)
        assert torch.equal(t.unpack_pm1(), torch.tensor([1.0, -1.0, 1.0, -1.0]))

    def test_shape_preserved(self):
        t = brute.rand(3, 4, dtype=brute.bit1)
        r = t.unpack_pm1()
        assert r.shape == torch.Size([3, 4]) and r.dtype == torch.float32

    def test_fails_on_non_bit1(self):
        with pytest.raises(TypeError):
            brute.zeros(4).unpack_pm1()

    def test_unpack_2d(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert torch.equal(t.unpack_pm1(), torch.tensor([[1., -1.], [-1., 1.]]))

    def test_unpack_3d(self):
        t = brute.ones(2, 3, 4, dtype=brute.bit1)
        r = t.unpack_pm1()
        assert r.shape == torch.Size([2, 3, 4]) and (r == 1.0).all()


class TestPackDtypes:

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_pack_dtype_preserved(self, pw):
        assert brute.zeros(8, dtype=brute.bit1, pack_dtype=pw).pack_dtype is pw

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_unpack_pm1_all_true(self, pw):
        assert torch.equal(brute.ones(8, dtype=brute.bit1, pack_dtype=pw).unpack_pm1(),
                           torch.ones(8))

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_unpack_pm1_all_false(self, pw):
        assert torch.equal(brute.zeros(8, dtype=brute.bit1, pack_dtype=pw).unpack_pm1(),
                           -torch.ones(8))

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_matmul_all_ones(self, pw):
        K = 16
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=pw)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=pw)
        assert (a @ b == K).all()

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_matmul_vs_float_reference(self, pw):
        torch.manual_seed(42)
        K = 16
        a_bool = torch.randint(0, 2, (4, K)).bool()
        b_bool = torch.randint(0, 2, (5, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=pw)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=pw)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

    @pytest.mark.parametrize("pw", PACK_DTYPES)
    def test_bool_ops_preserve_pack_dtype(self, pw):
        a = brute.ones(8, dtype=brute.bit1, pack_dtype=pw)
        b = brute.zeros(8, dtype=brute.bit1, pack_dtype=pw)
        r = a & b
        assert_bit1(r)


class TestMatmulKBoundary:

    @pytest.mark.parametrize("K", [7, 8, 9])
    def test_uint8_all_ones(self, K):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=brute.uint8)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert (a @ b == K).all()

    @pytest.mark.parametrize("K", [7, 8, 9])
    def test_uint8_vs_float(self, K):
        torch.manual_seed(K)
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=brute.uint8)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

    @pytest.mark.parametrize("K", [31, 32, 33])
    def test_uint32_all_ones(self, K):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=brute.uint32)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=brute.uint32)
        assert (a @ b == K).all()

    @pytest.mark.parametrize("K", [31, 32, 33])
    def test_uint32_vs_float(self, K):
        torch.manual_seed(K)
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=brute.uint32)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=brute.uint32)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

    @pytest.mark.parametrize("K", [63, 64, 65])
    def test_uint64_all_ones(self, K):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=brute.uint64)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=brute.uint64)
        assert (a @ b == K).all()

    @pytest.mark.parametrize("K", [63, 64, 65])
    def test_uint64_vs_float(self, K):
        torch.manual_seed(K)
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=brute.uint64)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=brute.uint64)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

    def test_large_k_uint8(self):
        K = 256
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=brute.uint8)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert (a @ b == K).all()

    def test_large_k_vs_float(self):
        torch.manual_seed(99)
        K = 128
        a_bool = torch.randint(0, 2, (4, K)).bool()
        b_bool = torch.randint(0, 2, (5, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=brute.uint32)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=brute.uint32)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))


# ════════════════════════════════════════════════════════════════════════════════
#   Section 16: Edge cases — empty tensors, 0-dim scalars
# ════════════════════════════════════════════════════════════════════════════════

class TestEmptyAndSingleton:

    def test_empty_1d_shape(self):
        t = brute.zeros(0, dtype=brute.bit1)
        assert t.shape == torch.Size([0])
        assert t.dtype == brute.bit1
        assert t.numel() == 0

    def test_empty_2d_shape(self):
        assert brute.zeros(0, 4, dtype=brute.bit1).shape == torch.Size([0, 4])

    def test_empty_sum(self):
        assert brute.sum(brute.zeros(0, dtype=brute.bit1)).item() == 0

    def test_empty_all_vacuous(self):
        assert brute.all(brute.zeros(0, dtype=brute.bit1)).item() is True

    def test_empty_any_false(self):
        assert brute.any(brute.zeros(0, dtype=brute.bit1)).item() is False

    def test_singleton_1d(self):
        t = brute.tensor([True], dtype=brute.bit1)
        assert t.shape == torch.Size([1])
        assert t[0].item() is True

    def test_singleton_matmul(self):
        a = brute.tensor([[True]], dtype=brute.bit1)
        b = brute.tensor([[True]], dtype=brute.bit1)
        assert (a @ b).item() == 1

    def test_singleton_matmul_false(self):
        a = brute.zeros(1, 1, dtype=brute.bit1)
        b = brute.zeros(1, 1, dtype=brute.bit1)
        assert (a @ b).item() == 1

    def test_singleton_matmul_mixed(self):
        a = brute.ones(1, 1, dtype=brute.bit1)
        b = brute.zeros(1, 1, dtype=brute.bit1)
        assert (a @ b).item() == -1

    def test_cat_empty_and_nonempty(self):
        r = torch.cat([brute.zeros(0, dtype=brute.bit1), brute.ones(3, dtype=brute.bit1)])
        assert r.shape == torch.Size([3])
        assert r.dtype == brute.bit1


class TestZeroDimScalars:

    def test_scalar_index_true(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert t[0, 0].item() is True

    def test_scalar_index_false(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert t[0, 1].item() is False

    def test_scalar_0dim_is_not_bit1(self):
        # 0-dim bool from bit1 indexing → plain brute (can't pack 0-dim)
        t = brute.tensor([True, False], dtype=brute.bit1)
        s = t[0]
        assert s.dim() == 0
        assert s.item() is True
        assert not getattr(s, '_is_bit1', False)

    def test_global_all_all_true(self):
        assert brute.all(brute.ones(6, dtype=brute.bit1)).item() is True

    def test_global_all_not_all_true(self):
        assert brute.all(brute.tensor([True, False, True], dtype=brute.bit1)).item() is False

    def test_tolist(self):
        data = [True, False, True]
        assert brute.tensor(data, dtype=brute.bit1).bool().tolist() == data

    def test_min_global(self):
        assert brute.min(brute.tensor([True, False, True], dtype=brute.bit1)).item() is False

    def test_max_global(self):
        assert brute.max(brute.tensor([True, False, True], dtype=brute.bit1)).item() is True


# ════════════════════════════════════════════════════════════════════════════════
#   Section 17: Multi-device (MPS)
# ════════════════════════════════════════════════════════════════════════════════

_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


class TestMultiDevice:
    @_mps
    def test_mps_zeros_bit1_shape(self):
        t = brute.zeros(3, 4, dtype=brute.bit1, device='mps')
        assert t.device.type == 'mps'
        assert t.shape == torch.Size([3, 4])
        assert t.dtype == brute.bit1

    @_mps
    def test_mps_ones_bit1_values(self):
        t = brute.ones(4, dtype=brute.bit1, device='mps')
        assert t.device.type == 'mps'
        assert base(t).all()

    @_mps
    def test_cpu_to_mps_preserves_bit1(self):
        t_cpu = brute.tensor([True, False, True], dtype=brute.bit1)
        t_mps = t_cpu.to('mps')
        assert_bit1(t_mps)
        assert t_mps.device.type == 'mps'
        assert torch.equal(base(t_mps).cpu(), base(t_cpu))

    @_mps
    def test_mps_to_cpu_round_trip(self):
        t = brute.tensor([True, False, True, False], dtype=brute.bit1)
        t2 = t.to('mps').to('cpu')
        assert_bit1(t2)
        assert torch.equal(base(t2), base(t))

    @_mps
    def test_mps_bool_and(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1, device='mps')
        b = brute.tensor([True, True, False], dtype=brute.bit1, device='mps')
        r = a & b
        assert_bit1(r)
        assert torch.equal(base(r).cpu(), torch.tensor([True, False, False]))

    @_mps
    def test_mps_matmul_all_ones(self):
        K = 16
        a = brute.ones(3, K, dtype=brute.bit1, device='mps')
        b = brute.ones(4, K, dtype=brute.bit1, device='mps')
        r = a @ b
        assert r.shape == torch.Size([3, 4]) and (r == K).all()

    @_mps
    def test_mps_packed_buf_on_mps(self):
        assert brute.ones(4, 8, dtype=brute.bit1, device='mps')._packed_buf.device.type == 'mps'

    @_mps
    def test_mps_popcount(self):
        assert brute.ones(8, dtype=brute.bit1, device='mps').popcount().item() == 8


# ════════════════════════════════════════════════════════════════════════════════
#   Section 18: brute.* namespace surface — verify every re-exported function exists
# ════════════════════════════════════════════════════════════════════════════════

class TestNamespaceCompleteness:
    """Spot-check that brute exports the surface we expect."""

    @pytest.mark.parametrize("name", [
        # Factories
        'zeros', 'ones', 'empty', 'full', 'tensor', 'as_tensor', 'from_numpy',
        'rand', 'randn', 'randint',
        'rand_like', 'randn_like', 'zeros_like', 'ones_like', 'full_like', 'empty_like',
        'arange', 'linspace', 'eye',
        # Reductions
        'all', 'any', 'sum', 'max', 'min', 'mean', 'prod',
        'amax', 'amin', 'aminmax', 'argmax', 'argmin',
        'count_nonzero', 'nonzero', 'argwhere',
        # Shape
        'reshape', 'flatten', 'squeeze', 'unsqueeze', 'permute', 'transpose', 't',
        'movedim', 'moveaxis', 'swapaxes', 'swapdims',
        'broadcast_to', 'broadcast_tensors', 'narrow', 'select',
        'atleast_1d', 'atleast_2d', 'atleast_3d',
        # Splitting
        'split', 'chunk', 'unbind', 'tensor_split', 'hsplit', 'vsplit', 'dsplit',
        # Cloning
        'clone', 'detach',
        # Logical
        'where', 'logical_and', 'logical_or', 'logical_xor', 'logical_not',
        # Bitwise
        'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
        'bitwise_left_shift', 'bitwise_right_shift',
        # Comparison
        'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'equal', 'allclose', 'isclose',
        # Arithmetic
        'add', 'sub', 'mul', 'div', 'neg', 'abs', 'sign',
        # Combining
        'cat', 'stack',
        # Sorting
        'sort', 'argsort', 'topk', 'unique', 'unique_consecutive',
        # Indexing
        'gather', 'index_select', 'masked_select', 'take', 'scatter',
        # Rearrangement
        'roll', 'flip', 'fliplr', 'flipud', 'rot90', 'tile', 'repeat_interleave',
        # Linear algebra
        'mm', 'bmm', 'matmul', 'mv', 'dot', 'inner', 'outer',
        # Diagonal
        'diagonal', 'diag', 'diag_embed', 'tril', 'triu', 'trace',
    ])
    def test_exported(self, name):
        assert hasattr(brute, name), f"brute.{name} not exported"
        assert callable(getattr(brute, name)), f"brute.{name} not callable"

    def test_bit1_dtype_exported(self):
        assert brute.bit1 is not None

    def test_Tensor_class_exported(self):
        assert brute.Tensor is not None
