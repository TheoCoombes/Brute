"""
Comprehensive test suite for brute tensor types.

Design rules
────────────
1. Every bit1 test has a matching bool-tensor reference computed with plain torch.
2. bool (brute.bool / torch.bool) and bit1 (brute.bit1) are strictly separate types
   and must never be accidentally coerced into one another.
3. pack_dtype is a plain torch.dtype: brute.uint8 / brute.uint32 / brute.uint64.
4. All matmul results are verified against a float {-1,+1} reference.
5. Edge cases: K=0, K=1, K non-divisible by pw, very large K.
"""

import copy

import pytest
import torch

import brute


# ── Helpers ────────────────────────────────────────────────────────────────────

def base(t) -> torch.Tensor:
    """Extract the underlying torch.Tensor (bool base for bit1, plain otherwise)."""
    return t.as_subclass(torch.Tensor)


def _ref_pm1(a_bool: torch.Tensor, b_bool: torch.Tensor) -> torch.Tensor:
    """Float {-1,+1} reference matmul."""
    return (a_bool.float() * 2 - 1) @ (b_bool.float() * 2 - 1).t()


# ── TestPackDtype ──────────────────────────────────────────────────────────────

class TestPackDtype:
    """pack_dtype is a plain torch.dtype — brute.uint8/uint32/uint64 == torch.uint8/uint32/uint64."""

    def test_pack_dtypes_are_torch_dtypes(self):
        assert brute.uint8  is torch.uint8
        assert brute.uint32 is torch.uint32
        assert brute.uint64 is torch.uint64

    def test_tensor_pack_dtype_uint8(self):
        t = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t.pack_dtype is torch.uint8

    def test_tensor_pack_dtype_uint32(self):
        t = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint32)
        assert t.pack_dtype is torch.uint32

    def test_tensor_pack_dtype_uint64(self):
        t = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint64)
        assert t.pack_dtype is torch.uint64

    def test_default_pack_dtype_is_uint8(self):
        t = brute.zeros(4, dtype=brute.bit1)
        assert t.pack_dtype is torch.uint8

    def test_pack_dtype_none_for_non_bit1(self):
        assert brute.zeros(4).pack_dtype is None

    def test_invalid_pack_dtype_raises(self):
        with pytest.raises(TypeError):
            brute.zeros(4, dtype=brute.bit1, pack_dtype=torch.float32)

    def test_invalid_pack_dtype_int16_raises(self):
        with pytest.raises(TypeError):
            brute.zeros(4, dtype=brute.bit1, pack_dtype=torch.int16)


# ── TestAllDtypes ──────────────────────────────────────────────────────────────

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


# ── TestDtypeSeparation ────────────────────────────────────────────────────────

class TestDtypeSeparation:
    """brute.bool and brute.bit1 are distinct; they must never fuse."""

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
        assert not b._is_bit1
        assert q._is_bit1

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
        assert getattr(r, '_is_bit1', False)
        assert r.dtype == brute.bit1

    def test_and_two_bit1_preserves_bit1(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1)
        b = brute.tensor([True, True, False], dtype=brute.bit1)
        r = a & b
        assert r.dtype == brute.bit1

    def test_bool_ops_on_plain_bool_stay_bool(self):
        a = brute.tensor([True, False], dtype=torch.bool)
        b = brute.tensor([False, True], dtype=torch.bool)
        r = a & b
        assert r.dtype == torch.bool and not getattr(r, '_is_bit1', False)


# ── TestBoolMethod ─────────────────────────────────────────────────────────────

class TestBoolMethod:
    """.bool() must return a brute.Tensor with dtype=torch.bool."""

    def test_bit1_bool_returns_brute_tensor(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        b = t.bool()
        assert isinstance(b, brute.Tensor)
        assert b.dtype == torch.bool and not b._is_bit1

    def test_bit1_bool_values_correct(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        assert torch.equal(base(t.bool()), torch.tensor([True, False, True]))

    def test_float_bool_returns_brute_tensor(self):
        b = brute.tensor([1.5, -0.5, 0.1]).bool()
        assert isinstance(b, brute.Tensor) and b.dtype == torch.bool

    def test_bool_tensor_bool_is_noop(self):
        t = brute.tensor([True, False], dtype=torch.bool)
        b = t.bool()
        assert isinstance(b, brute.Tensor) and b.dtype == torch.bool
        assert torch.equal(base(b), torch.tensor([True, False]))

    def test_bit1_bool_then_bit1_roundtrip(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        t2 = brute.as_tensor(t.bool(), dtype=brute.bit1)
        assert t2.dtype == brute.bit1
        assert torch.equal(base(t2), base(t))


# ── TestElementSize ────────────────────────────────────────────────────────────

class TestElementSize:
    def test_bit1_raises(self):
        with pytest.raises(TypeError, match="element_size"):
            brute.zeros(4, dtype=brute.bit1).element_size()

    def test_float_element_size(self):
        assert brute.zeros(4, dtype=torch.float32).element_size() == 4

    def test_bool_element_size(self):
        assert brute.zeros(4, dtype=torch.bool).element_size() == 1

    def test_packed_buf_nbytes_uint8(self):
        # 8 logical bits → 1 packed uint8 → 1 byte
        assert brute.zeros(8, dtype=brute.bit1, pack_dtype=brute.uint8)._packed_buf.nbytes == 1

    def test_packed_buf_nbytes_uint32(self):
        # 32 bits → 1 uint32 → 4 bytes
        assert brute.zeros(32, dtype=brute.bit1, pack_dtype=brute.uint32)._packed_buf.nbytes == 4

    def test_packed_buf_nbytes_non_multiple(self):
        # 9 bits → 2 packed uint8 → 2 bytes
        assert brute.zeros(9, dtype=brute.bit1, pack_dtype=brute.uint8)._packed_buf.nbytes == 2


# ── TestPopcount ───────────────────────────────────────────────────────────────

class TestPopcount:
    """popcount() for bit1 uses libpopcnt; for bool uses sum."""

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
        assert isinstance(brute.ones(4, dtype=brute.bit1).popcount(), brute.Tensor)

    def test_bit1_returns_int64_scalar(self):
        r = brute.ones(4, dtype=brute.bit1).popcount()
        assert r.dtype == torch.int64 and r.dim() == 0

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_bit1_all_packs(self, pw):
        data = [True, False, True, True, False, True, False, True]
        t = brute.tensor(data, dtype=brute.bit1, pack_dtype=pw)
        assert t.popcount().item() == sum(data)

    def test_bit1_non_multiple_k(self):
        # K=9 (not a multiple of 8); padding bits must not be counted
        t = brute.tensor([True] * 9, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t.popcount().item() == 9

    def test_bit1_k65_no_padding_inflation(self):
        # K=65 with uint64: 2 words, 63 padding bits all zero
        t = brute.tensor([True] * 65, dtype=brute.bit1, pack_dtype=brute.uint64)
        assert t.popcount().item() == 65

    def test_bool_all_true(self):
        assert brute.ones(6, dtype=torch.bool).popcount().item() == 6

    def test_bool_all_false(self):
        assert brute.zeros(6, dtype=torch.bool).popcount().item() == 0

    def test_bool_mixed(self):
        assert brute.tensor([True, False, True], dtype=torch.bool).popcount().item() == 2

    def test_bool_returns_brute_tensor(self):
        assert isinstance(brute.ones(4, dtype=torch.bool).popcount(), brute.Tensor)

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


# ── TestPaddingEdgeCases ───────────────────────────────────────────────────────

class TestPaddingEdgeCases:
    """K not divisible by pack width: ensure no phantom bits bleed in/out."""

    @pytest.mark.parametrize("K,pw", [
        (1, brute.uint8), (7, brute.uint8), (8, brute.uint8), (9, brute.uint8),
        (31, brute.uint32), (32, brute.uint32), (33, brute.uint32),
        (63, brute.uint64), (64, brute.uint64), (65, brute.uint64),
    ])
    def test_matmul_all_ones_correctness(self, K, pw):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=pw)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=pw)
        assert (a @ b == K).all(), f"K={K}, pw={pw}: got {a @ b}"

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

    @pytest.mark.parametrize("K,pw", [
        (7, brute.uint8), (9, brute.uint8), (65, brute.uint64),
    ])
    def test_popcount_non_multiple_k_all_false(self, K, pw):
        assert brute.zeros(K, dtype=brute.bit1, pack_dtype=pw).popcount().item() == 0

    def test_packed_buf_trailing_zeros(self):
        # K=3, uint8: bits 0-2 = data, bits 3-7 = 0 → packed word = 0b00000111 = 7
        t = brute.tensor([True, True, True], dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t._packed_buf[0].item() == 7

    def test_packed_buf_all_false_trailing_zero(self):
        t = brute.tensor([False, False, False], dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t._packed_buf[0].item() == 0


# ── TestFactories ──────────────────────────────────────────────────────────────

class TestFactories:

    def test_zeros_bit1_values(self):
        t   = brute.zeros(3, 4, dtype=brute.bit1)
        ref = torch.zeros(3, 4, dtype=torch.bool)
        assert isinstance(t, brute.Tensor) and t.dtype == brute.bit1
        assert t.shape == torch.Size([3, 4])
        assert torch.equal(base(t), ref)

    def test_zeros_bit1_tuple_size(self):
        assert brute.zeros((2, 5), dtype=brute.bit1).shape == torch.Size([2, 5])

    def test_zeros_float(self):
        assert torch.equal(base(brute.zeros(3, 4)), torch.zeros(3, 4))

    def test_zeros_bit1_pack_dtype(self):
        t = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint32)
        assert t.pack_dtype is torch.uint32

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

    def test_tensor_bit1(self):
        data = [True, False, True, True, False]
        t    = brute.tensor(data, dtype=brute.bit1)
        assert t.dtype == brute.bit1
        assert torch.equal(base(t), torch.tensor(data, dtype=torch.bool))

    def test_tensor_float(self):
        data = [1.0, 2.0, 3.0]
        assert torch.equal(base(brute.tensor(data)), torch.tensor(data))

    def test_tensor_bool_list_is_torch_bool(self):
        t = brute.tensor([True, False])
        assert t.dtype == torch.bool and not t._is_bit1

    def test_as_tensor_bool(self):
        raw = torch.tensor([True, False, True])
        t   = brute.as_tensor(raw, dtype=brute.bit1)
        assert t.dtype == brute.bit1 and torch.equal(base(t), raw)

    def test_as_tensor_list(self):
        t   = brute.as_tensor([True, False], dtype=brute.bit1)
        assert torch.equal(base(t), torch.tensor([True, False], dtype=torch.bool))

    def test_rand_bit1_shape(self):
        t = brute.rand(4, 5, dtype=brute.bit1)
        assert t.shape == torch.Size([4, 5]) and t.dtype == brute.bit1

    def test_randn_bit1_shape(self):
        assert brute.randn(4, 5, dtype=brute.bit1).dtype == brute.bit1

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
        assert t.shape == torch.Size([4, 5]) and t.dtype == brute.bit1

    def test_zeros_like_bit1(self):
        src = brute.ones(3, 4, dtype=brute.bit1)
        t   = brute.zeros_like(src)
        assert t.dtype == brute.bit1 and not base(t).any()

    def test_ones_like_bit1(self):
        src = brute.zeros(3, 4, dtype=brute.bit1)
        t   = brute.ones_like(src)
        assert t.dtype == brute.bit1 and base(t).all()

    def test_zeros_like_preserves_pack_dtype(self):
        src = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint32)
        t   = brute.zeros_like(src)
        assert t.pack_dtype is torch.uint32

    def test_arange(self):
        assert torch.equal(base(brute.arange(5)), torch.arange(5))

    def test_linspace(self):
        assert torch.allclose(base(brute.linspace(0.0, 1.0, 5)), torch.linspace(0.0, 1.0, 5))

    def test_eye(self):
        assert torch.equal(base(brute.eye(4)), torch.eye(4))

    def test_from_numpy(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert torch.equal(base(brute.from_numpy(arr)), torch.from_numpy(arr))


# ── TestProperties ─────────────────────────────────────────────────────────────

class TestProperties:
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

    def test_numel(self):
        assert brute.zeros(3, 4, dtype=brute.bit1).numel() == 12

    def test_ndim(self):
        assert brute.zeros(2, 3, dtype=brute.bit1).ndim == 2

    def test_device_cpu(self):
        assert brute.zeros(3, dtype=brute.bit1).device.type == 'cpu'

    def test_isinstance_torch_tensor(self):
        t = brute.zeros(3, dtype=brute.bit1)
        assert isinstance(t, torch.Tensor) and isinstance(t, brute.Tensor)

    def test_pack_dtype_attribute(self):
        t = brute.zeros(3, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert t.pack_dtype is torch.uint8

    def test_no_grad_bit1(self):
        assert not brute.zeros(3, dtype=brute.bit1).requires_grad


# ── TestBoolOps ────────────────────────────────────────────────────────────────

class TestBoolOps:
    A = [True, False, True,  False]
    B = [True, True,  False, False]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a     = brute.tensor(self.A, dtype=brute.bit1)
        self.b     = brute.tensor(self.B, dtype=brute.bit1)
        self.ref_a = torch.tensor(self.A)
        self.ref_b = torch.tensor(self.B)

    def _chk(self, result, ref):
        assert isinstance(result, brute.Tensor) and result.dtype == brute.bit1
        assert torch.equal(base(result), ref)

    def test_and(self):  self._chk(self.a & self.b, self.ref_a & self.ref_b)
    def test_or(self):   self._chk(self.a | self.b, self.ref_a | self.ref_b)
    def test_xor(self):  self._chk(self.a ^ self.b, self.ref_a ^ self.ref_b)
    def test_not(self):  self._chk(~self.a,          ~self.ref_a)

    def test_eq(self):
        assert torch.equal(base(self.a == self.b), self.ref_a == self.ref_b)

    def test_ne(self):
        assert torch.equal(base(self.a != self.b), self.ref_a != self.ref_b)

    def test_torch_logical_and(self):
        assert torch.equal(base(torch.logical_and(self.a, self.b)),
                           torch.logical_and(self.ref_a, self.ref_b))

    def test_torch_logical_or(self):
        assert torch.equal(base(torch.logical_or(self.a, self.b)),
                           torch.logical_or(self.ref_a, self.ref_b))

    def test_torch_logical_not(self):
        assert torch.equal(base(torch.logical_not(self.a)),
                           torch.logical_not(self.ref_a))

    def test_torch_logical_xor(self):
        assert torch.equal(base(torch.logical_xor(self.a, self.b)),
                           torch.logical_xor(self.ref_a, self.ref_b))

    def test_bool_ops_on_bool_tensor_stay_bool(self):
        a = brute.tensor(self.A, dtype=torch.bool)
        b = brute.tensor(self.B, dtype=torch.bool)
        r = a & b
        assert r.dtype == torch.bool and not getattr(r, '_is_bit1', False)


# ── TestReductions ─────────────────────────────────────────────────────────────

class TestReductions:
    DATA = [[True, False, True], [False, False, True]]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t   = brute.tensor(self.DATA, dtype=brute.bit1)
        self.ref = torch.tensor(self.DATA)

    def test_all_global(self):
        assert brute.all(self.t).item() == torch.all(self.ref).item()

    def test_any_global(self):
        assert brute.any(self.t).item() == torch.any(self.ref).item()

    def test_all_dim0(self):
        assert torch.equal(base(brute.all(self.t, dim=0)), torch.all(self.ref, dim=0))

    def test_any_dim1(self):
        assert torch.equal(base(brute.any(self.t, dim=1)), torch.any(self.ref, dim=1))

    def test_sum(self):
        assert brute.sum(self.t).item() == torch.sum(self.ref).item()

    def test_count_nonzero(self):
        assert torch.count_nonzero(self.t).item() == torch.count_nonzero(self.ref).item()

    def test_all_false_tensor(self):
        assert not brute.all(brute.zeros(4, dtype=brute.bit1)).item()

    def test_all_true_tensor(self):
        assert brute.all(brute.ones(4, dtype=brute.bit1)).item()

    def test_any_false_tensor(self):
        assert not brute.any(brute.zeros(4, dtype=brute.bit1)).item()

    def test_any_true_tensor(self):
        assert brute.any(brute.ones(4, dtype=brute.bit1)).item()


# ── TestIndexing ───────────────────────────────────────────────────────────────

class TestIndexing:
    DATA = [[True, False, True], [False, True, False]]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t   = brute.tensor(self.DATA, dtype=brute.bit1)
        self.ref = torch.tensor(self.DATA)

    def test_row_index(self):
        r = self.t[0]
        assert isinstance(r, brute.Tensor) and r.dtype == brute.bit1
        assert torch.equal(base(r), self.ref[0])

    def test_col_slice(self):
        assert torch.equal(base(self.t[:, 1:]), self.ref[:, 1:])

    def test_scalar_index_item(self):
        assert self.t[0, 0].item() == self.ref[0, 0].item()
        assert self.t[0, 1].item() == self.ref[0, 1].item()

    def test_bool_mask(self):
        mask   = torch.tensor([True, False])
        result = self.t[mask]
        assert torch.equal(base(result).bool(), self.ref[mask])

    def test_advanced_index(self):
        assert torch.equal(base(self.t[[0, 1], [2, 0]]), self.ref[[0, 1], [2, 0]])


# ── TestShape ──────────────────────────────────────────────────────────────────

class TestShape:
    DATA = [[True, False, True, False], [False, True, False, True]]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t   = brute.tensor(self.DATA, dtype=brute.bit1)
        self.ref = torch.tensor(self.DATA)

    def test_view(self):
        assert torch.equal(base(self.t.view(8)), self.ref.view(8))

    def test_reshape(self):
        assert torch.equal(base(self.t.reshape(4, 2)), self.ref.reshape(4, 2))

    def test_t(self):
        assert torch.equal(base(self.t.t()), self.ref.t())

    def test_permute_3d(self):
        d   = [[[True, False], [True, True]], [[False, True], [False, False]]]
        t   = brute.tensor(d, dtype=brute.bit1)
        ref = torch.tensor(d)
        assert torch.equal(base(t.permute(2, 0, 1)), ref.permute(2, 0, 1))

    def test_flatten(self):
        assert torch.equal(base(self.t.flatten()), self.ref.flatten())

    def test_unsqueeze(self):
        r = self.t.unsqueeze(0)
        assert r.shape == torch.Size([1, 2, 4]) and torch.equal(base(r), self.ref.unsqueeze(0))

    def test_squeeze(self):
        t   = brute.tensor([[[True, False]]], dtype=brute.bit1)
        ref = torch.tensor([[[True, False]]])
        assert torch.equal(base(t.squeeze()), ref.squeeze())

    def test_expand(self):
        t   = brute.tensor([[True], [False]], dtype=brute.bit1)
        ref = torch.tensor([[True], [False]])
        assert torch.equal(base(t.expand(2, 3)), ref.expand(2, 3))

    def test_contiguous(self):
        assert torch.equal(base(self.t.t().contiguous()), self.ref.t().contiguous())


# ── TestCat ────────────────────────────────────────────────────────────────────

class TestCat:
    def test_cat_dim0(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=brute.bit1)
        r = torch.cat([a, b])
        ref = torch.cat([torch.tensor([True, False]), torch.tensor([False, True])])
        assert isinstance(r, brute.Tensor) and r.dtype == brute.bit1
        assert torch.equal(base(r), ref)

    def test_cat_dim1(self):
        a   = brute.zeros(3, 2, dtype=brute.bit1)
        b   = brute.ones(3, 2, dtype=brute.bit1)
        r   = torch.cat([a, b], dim=1)
        ref = torch.cat([torch.zeros(3, 2, dtype=torch.bool),
                         torch.ones(3, 2, dtype=torch.bool)], dim=1)
        assert torch.equal(base(r), ref)

    def test_stack(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=brute.bit1)
        r = torch.stack([a, b])
        assert torch.equal(base(r), torch.stack([torch.tensor([True, False]),
                                                  torch.tensor([False, True])]))

    def test_brute_cat(self):
        r = brute.cat([brute.ones(4, dtype=brute.bit1), brute.zeros(4, dtype=brute.bit1)])
        assert r.shape == torch.Size([8])


# ── TestConversion ─────────────────────────────────────────────────────────────

class TestConversion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.t = brute.tensor([True, False, True], dtype=brute.bit1)

    def test_to_bool_returns_brute_tensor(self):
        b = self.t.bool()
        assert isinstance(b, brute.Tensor) and b.dtype == torch.bool and not b._is_bit1

    def test_to_bool_values(self):
        assert torch.equal(base(self.t.bool()), torch.tensor([True, False, True]))

    def test_to_float(self):
        r = self.t.float()
        assert r.dtype == torch.float32
        assert torch.equal(r.as_subclass(torch.Tensor), torch.tensor([1.0, 0.0, 1.0]))

    def test_to_int(self):
        assert torch.equal(self.t.int().as_subclass(torch.Tensor),
                           torch.tensor([1, 0, 1], dtype=torch.int32))

    def test_to_long(self):
        assert torch.equal(self.t.long().as_subclass(torch.Tensor),
                           torch.tensor([1, 0, 1], dtype=torch.int64))

    def test_unpack_pm1(self):
        assert torch.equal(self.t.unpack_pm1(), torch.tensor([1.0, -1.0, 1.0]))

    def test_float_to_bit1_via_to(self):
        t = brute.tensor([1.5, -0.5, 0.1], dtype=torch.float32).to(brute.bit1)
        assert t.dtype == brute.bit1
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
        assert t2.dtype == brute.bit1 and torch.equal(base(t2), base(self.t))

    def test_to_pack_dtype_uint32(self):
        t  = brute.tensor([True, False, True, False], dtype=brute.bit1, pack_dtype=brute.uint8)
        t2 = t.to(brute.bit1, pack_dtype=brute.uint32)
        assert t2.pack_dtype is torch.uint32
        assert torch.equal(base(t2), base(t))

    def test_float_round_trip(self):
        t_bit  = brute.tensor([True, False, True], dtype=brute.bit1)
        t_back = t_bit.float().to(brute.bit1)
        assert torch.equal(base(t_back), base(t_bit))


# ── TestDeepCopy ───────────────────────────────────────────────────────────────

class TestDeepCopy:
    def test_deepcopy_bit1(self):
        t  = brute.tensor([True, False, True], dtype=brute.bit1)
        t2 = copy.deepcopy(t)
        assert t2.dtype == brute.bit1 and torch.equal(base(t2), base(t))
        assert t2.data_ptr() != t.data_ptr()

    def test_deepcopy_preserves_pack_dtype(self):
        t  = brute.zeros(4, dtype=brute.bit1, pack_dtype=brute.uint64)
        t2 = copy.deepcopy(t)
        assert t2.pack_dtype is torch.uint64

    def test_deepcopy_float(self):
        t  = brute.randn(4)
        t2 = copy.deepcopy(t)
        assert torch.equal(base(t2), base(t))


# ── TestMatmul ─────────────────────────────────────────────────────────────────

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


# ── TestUnpackPm1 ──────────────────────────────────────────────────────────────

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
        t   = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert torch.equal(t.unpack_pm1(), torch.tensor([[1., -1.], [-1., 1.]]))

    def test_unpack_3d(self):
        t = brute.ones(2, 3, 4, dtype=brute.bit1)
        r = t.unpack_pm1()
        assert r.shape == torch.Size([2, 3, 4]) and (r == 1.0).all()


# ── TestPackDtypes ─────────────────────────────────────────────────────────────

class TestPackDtypes:
    """Verify all pack dtypes round-trip and compute matmul correctly."""

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_pack_dtype_preserved(self, pw):
        assert brute.zeros(8, dtype=brute.bit1, pack_dtype=pw).pack_dtype is pw

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_unpack_pm1_all_true(self, pw):
        assert torch.equal(brute.ones(8, dtype=brute.bit1, pack_dtype=pw).unpack_pm1(), torch.ones(8))

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_unpack_pm1_all_false(self, pw):
        assert torch.equal(brute.zeros(8, dtype=brute.bit1, pack_dtype=pw).unpack_pm1(), -torch.ones(8))

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_unpack_pm1_alternating(self, pw):
        t   = brute.tensor([True, False, True, False, True, False, True, False],
                           dtype=brute.bit1, pack_dtype=pw)
        ref = torch.tensor([1., -1., 1., -1., 1., -1., 1., -1.])
        assert torch.equal(t.unpack_pm1(), ref)

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_matmul_all_ones(self, pw):
        K = 16
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=pw)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=pw)
        assert (a @ b == K).all()

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_matmul_vs_float_reference(self, pw):
        torch.manual_seed(42)
        K = 16
        a_bool = torch.randint(0, 2, (4, K)).bool()
        b_bool = torch.randint(0, 2, (5, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=pw)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=pw)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

    @pytest.mark.parametrize("pw", [brute.uint8, brute.uint32, brute.uint64])
    def test_bool_ops_preserve_pack_dtype(self, pw):
        a = brute.ones(8, dtype=brute.bit1, pack_dtype=pw)
        b = brute.zeros(8, dtype=brute.bit1, pack_dtype=pw)
        r = a & b
        assert isinstance(r, brute.Tensor) and r.dtype == brute.bit1


# ── TestMatmulKBoundary ────────────────────────────────────────────────────────

class TestMatmulKBoundary:
    def test_k1_all_true(self):
        a = brute.ones(3, 1, dtype=brute.bit1, pack_dtype=brute.uint8)
        b = brute.ones(4, 1, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert (a @ b == 1).all()

    def test_k1_vs_float(self):
        torch.manual_seed(1)
        a_bool = torch.randint(0, 2, (3, 1)).bool()
        b_bool = torch.randint(0, 2, (4, 1)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=brute.uint8)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=brute.uint8)
        assert torch.equal((a @ b).float(), _ref_pm1(a_bool, b_bool))

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

    def test_k65_mixed_values(self):
        torch.manual_seed(7)
        K = 65
        a_bool = torch.randint(0, 2, (5, K)).bool()
        b_bool = torch.randint(0, 2, (6, K)).bool()
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


# ── TestEmptyAndSingleton ──────────────────────────────────────────────────────

class TestEmptyAndSingleton:
    def test_empty_1d_shape(self):
        t = brute.zeros(0, dtype=brute.bit1)
        assert t.shape == torch.Size([0]) and t.dtype == brute.bit1 and t.numel() == 0

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
        assert t.shape == torch.Size([1]) and t[0].item() is True

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
        assert r.shape == torch.Size([3]) and r.dtype == brute.bit1


# ── TestZeroDimScalars ─────────────────────────────────────────────────────────

class TestZeroDimScalars:
    def test_scalar_index_true(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert t[0, 0].item() is True

    def test_scalar_index_false(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert t[0, 1].item() is False

    def test_scalar_0dim_is_not_bit1(self):
        t = brute.tensor([True, False], dtype=brute.bit1)
        s = t[0]
        assert s.dim() == 0 and s.item() is True

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


# ── TestRepr ───────────────────────────────────────────────────────────────────

class TestRepr:
    def test_small_bit1_has_dtype(self):
        assert 'bit1' in repr(brute.tensor([True, False], dtype=brute.bit1))

    def test_large_bit1_has_dtype(self):
        assert 'bit1' in repr(brute.ones(200, dtype=brute.bit1))

    def test_repr_has_brute_tensor(self):
        t = brute.zeros(4, dtype=brute.bit1)
        assert 'brute.Tensor' in repr(t) and 'tensor(' not in repr(t)

    def test_no_torch_bool_in_bit1_repr(self):
        assert 'torch.bool' not in repr(brute.tensor([True, False, True], dtype=brute.bit1))

    def test_float_repr_no_bit1(self):
        r = repr(brute.randn(4))
        assert 'brute.Tensor' in r and 'bit1' not in r

    def test_bool_tensor_repr_no_bit1(self):
        assert 'bit1' not in repr(brute.tensor([True, False], dtype=torch.bool))


# ── TestTorchCompat ────────────────────────────────────────────────────────────

class TestTorchCompat:
    def test_torch_where_with_bit1_cond(self):
        cond   = brute.tensor([True, False, True], dtype=brute.bit1)
        x      = torch.tensor([1.0, 2.0, 3.0])
        y      = torch.tensor([4.0, 5.0, 6.0])
        assert torch.equal(torch.where(cond.bool(), x, y), torch.tensor([1., 5., 3.]))

    def test_torch_clone(self):
        t  = brute.tensor([True, False], dtype=brute.bit1)
        t2 = t.clone()
        assert t2.dtype == brute.bit1 and torch.equal(base(t2), base(t))
        assert t2.data_ptr() != t.data_ptr()

    def test_len(self):
        assert len(brute.zeros(5, dtype=brute.bit1)) == 5

    def test_iter(self):
        data = [True, False, True]
        assert [elem.item() for elem in brute.tensor(data, dtype=brute.bit1)] == data

    def test_plain_torch_tensor_float_matmul(self):
        assert (brute.randn(3, 4) @ torch.randn(4, 5)).shape == torch.Size([3, 5])

    def test_plain_bool_and_bit1_no_promotion(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1)
        b = torch.tensor([True, True, False])
        r = torch.logical_and(a, b)
        assert not getattr(r, '_is_bit1', False)
        assert torch.equal(base(r), torch.logical_and(torch.tensor([True, False, True]), b))

    def test_nonzero_bit1(self):
        t   = brute.tensor([False, True, False, True], dtype=brute.bit1)
        idx = torch.nonzero(t)
        assert idx.shape == torch.Size([2, 1])
        assert idx[0].item() == 1 and idx[1].item() == 3


# ── TestMultiDevice ────────────────────────────────────────────────────────────

_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


class TestMultiDevice:
    @_mps
    def test_mps_zeros_bit1_shape(self):
        t = brute.zeros(3, 4, dtype=brute.bit1, device='mps')
        assert t.device.type == 'mps' and t.shape == torch.Size([3, 4])
        assert t.dtype == brute.bit1

    @_mps
    def test_mps_ones_bit1_values(self):
        t = brute.ones(4, dtype=brute.bit1, device='mps')
        assert t.device.type == 'mps' and base(t).all()

    @_mps
    def test_cpu_to_mps_preserves_bit1(self):
        t_cpu = brute.tensor([True, False, True], dtype=brute.bit1)
        t_mps = t_cpu.to('mps')
        assert t_mps.dtype == brute.bit1 and t_mps.device.type == 'mps'
        assert torch.equal(base(t_mps).cpu(), base(t_cpu))

    @_mps
    def test_mps_to_cpu_round_trip(self):
        t  = brute.tensor([True, False, True, False], dtype=brute.bit1)
        t2 = t.to('mps').to('cpu')
        assert t2.dtype == brute.bit1 and torch.equal(base(t2), base(t))

    @_mps
    def test_mps_bool_and(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1, device='mps')
        b = brute.tensor([True, True, False], dtype=brute.bit1, device='mps')
        r = a & b
        assert r.dtype == brute.bit1
        assert torch.equal(base(r).cpu(), torch.tensor([True, False, False]))

    @_mps
    def test_mps_matmul_all_ones(self):
        K  = 16
        a  = brute.ones(3, K, dtype=brute.bit1, device='mps')
        b  = brute.ones(4, K, dtype=brute.bit1, device='mps')
        r  = a @ b
        assert r.shape == torch.Size([3, 4]) and (r == K).all()

    @_mps
    def test_mps_packed_buf_on_mps(self):
        assert brute.ones(4, 8, dtype=brute.bit1, device='mps')._packed_buf.device.type == 'mps'

    @_mps
    def test_mps_popcount(self):
        assert brute.ones(8, dtype=brute.bit1, device='mps').popcount().item() == 8
