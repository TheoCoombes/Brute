"""
Comprehensive tests for brute.bit1 behaviour parity with torch.BoolTensor.

Design rules
────────────
1. Every bit1 test has a matching bool-tensor reference computed with plain torch.
2. The factory functions used *inside* tests (e.g. brute.tensor, brute.zeros) are
   themselves tested in TestFactories — so they are safe to use elsewhere.
3. Matmul tests verify the {−1,+1} convention against a manual float reference.
"""

import copy

import pytest
import torch

import brute


# ── helpers ───────────────────────────────────────────────────────────────────

def base(t) -> torch.Tensor:
    """Extract the underlying torch.Tensor (bool base for bit1, plain otherwise)."""
    return t.as_subclass(torch.Tensor)


def ref_bool(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


# ── TestFactories ─────────────────────────────────────────────────────────────

class TestFactories:
    """Test that all brute factory functions produce correct bit1 / float tensors."""

    # --- zeros ----------------------------------------------------------------

    def test_zeros_bit1_values(self):
        t   = brute.zeros(3, 4, dtype=brute.bit1)
        ref = torch.zeros(3, 4, dtype=torch.bool)
        assert isinstance(t, brute.Tensor)
        assert t.dtype == brute.bit1
        assert t.shape == torch.Size([3, 4])
        assert torch.equal(base(t), ref)

    def test_zeros_bit1_tuple_size(self):
        t = brute.zeros((2, 5), dtype=brute.bit1)
        assert t.shape == torch.Size([2, 5])

    def test_zeros_float(self):
        t   = brute.zeros(3, 4)
        ref = torch.zeros(3, 4)
        assert torch.equal(base(t), ref)

    # --- ones -----------------------------------------------------------------

    def test_ones_bit1_values(self):
        t   = brute.ones(3, 4, dtype=brute.bit1)
        ref = torch.ones(3, 4, dtype=torch.bool)
        assert torch.equal(base(t), ref)

    def test_ones_float(self):
        t   = brute.ones(3, 4)
        ref = torch.ones(3, 4)
        assert torch.equal(base(t), ref)

    # --- full -----------------------------------------------------------------

    def test_full_bit1_true(self):
        t   = brute.full((2, 3), True, dtype=brute.bit1)
        ref = torch.full((2, 3), True, dtype=torch.bool)
        assert torch.equal(base(t), ref)

    def test_full_bit1_false(self):
        t   = brute.full((2, 3), False, dtype=brute.bit1)
        ref = torch.full((2, 3), False, dtype=torch.bool)
        assert torch.equal(base(t), ref)

    # --- tensor ---------------------------------------------------------------

    def test_tensor_bit1(self):
        data = [True, False, True, True, False]
        t    = brute.tensor(data, dtype=brute.bit1)
        ref  = torch.tensor(data, dtype=torch.bool)
        assert t.dtype == brute.bit1
        assert torch.equal(base(t), ref)

    def test_tensor_float(self):
        data = [1.0, 2.0, 3.0]
        t    = brute.tensor(data)
        ref  = torch.tensor(data)
        assert torch.equal(base(t), ref)

    # --- as_tensor ------------------------------------------------------------

    def test_as_tensor_bool(self):
        raw = torch.tensor([True, False, True])
        t   = brute.as_tensor(raw, dtype=brute.bit1)
        assert t.dtype == brute.bit1
        assert torch.equal(base(t), raw)

    def test_as_tensor_list(self):
        t   = brute.as_tensor([True, False], dtype=brute.bit1)
        ref = torch.tensor([True, False], dtype=torch.bool)
        assert torch.equal(base(t), ref)

    # --- rand / randn ---------------------------------------------------------

    def test_rand_bit1_shape(self):
        t = brute.rand(4, 5, dtype=brute.bit1)
        assert t.shape == torch.Size([4, 5])
        assert t.dtype == brute.bit1
        assert base(t).dtype == torch.bool

    def test_randn_bit1_shape(self):
        t = brute.randn(4, 5, dtype=brute.bit1)
        assert t.shape == torch.Size([4, 5])
        assert t.dtype == brute.bit1

    def test_rand_bit1_has_both_values(self):
        # With 1000 bits, probability of all-True or all-False is astronomically low
        torch.manual_seed(0)
        t = brute.rand(1000, dtype=brute.bit1)
        assert base(t).any().item()
        assert (~base(t)).any().item()

    def test_rand_float_range(self):
        t = brute.rand(100)
        assert (base(t) >= 0).all() and (base(t) <= 1).all()

    def test_randn_float(self):
        t = brute.randn(50)
        assert t.dtype == torch.float32

    # --- randint --------------------------------------------------------------

    def test_randint_bit1(self):
        t = brute.randint(0, 2, size=(4, 5), dtype=brute.bit1)
        assert t.shape == torch.Size([4, 5])
        assert t.dtype == brute.bit1

    # --- zeros_like / ones_like -----------------------------------------------

    def test_zeros_like_bit1(self):
        src = brute.ones(3, 4, dtype=brute.bit1)
        t   = brute.zeros_like(src)
        assert t.dtype == brute.bit1
        assert not base(t).any()

    def test_ones_like_bit1(self):
        src = brute.zeros(3, 4, dtype=brute.bit1)
        t   = brute.ones_like(src)
        assert t.dtype == brute.bit1
        assert base(t).all()

    # --- arange / linspace / eye ----------------------------------------------

    def test_arange(self):
        t   = brute.arange(5)
        ref = torch.arange(5)
        assert torch.equal(base(t), ref)

    def test_linspace(self):
        t   = brute.linspace(0.0, 1.0, 5)
        ref = torch.linspace(0.0, 1.0, 5)
        assert torch.allclose(base(t), ref)

    def test_eye(self):
        t   = brute.eye(4)
        ref = torch.eye(4)
        assert torch.equal(base(t), ref)

    # --- from_numpy -----------------------------------------------------------

    def test_from_numpy(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t   = brute.from_numpy(arr)
        ref = torch.from_numpy(arr)
        assert torch.equal(base(t), ref)


# ── TestProperties ────────────────────────────────────────────────────────────

class TestProperties:
    def test_dtype_bit1_equals_brute_bit1(self):
        t = brute.zeros(3, dtype=brute.bit1)
        assert t.dtype == brute.bit1

    def test_dtype_bit1_equals_torch_bool(self):
        # bit1 must compare equal to torch.bool for downstream compat
        t = brute.zeros(3, dtype=brute.bit1)
        assert t.dtype == torch.bool

    def test_dtype_float(self):
        t = brute.zeros(3)
        assert t.dtype == torch.float32

    def test_shape(self):
        t = brute.zeros(3, 4, 5, dtype=brute.bit1)
        assert t.shape == torch.Size([3, 4, 5])

    def test_numel(self):
        t = brute.zeros(3, 4, dtype=brute.bit1)
        assert t.numel() == 12

    def test_ndim(self):
        t = brute.zeros(2, 3, dtype=brute.bit1)
        assert t.ndim == 2

    def test_device_cpu(self):
        t = brute.zeros(3, dtype=brute.bit1)
        assert t.device.type == 'cpu'

    def test_isinstance_torch_tensor(self):
        t = brute.zeros(3, dtype=brute.bit1)
        assert isinstance(t, torch.Tensor)
        assert isinstance(t, brute.Tensor)

    def test_pack_dtype_attribute(self):
        t = brute.zeros(3, dtype=brute.bit1, pack_dtype='uint8')
        assert t.pack_dtype == 'uint8'

    def test_no_grad_bit1(self):
        # bool tensors cannot have gradients
        t = brute.zeros(3, dtype=brute.bit1)
        assert not t.requires_grad


# ── TestBoolOps ───────────────────────────────────────────────────────────────

class TestBoolOps:
    """Boolean ops on bit1 should be identical to the same op on BoolTensor."""

    A = [True,  False, True,  False]
    B = [True,  True,  False, False]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.a     = brute.tensor(self.A, dtype=brute.bit1)
        self.b     = brute.tensor(self.B, dtype=brute.bit1)
        self.ref_a = torch.tensor(self.A)
        self.ref_b = torch.tensor(self.B)

    def _chk(self, result, ref):
        assert isinstance(result, brute.Tensor)
        assert result.dtype == brute.bit1
        assert torch.equal(base(result), ref)

    def test_and(self):
        self._chk(self.a & self.b, self.ref_a & self.ref_b)

    def test_or(self):
        self._chk(self.a | self.b, self.ref_a | self.ref_b)

    def test_xor(self):
        self._chk(self.a ^ self.b, self.ref_a ^ self.ref_b)

    def test_not(self):
        self._chk(~self.a, ~self.ref_a)

    def test_eq(self):
        result = self.a == self.b
        ref    = self.ref_a == self.ref_b
        # equality returns bool; re-wrapping as bit1 is correct
        assert torch.equal(base(result), ref)

    def test_ne(self):
        result = self.a != self.b
        ref    = self.ref_a != self.ref_b
        assert torch.equal(base(result), ref)

    def test_torch_logical_and(self):
        result = torch.logical_and(self.a, self.b)
        ref    = torch.logical_and(self.ref_a, self.ref_b)
        assert torch.equal(base(result), ref)

    def test_torch_logical_or(self):
        result = torch.logical_or(self.a, self.b)
        ref    = torch.logical_or(self.ref_a, self.ref_b)
        assert torch.equal(base(result), ref)

    def test_torch_logical_not(self):
        result = torch.logical_not(self.a)
        ref    = torch.logical_not(self.ref_a)
        assert torch.equal(base(result), ref)

    def test_torch_logical_xor(self):
        result = torch.logical_xor(self.a, self.b)
        ref    = torch.logical_xor(self.ref_a, self.ref_b)
        assert torch.equal(base(result), ref)


# ── TestReductions ────────────────────────────────────────────────────────────

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
        result = brute.all(self.t, dim=0)
        ref    = torch.all(self.ref, dim=0)
        assert torch.equal(base(result), ref)

    def test_any_dim1(self):
        result = brute.any(self.t, dim=1)
        ref    = torch.any(self.ref, dim=1)
        assert torch.equal(base(result), ref)

    def test_sum(self):
        assert brute.sum(self.t).item() == torch.sum(self.ref).item()

    def test_count_nonzero(self):
        r1 = torch.count_nonzero(self.t).item()
        r2 = torch.count_nonzero(self.ref).item()
        assert r1 == r2

    def test_all_false_tensor(self):
        t   = brute.zeros(4, dtype=brute.bit1)
        assert not brute.all(t).item()

    def test_all_true_tensor(self):
        t   = brute.ones(4, dtype=brute.bit1)
        assert brute.all(t).item()

    def test_any_false_tensor(self):
        t   = brute.zeros(4, dtype=brute.bit1)
        assert not brute.any(t).item()

    def test_any_true_tensor(self):
        t   = brute.ones(4, dtype=brute.bit1)
        assert brute.any(t).item()


# ── TestIndexing ──────────────────────────────────────────────────────────────

class TestIndexing:
    DATA = [[True, False, True], [False, True, False]]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t   = brute.tensor(self.DATA, dtype=brute.bit1)
        self.ref = torch.tensor(self.DATA)

    def test_row_index(self):
        result = self.t[0]
        ref    = self.ref[0]
        assert isinstance(result, brute.Tensor)
        assert result.dtype == brute.bit1
        assert torch.equal(base(result), ref)

    def test_col_slice(self):
        result = self.t[:, 1:]
        ref    = self.ref[:, 1:]
        assert torch.equal(base(result), ref)

    def test_scalar_index_item(self):
        assert self.t[0, 0].item() == self.ref[0, 0].item()
        assert self.t[0, 1].item() == self.ref[0, 1].item()

    def test_bool_mask(self):
        mask   = torch.tensor([True, False])
        result = self.t[mask]
        ref    = self.ref[mask]
        assert torch.equal(base(result).bool(), ref)

    def test_advanced_index(self):
        result = self.t[[0, 1], [2, 0]]
        ref    = self.ref[[0, 1], [2, 0]]
        assert torch.equal(base(result), ref)


# ── TestShape ─────────────────────────────────────────────────────────────────

class TestShape:
    DATA = [[True, False, True, False], [False, True, False, True]]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t   = brute.tensor(self.DATA, dtype=brute.bit1)
        self.ref = torch.tensor(self.DATA)

    def test_view(self):
        result = self.t.view(8)
        ref    = self.ref.view(8)
        assert result.shape == torch.Size([8])
        assert torch.equal(base(result), ref)

    def test_reshape(self):
        result = self.t.reshape(4, 2)
        ref    = self.ref.reshape(4, 2)
        assert torch.equal(base(result), ref)

    def test_t(self):
        result = self.t.t()
        ref    = self.ref.t()
        assert torch.equal(base(result), ref)

    def test_permute_3d(self):
        d   = [[[True, False], [True, True]], [[False, True], [False, False]]]
        t   = brute.tensor(d, dtype=brute.bit1)
        ref = torch.tensor(d)
        assert torch.equal(base(t.permute(2, 0, 1)), ref.permute(2, 0, 1))

    def test_flatten(self):
        result = self.t.flatten()
        ref    = self.ref.flatten()
        assert torch.equal(base(result), ref)

    def test_unsqueeze(self):
        result = self.t.unsqueeze(0)
        ref    = self.ref.unsqueeze(0)
        assert result.shape == torch.Size([1, 2, 4])
        assert torch.equal(base(result), ref)

    def test_squeeze(self):
        t   = brute.tensor([[[True, False]]], dtype=brute.bit1)
        ref = torch.tensor([[[True, False]]])
        assert torch.equal(base(t.squeeze()), ref.squeeze())

    def test_expand(self):
        t   = brute.tensor([[True], [False]], dtype=brute.bit1)
        ref = torch.tensor([[True], [False]])
        assert torch.equal(base(t.expand(2, 3)), ref.expand(2, 3))

    def test_contiguous(self):
        result = self.t.t().contiguous()
        ref    = self.ref.t().contiguous()
        assert torch.equal(base(result), ref)


# ── TestCat ───────────────────────────────────────────────────────────────────

class TestCat:
    def test_cat_dim0(self):
        a   = brute.tensor([True, False], dtype=brute.bit1)
        b   = brute.tensor([False, True], dtype=brute.bit1)
        r   = torch.cat([a, b])
        ref = torch.cat([torch.tensor([True, False]), torch.tensor([False, True])])
        assert isinstance(r, brute.Tensor)
        assert r.dtype == brute.bit1
        assert torch.equal(base(r), ref)

    def test_cat_dim1(self):
        a   = brute.zeros(3, 2, dtype=brute.bit1)
        b   = brute.ones(3, 2, dtype=brute.bit1)
        r   = torch.cat([a, b], dim=1)
        ref = torch.cat([torch.zeros(3, 2, dtype=torch.bool),
                         torch.ones(3, 2, dtype=torch.bool)], dim=1)
        assert torch.equal(base(r), ref)

    def test_stack(self):
        a   = brute.tensor([True, False], dtype=brute.bit1)
        b   = brute.tensor([False, True], dtype=brute.bit1)
        r   = torch.stack([a, b])
        ref = torch.stack([torch.tensor([True, False]), torch.tensor([False, True])])
        assert torch.equal(base(r), ref)

    def test_brute_cat(self):
        a = brute.ones(4, dtype=brute.bit1)
        b = brute.zeros(4, dtype=brute.bit1)
        r = brute.cat([a, b])
        assert r.shape == torch.Size([8])


# ── TestConversion ────────────────────────────────────────────────────────────

class TestConversion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.t = brute.tensor([True, False, True], dtype=brute.bit1)

    def test_to_bool(self):
        result = self.t.bool()
        assert result.dtype == torch.bool
        assert torch.equal(result, torch.tensor([True, False, True]))

    def test_to_float(self):
        result = self.t.float()
        assert result.dtype == torch.float32
        assert torch.equal(result.as_subclass(torch.Tensor),
                           torch.tensor([1.0, 0.0, 1.0]))

    def test_to_int(self):
        result = self.t.int()
        assert torch.equal(result.as_subclass(torch.Tensor),
                           torch.tensor([1, 0, 1], dtype=torch.int32))

    def test_to_long(self):
        result = self.t.long()
        assert torch.equal(result.as_subclass(torch.Tensor),
                           torch.tensor([1, 0, 1], dtype=torch.int64))

    def test_unpack_pm1(self):
        # bit1 convention: True→+1, False→−1
        result = self.t.unpack_pm1()
        assert torch.equal(result, torch.tensor([1.0, -1.0, 1.0]))

    def test_float_to_bit1_via_to(self):
        ft = brute.tensor([1.5, -0.5, 0.1], dtype=torch.float32)
        t  = ft.to(brute.bit1)
        assert t.dtype == brute.bit1
        # sign convention: >0 → True (matches pack_bits kernel)
        ref = torch.tensor([True, False, True])
        assert torch.equal(base(t), ref)

    def test_bool_to_bit1_via_to(self):
        bt  = brute.tensor([True, False, True], dtype=torch.float32).bool()
        t   = brute.as_tensor(bt, dtype=brute.bit1)
        ref = bt
        assert torch.equal(base(t), ref)

    def test_to_device_preserves_bit1(self):
        t2 = self.t.to('cpu')
        assert t2.dtype == brute.bit1
        assert torch.equal(base(t2), base(self.t))


# ── TestDeepCopy ──────────────────────────────────────────────────────────────

class TestDeepCopy:
    def test_deepcopy_bit1(self):
        t  = brute.tensor([True, False, True], dtype=brute.bit1)
        t2 = copy.deepcopy(t)
        assert t2.dtype == brute.bit1
        assert torch.equal(base(t2), base(t))
        # Ensure it's a distinct copy
        assert t2.data_ptr() != t.data_ptr()

    def test_deepcopy_float(self):
        t  = brute.randn(4)
        t2 = copy.deepcopy(t)
        assert torch.equal(base(t2), base(t))


# ── TestMatmul ────────────────────────────────────────────────────────────────

class TestMatmul:
    def test_bit1_x_bit1_manual(self):
        # a = [+1, +1, -1, +1], b0 = [+1, -1, -1, +1], b1 = [-1, +1, +1, -1]
        # a·b0 = 1-1+1+1 = 2,  a·b1 = -1+1-1-1 = -2
        a = brute.tensor([[True, True, False, True]], dtype=brute.bit1)
        b = brute.tensor([[True, False, False, True],
                          [False, True,  True,  False]], dtype=brute.bit1)
        r = a @ b
        assert r.shape == torch.Size([1, 2])
        assert r[0, 0].item() == 2
        assert r[0, 1].item() == -2

    def test_bit1_x_bit1_vs_float_reference(self):
        torch.manual_seed(0)
        a_bool = torch.randint(0, 2, (8, 16)).bool()
        b_bool = torch.randint(0, 2, (12, 16)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1)
        b = brute.tensor(b_bool, dtype=brute.bit1)
        result = (a @ b).float()

        # Reference: full-precision {−1, +1} dot product
        a_pm = a_bool.float() * 2 - 1
        b_pm = b_bool.float() * 2 - 1
        ref  = a_pm @ b_pm.t()
        assert torch.equal(result, ref)

    def test_torch_matmul_dispatches(self):
        a = brute.ones(4, 8, dtype=brute.bit1)
        b = brute.ones(6, 8, dtype=brute.bit1)
        r = torch.matmul(a, b)
        assert r.shape == torch.Size([4, 6])

    def test_float_x_float(self):
        a = brute.randn(3, 4)
        b = brute.randn(4, 5)
        r = a @ b
        assert r.shape == torch.Size([3, 5])

    def test_all_true_x_all_true(self):
        # a = all +1, b = all +1 → each dot product = K
        K  = 8
        a  = brute.ones(2, K, dtype=brute.bit1)
        b  = brute.ones(3, K, dtype=brute.bit1)
        r  = a @ b
        assert (r == K).all()

    def test_all_false_x_all_false(self):
        # a = all -1, b = all -1 → each dot product = K
        K  = 8
        a  = brute.zeros(2, K, dtype=brute.bit1)
        b  = brute.zeros(2, K, dtype=brute.bit1)
        r  = a @ b
        assert (r == K).all()

    def test_all_true_x_all_false(self):
        # a = all +1, b = all -1 → each dot product = -K
        K  = 8
        a  = brute.ones(2, K, dtype=brute.bit1)
        b  = brute.zeros(3, K, dtype=brute.bit1)
        r  = a @ b
        assert (r == -K).all()


# ── TestUnpackPm1 ─────────────────────────────────────────────────────────────

class TestUnpackPm1:
    """unpack_pm1 is the bridge between bool semantics and {−1,+1} arithmetic."""

    def test_all_true(self):
        t = brute.ones(4, dtype=brute.bit1)
        r = t.unpack_pm1()
        assert torch.equal(r, torch.ones(4))

    def test_all_false(self):
        t = brute.zeros(4, dtype=brute.bit1)
        r = t.unpack_pm1()
        assert torch.equal(r, -torch.ones(4))

    def test_alternating(self):
        t = brute.tensor([True, False, True, False], dtype=brute.bit1)
        r = t.unpack_pm1()
        assert torch.equal(r, torch.tensor([1.0, -1.0, 1.0, -1.0]))

    def test_shape_preserved(self):
        t = brute.rand(3, 4, dtype=brute.bit1)
        r = t.unpack_pm1()
        assert r.shape == torch.Size([3, 4])
        assert r.dtype == torch.float32

    def test_fails_on_non_bit1(self):
        t = brute.zeros(4)
        with pytest.raises(AssertionError):
            t.unpack_pm1()


# ── TestMiscTorchCompat ───────────────────────────────────────────────────────

class TestMiscTorchCompat:
    """Spot-checks that common torch ops work on bit1 tensors."""

    def test_torch_where(self):
        cond  = brute.tensor([True, False, True], dtype=brute.bit1)
        x     = torch.tensor([1.0, 2.0, 3.0])
        y     = torch.tensor([4.0, 5.0, 6.0])
        ref   = torch.where(cond.bool(), x, y)
        result = torch.where(cond.bool(), x, y)
        assert torch.equal(result, ref)

    def test_torch_clone(self):
        t  = brute.tensor([True, False], dtype=brute.bit1)
        t2 = t.clone()
        assert t2.dtype == brute.bit1
        assert torch.equal(base(t2), base(t))
        assert t2.data_ptr() != t.data_ptr()

    def test_repr_bit1(self):
        t = brute.tensor([True, False], dtype=brute.bit1)
        r = repr(t)
        assert 'brute.Tensor' in r
        assert 'bit1' in r

    def test_repr_float(self):
        t = brute.tensor([1.0, 2.0])
        r = repr(t)
        assert 'brute.Tensor' in r

    def test_len(self):
        t = brute.zeros(5, dtype=brute.bit1)
        assert len(t) == 5

    def test_iter(self):
        data = [True, False, True]
        t    = brute.tensor(data, dtype=brute.bit1)
        vals = [elem.item() for elem in t]
        assert vals == data


# ── TestZeroDimScalars ────────────────────────────────────────────────────────

class TestZeroDimScalars:
    """0-dim results from reductions and scalar indexing must not crash."""

    def test_scalar_index_true_value(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert t[0, 0].item() is True

    def test_scalar_index_false_value(self):
        t = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        assert t[0, 1].item() is False

    def test_scalar_index_not_bit1(self):
        # A 0-dim result cannot be packed; it should be a plain bool brute.Tensor
        t = brute.tensor([True, False], dtype=brute.bit1)
        s = t[0]
        # s is 0-dim; its dtype should be bool (not bit1) since packing requires >=1 dim
        assert s.dim() == 0
        assert s.item() is True

    def test_global_all_all_true(self):
        t = brute.ones(6, dtype=brute.bit1)
        assert brute.all(t).item() is True

    def test_global_all_not_all_true(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        assert brute.all(t).item() is False

    def test_global_any_has_true(self):
        t = brute.tensor([False, False, True], dtype=brute.bit1)
        assert brute.any(t).item() is True

    def test_global_any_all_false(self):
        t = brute.zeros(5, dtype=brute.bit1)
        assert brute.any(t).item() is False

    def test_iter_elements_match(self):
        data = [True, False, False, True]
        t    = brute.tensor(data, dtype=brute.bit1)
        vals = [elem.item() for elem in t]
        assert vals == data

    def test_tolist(self):
        data = [True, False, True]
        t    = brute.tensor(data, dtype=brute.bit1)
        assert t.bool().tolist() == data

    def test_min_global(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        assert brute.min(t).item() is False

    def test_max_global(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        assert brute.max(t).item() is True


# ── TestPackDtypes ────────────────────────────────────────────────────────────

class TestPackDtypes:
    """Verify uint8 / uint32 / uint64 packing round-trips and matmul correctness."""

    @pytest.mark.parametrize("pw", ['uint8', 'uint32', 'uint64'])
    def test_pack_dtype_preserved(self, pw):
        t = brute.zeros(8, dtype=brute.bit1, pack_dtype=pw)
        assert t.pack_dtype == pw

    @pytest.mark.parametrize("pw", ['uint8', 'uint32', 'uint64'])
    def test_unpack_pm1_all_true(self, pw):
        t = brute.ones(8, dtype=brute.bit1, pack_dtype=pw)
        r = t.unpack_pm1()
        assert torch.equal(r, torch.ones(8))

    @pytest.mark.parametrize("pw", ['uint8', 'uint32', 'uint64'])
    def test_unpack_pm1_all_false(self, pw):
        t = brute.zeros(8, dtype=brute.bit1, pack_dtype=pw)
        r = t.unpack_pm1()
        assert torch.equal(r, -torch.ones(8))

    @pytest.mark.parametrize("pw", ['uint8', 'uint32', 'uint64'])
    def test_unpack_pm1_alternating(self, pw):
        t = brute.tensor([True, False, True, False, True, False, True, False],
                         dtype=brute.bit1, pack_dtype=pw)
        r = t.unpack_pm1()
        ref = torch.tensor([1., -1., 1., -1., 1., -1., 1., -1.])
        assert torch.equal(r, ref)

    @pytest.mark.parametrize("pw", ['uint8', 'uint32', 'uint64'])
    def test_matmul_all_ones_correctness(self, pw):
        K = 16
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype=pw)
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype=pw)
        r = a @ b
        assert r.shape == torch.Size([2, 3])
        assert (r == K).all()

    @pytest.mark.parametrize("pw", ['uint8', 'uint32', 'uint64'])
    def test_matmul_vs_float_reference(self, pw):
        torch.manual_seed(42)
        K = 16
        a_bool = torch.randint(0, 2, (4, K)).bool()
        b_bool = torch.randint(0, 2, (5, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype=pw)
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype=pw)
        result = (a @ b).float()
        ref    = (a_bool.float() * 2 - 1) @ (b_bool.float() * 2 - 1).t()
        assert torch.equal(result, ref)

    @pytest.mark.parametrize("pw", ['uint8', 'uint32', 'uint64'])
    def test_bool_ops_preserve_pack_dtype(self, pw):
        a = brute.ones(8, dtype=brute.bit1, pack_dtype=pw)
        b = brute.zeros(8, dtype=brute.bit1, pack_dtype=pw)
        r = a & b
        assert isinstance(r, brute.Tensor)
        assert r.dtype == brute.bit1


# ── TestMatmulKBoundary ───────────────────────────────────────────────────────

def _matmul_ref(a_bool: torch.Tensor, b_bool: torch.Tensor) -> torch.Tensor:
    """Float-precision {-1,+1} reference matmul."""
    return (a_bool.float() * 2 - 1) @ (b_bool.float() * 2 - 1).t()


class TestMatmulKBoundary:
    """K at and around pack-width boundaries; the 65-bit case is key."""

    # ── single bit ────────────────────────────────────────────────────────────

    def test_k1_all_true(self):
        a = brute.ones(3, 1, dtype=brute.bit1, pack_dtype='uint8')
        b = brute.ones(4, 1, dtype=brute.bit1, pack_dtype='uint8')
        r = a @ b
        assert (r == 1).all()

    def test_k1_vs_float(self):
        torch.manual_seed(1)
        a_bool = torch.randint(0, 2, (3, 1)).bool()
        b_bool = torch.randint(0, 2, (4, 1)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype='uint8')
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype='uint8')
        assert torch.equal((a @ b).float(), _matmul_ref(a_bool, b_bool))

    # ── uint8 boundaries (7 / 8 / 9) ─────────────────────────────────────────

    @pytest.mark.parametrize("K", [7, 8, 9])
    def test_uint8_all_ones(self, K):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype='uint8')
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype='uint8')
        assert (a @ b == K).all()

    @pytest.mark.parametrize("K", [7, 8, 9])
    def test_uint8_vs_float(self, K):
        torch.manual_seed(K)
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype='uint8')
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype='uint8')
        assert torch.equal((a @ b).float(), _matmul_ref(a_bool, b_bool))

    # ── uint32 boundaries (31 / 32 / 33) ─────────────────────────────────────

    @pytest.mark.parametrize("K", [31, 32, 33])
    def test_uint32_all_ones(self, K):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype='uint32')
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype='uint32')
        assert (a @ b == K).all()

    @pytest.mark.parametrize("K", [31, 32, 33])
    def test_uint32_vs_float(self, K):
        torch.manual_seed(K)
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype='uint32')
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype='uint32')
        assert torch.equal((a @ b).float(), _matmul_ref(a_bool, b_bool))

    # ── uint64 boundaries (63 / 64 / 65) ─────────────────────────────────────

    @pytest.mark.parametrize("K", [63, 64, 65])
    def test_uint64_all_ones(self, K):
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype='uint64')
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype='uint64')
        assert (a @ b == K).all()

    @pytest.mark.parametrize("K", [63, 64, 65])
    def test_uint64_vs_float(self, K):
        torch.manual_seed(K)
        a_bool = torch.randint(0, 2, (3, K)).bool()
        b_bool = torch.randint(0, 2, (4, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype='uint64')
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype='uint64')
        assert torch.equal((a @ b).float(), _matmul_ref(a_bool, b_bool))

    def test_k65_uint64_mixed_values(self):
        """65-bit tensor: 2 uint64 words needed; padding must not corrupt result."""
        torch.manual_seed(7)
        K = 65
        a_bool = torch.randint(0, 2, (5, K)).bool()
        b_bool = torch.randint(0, 2, (6, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype='uint64')
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype='uint64')
        assert torch.equal((a @ b).float(), _matmul_ref(a_bool, b_bool))

    def test_large_K_uint8(self):
        K = 256
        a = brute.ones(2, K, dtype=brute.bit1, pack_dtype='uint8')
        b = brute.ones(3, K, dtype=brute.bit1, pack_dtype='uint8')
        assert (a @ b == K).all()

    def test_large_K_vs_float_random(self):
        torch.manual_seed(99)
        K = 128
        a_bool = torch.randint(0, 2, (4, K)).bool()
        b_bool = torch.randint(0, 2, (5, K)).bool()
        a = brute.tensor(a_bool, dtype=brute.bit1, pack_dtype='uint32')
        b = brute.tensor(b_bool, dtype=brute.bit1, pack_dtype='uint32')
        assert torch.equal((a @ b).float(), _matmul_ref(a_bool, b_bool))


# ── TestEmptyAndSingleton ─────────────────────────────────────────────────────

class TestEmptyAndSingleton:
    """Zero-element and single-element edge cases."""

    def test_empty_1d_shape(self):
        t = brute.zeros(0, dtype=brute.bit1)
        assert t.shape == torch.Size([0])
        assert t.dtype == brute.bit1
        assert t.numel() == 0

    def test_empty_2d_shape(self):
        t = brute.zeros(0, 4, dtype=brute.bit1)
        assert t.shape == torch.Size([0, 4])

    def test_empty_sum(self):
        t = brute.zeros(0, dtype=brute.bit1)
        assert brute.sum(t).item() == 0

    def test_empty_all_vacuous(self):
        t = brute.zeros(0, dtype=brute.bit1)
        assert brute.all(t).item() is True  # vacuous truth

    def test_empty_any_false(self):
        t = brute.zeros(0, dtype=brute.bit1)
        assert brute.any(t).item() is False

    def test_singleton_1d(self):
        t = brute.tensor([True], dtype=brute.bit1)
        assert t.shape == torch.Size([1])
        assert t[0].item() is True

    def test_singleton_2d_1x1(self):
        t = brute.tensor([[True]], dtype=brute.bit1)
        assert t.shape == torch.Size([1, 1])
        assert t[0, 0].item() is True

    def test_singleton_matmul(self):
        # 1×1 @ 1×1 = 1×1; [True]@[True]=[+1]*[+1]=1
        a = brute.tensor([[True]], dtype=brute.bit1)
        b = brute.tensor([[True]], dtype=brute.bit1)
        assert (a @ b).item() == 1

    def test_singleton_matmul_false(self):
        # [False]@[False]=[-1]*[-1]=1
        a = brute.zeros(1, 1, dtype=brute.bit1)
        b = brute.zeros(1, 1, dtype=brute.bit1)
        assert (a @ b).item() == 1

    def test_singleton_matmul_mixed(self):
        # [True]@[False]=[+1]*[-1]=-1
        a = brute.ones(1, 1, dtype=brute.bit1)
        b = brute.zeros(1, 1, dtype=brute.bit1)
        assert (a @ b).item() == -1

    def test_cat_empty_and_nonempty(self):
        empty  = brute.zeros(0, dtype=brute.bit1)
        nonempty = brute.ones(3, dtype=brute.bit1)
        r = torch.cat([empty, nonempty])
        assert r.shape == torch.Size([3])
        assert r.dtype == brute.bit1


# ── TestReprEdgeCases ─────────────────────────────────────────────────────────

class TestReprEdgeCases:
    """repr must always show dtype=brute.bit1 regardless of tensor size."""

    def test_small_bit1_has_bit1_dtype(self):
        # PyTorch omits dtype for small bool tensors; we inject it
        t = brute.tensor([True, False], dtype=brute.bit1)
        assert 'bit1' in repr(t)

    def test_large_bit1_has_bit1_dtype(self):
        t = brute.ones(200, dtype=brute.bit1)
        assert 'bit1' in repr(t)

    def test_2d_bit1_has_bit1_dtype(self):
        t = brute.zeros(3, 4, dtype=brute.bit1)
        assert 'bit1' in repr(t)

    def test_repr_has_brute_tensor(self):
        t = brute.zeros(4, dtype=brute.bit1)
        assert 'brute.Tensor' in repr(t)
        assert 'tensor(' not in repr(t)

    def test_float_repr_has_no_bit1(self):
        t = brute.randn(4)
        r = repr(t)
        assert 'brute.Tensor' in r
        assert 'bit1' not in r

    def test_bit1_repr_no_torch_bool(self):
        # torch.bool should not appear raw; replaced with brute.bit1
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        assert 'torch.bool' not in repr(t)

    def test_singleton_bit1_repr(self):
        t = brute.tensor([True], dtype=brute.bit1)
        assert 'bit1' in repr(t)


# ── TestReduceExtended ────────────────────────────────────────────────────────

class TestReduceExtended:
    """keepdim, dim on 3-D tensors, and additional reduction ops."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = [[[True, False, True], [False, True, False]],
                [[True, True, False], [False, False, True]]]
        self.t3   = brute.tensor(data, dtype=brute.bit1)
        self.ref3 = torch.tensor(data)

    def test_all_keepdim_dim0(self):
        r   = brute.all(self.t3, dim=0, keepdim=True)
        ref = torch.all(self.ref3, dim=0, keepdim=True)
        assert r.shape == ref.shape
        assert torch.equal(base(r), ref)

    def test_any_keepdim_dim1(self):
        r   = brute.any(self.t3, dim=1, keepdim=True)
        ref = torch.any(self.ref3, dim=1, keepdim=True)
        assert r.shape == ref.shape
        assert torch.equal(base(r), ref)

    def test_all_dim2(self):
        r   = brute.all(self.t3, dim=2)
        ref = torch.all(self.ref3, dim=2)
        assert torch.equal(base(r), ref)

    def test_any_dim0(self):
        r   = brute.any(self.t3, dim=0)
        ref = torch.any(self.ref3, dim=0)
        assert torch.equal(base(r), ref)

    def test_sum_dim(self):
        r   = brute.sum(self.t3, dim=2)
        ref = torch.sum(self.ref3, dim=2)
        assert torch.equal(r.as_subclass(torch.Tensor), ref)

    def test_sum_keepdim(self):
        r   = brute.sum(self.t3, dim=1, keepdim=True)
        ref = torch.sum(self.ref3, dim=1, keepdim=True)
        assert r.shape == ref.shape

    def test_count_nonzero_dim(self):
        r   = torch.count_nonzero(self.t3, dim=2)
        ref = torch.count_nonzero(self.ref3, dim=2)
        assert torch.equal(r.as_subclass(torch.Tensor), ref)

    def test_mean_global(self):
        # mean() requires float; cast first
        t   = brute.tensor([True, False, True, True], dtype=brute.bit1)
        ref = torch.tensor([True, False, True, True]).float().mean()
        assert abs(brute.mean(t.float()).item() - ref.item()) < 1e-6

    def test_prod_all_true(self):
        t = brute.ones(4, dtype=brute.bit1)
        assert brute.prod(t).item() == 1


# ── TestIndexingEdgeCases ─────────────────────────────────────────────────────

class TestIndexingEdgeCases:
    DATA = [[True, False, True, False], [False, True, False, True]]

    @pytest.fixture(autouse=True)
    def setup(self):
        self.t   = brute.tensor(self.DATA, dtype=brute.bit1)
        self.ref = torch.tensor(self.DATA)

    def test_negative_row_index(self):
        assert torch.equal(base(self.t[-1]), self.ref[-1])

    def test_negative_col_index(self):
        assert torch.equal(base(self.t[:, -1]), self.ref[:, -1])

    def test_step_slice(self):
        assert torch.equal(base(self.t[:, ::2]), self.ref[:, ::2])

    def test_reverse_flip(self):
        # PyTorch doesn't support ::-1 step slicing; use torch.flip
        assert torch.equal(base(torch.flip(self.t, [1])), torch.flip(self.ref, [1]))

    def test_ellipsis(self):
        t3  = brute.tensor([[[True, False], [True, True]]], dtype=brute.bit1)
        ref = torch.tensor([[[True, False], [True, True]]])
        assert torch.equal(base(t3[..., 0]), ref[..., 0])

    def test_3d_indexing(self):
        data = [[[True, False], [False, True]], [[True, True], [False, False]]]
        t    = brute.tensor(data, dtype=brute.bit1)
        ref  = torch.tensor(data)
        assert torch.equal(base(t[0, 1, :]), ref[0, 1, :])

    def test_clone_after_slice_is_bit1(self):
        row = self.t[0].clone()
        assert row.dtype == brute.bit1
        assert row.shape == torch.Size([4])

    def test_advanced_index_1d(self):
        idx    = torch.tensor([0, 2, 3])
        result = self.t[0][idx]
        ref    = self.ref[0][idx]
        assert torch.equal(base(result), ref)


# ── TestBoolOpsBroadcast ──────────────────────────────────────────────────────

class TestBoolOpsBroadcast:
    """Boolean ops with broadcast shapes."""

    def test_and_broadcast_col_row(self):
        col = brute.tensor([[True], [False]], dtype=brute.bit1)  # (2,1)
        row = brute.tensor([[True, False, True]], dtype=brute.bit1)  # (1,3)
        r   = col & row
        ref = torch.tensor([[True], [False]]) & torch.tensor([[True, False, True]])
        assert torch.equal(base(r), ref)

    def test_or_broadcast_col_row(self):
        col = brute.tensor([[True], [False]], dtype=brute.bit1)
        row = brute.tensor([[False, True, False]], dtype=brute.bit1)
        r   = col | row
        ref = torch.tensor([[True], [False]]) | torch.tensor([[False, True, False]])
        assert torch.equal(base(r), ref)

    def test_xor_same_tensor(self):
        t = brute.tensor([True, False, True], dtype=brute.bit1)
        r = t ^ t
        assert not base(r).any()

    def test_not_then_and(self):
        t   = brute.tensor([True, False, True, False], dtype=brute.bit1)
        ref = torch.tensor([True, False, True, False])
        r   = (~t) & t
        assert not base(r).any()


# ── TestConversionEdgeCases ───────────────────────────────────────────────────

class TestConversionEdgeCases:
    """Extreme numeric values, various source dtypes, and 2-D/3-D unpack."""

    def test_zero_float_to_bit1_is_false(self):
        t = brute.tensor([0.0], dtype=torch.float32)
        b = t.to(brute.bit1)
        assert base(b)[0].item() is False

    def test_negative_float_to_bit1_is_false(self):
        t = brute.tensor([-1.0, -1e6, -0.001], dtype=torch.float32)
        b = t.to(brute.bit1)
        assert not base(b).any()

    def test_positive_float_to_bit1_is_true(self):
        t = brute.tensor([0.001, 1.0, 1e6], dtype=torch.float32)
        b = t.to(brute.bit1)
        assert base(b).all()

    def test_int_nonzero_to_bit1_is_true(self):
        t = brute.tensor([1, 2, -1, 100], dtype=torch.int32)
        b = t.to(brute.bit1)
        assert base(b).all()

    def test_int_zero_to_bit1_is_false(self):
        t = brute.tensor([0], dtype=torch.int32)
        b = t.to(brute.bit1)
        assert base(b)[0].item() is False

    def test_to_pack_dtype_uint32(self):
        t = brute.tensor([True, False, True, False], dtype=brute.bit1, pack_dtype='uint8')
        t2 = t.to(brute.bit1, pack_dtype='uint32')
        assert t2.pack_dtype == 'uint32'
        assert torch.equal(base(t2), base(t))

    def test_unpack_pm1_2d(self):
        t   = brute.tensor([[True, False], [False, True]], dtype=brute.bit1)
        r   = t.unpack_pm1()
        ref = torch.tensor([[1., -1.], [-1., 1.]])
        assert torch.equal(r, ref)

    def test_unpack_pm1_3d(self):
        t   = brute.ones(2, 3, 4, dtype=brute.bit1)
        r   = t.unpack_pm1()
        assert r.shape == torch.Size([2, 3, 4])
        assert (r == 1.0).all()

    def test_to_device_cpu_noop(self):
        t  = brute.tensor([True, False], dtype=brute.bit1)
        t2 = t.to('cpu')
        assert t2.dtype == brute.bit1
        assert torch.equal(base(t), base(t2))

    def test_float_round_trip(self):
        t_bit = brute.tensor([True, False, True], dtype=brute.bit1)
        t_flt = t_bit.float()
        t_back = t_flt.to(brute.bit1)
        assert torch.equal(base(t_back), base(t_bit))


# ── TestFunctionalOps ─────────────────────────────────────────────────────────

class TestFunctionalOps:
    """brute.where, brute.stack, brute.cat, and other functional ops."""

    def test_where_bit1_condition(self):
        cond = brute.tensor([True, False, True], dtype=brute.bit1)
        x    = torch.tensor([1., 2., 3.])
        y    = torch.tensor([4., 5., 6.])
        r    = torch.where(cond.bool(), x, y)
        ref  = torch.tensor([1., 5., 3.])
        assert torch.equal(r, ref)

    def test_cat_preserves_dtype(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=brute.bit1)
        r = brute.cat([a, b])
        assert r.dtype == brute.bit1
        assert r.shape == torch.Size([4])

    def test_stack_bit1(self):
        a = brute.tensor([True, False], dtype=brute.bit1)
        b = brute.tensor([False, True], dtype=brute.bit1)
        r = brute.stack([a, b], dim=0)
        assert r.dtype == brute.bit1
        assert r.shape == torch.Size([2, 2])

    def test_stack_vs_ref(self):
        a_data = [True, False, True]
        b_data = [False, True, False]
        a = brute.tensor(a_data, dtype=brute.bit1)
        b = brute.tensor(b_data, dtype=brute.bit1)
        r   = brute.stack([a, b])
        ref = torch.stack([torch.tensor(a_data), torch.tensor(b_data)])
        assert torch.equal(base(r), ref)

    def test_cat_3_tensors(self):
        a = brute.ones(2, dtype=brute.bit1)
        b = brute.zeros(3, dtype=brute.bit1)
        c = brute.ones(1, dtype=brute.bit1)
        r = brute.cat([a, b, c])
        assert r.shape == torch.Size([6])
        assert base(r).sum().item() == 3

    def test_nonzero_bit1(self):
        t   = brute.tensor([False, True, False, True], dtype=brute.bit1)
        idx = torch.nonzero(t)
        assert idx.shape == torch.Size([2, 1])
        assert idx[0].item() == 1
        assert idx[1].item() == 3

    def test_sum_returns_int(self):
        t = brute.ones(5, dtype=brute.bit1)
        s = brute.sum(t)
        assert s.item() == 5

    def test_mean_range(self):
        # mean() requires float; bit1 tensors must be cast first
        t = brute.ones(4, dtype=brute.bit1)
        m = brute.mean(t.float())
        assert abs(m.item() - 1.0) < 1e-6


# ── TestFactoriesExtended ─────────────────────────────────────────────────────

class TestFactoriesExtended:
    """factory edge cases: empty, pack_dtype variants, rand_like, randn_like."""

    def test_empty_bit1_shape(self):
        t = brute.empty(3, 4, dtype=brute.bit1)
        assert t.shape == torch.Size([3, 4])
        assert t.dtype == brute.bit1

    def test_rand_like_bit1(self):
        src = brute.zeros(4, 5, dtype=brute.bit1)
        t   = brute.rand_like(src)
        assert t.dtype == brute.bit1
        assert t.shape == src.shape

    def test_randn_like_bit1(self):
        src = brute.zeros(4, 5, dtype=brute.bit1)
        t   = brute.randn_like(src)
        assert t.dtype == brute.bit1
        assert t.shape == src.shape

    def test_rand_like_float(self):
        src = brute.zeros(4, 5)
        t   = brute.rand_like(src)
        assert t.dtype == torch.float32
        assert (base(t) >= 0).all() and (base(t) <= 1).all()

    def test_zeros_uint32(self):
        t = brute.zeros(4, dtype=brute.bit1, pack_dtype='uint32')
        assert t.pack_dtype == 'uint32'
        assert not base(t).any()

    def test_ones_uint64(self):
        t = brute.ones(4, dtype=brute.bit1, pack_dtype='uint64')
        assert t.pack_dtype == 'uint64'
        assert base(t).all()

    def test_full_large_false(self):
        t = brute.full((100,), False, dtype=brute.bit1)
        assert t.numel() == 100
        assert not base(t).any()

    def test_randint_only_0_1(self):
        t = brute.randint(0, 2, size=(100,))
        assert (base(t) >= 0).all() and (base(t) <= 1).all()

    def test_zeros_like_float(self):
        src = brute.randn(3, 4)
        t   = brute.zeros_like(src)
        assert t.dtype == torch.float32
        assert not base(t).any()

    def test_ones_like_float(self):
        src = brute.randn(3, 4)
        t   = brute.ones_like(src)
        assert (base(t) == 1).all()

    def test_tensor_from_nested_list(self):
        data = [[True, False], [False, True], [True, True]]
        t    = brute.tensor(data, dtype=brute.bit1)
        assert t.shape == torch.Size([3, 2])
        assert t[1, 1].item() is True

    def test_as_tensor_from_torch_bool(self):
        raw = torch.tensor([[True, False], [True, True]])
        t   = brute.as_tensor(raw, dtype=brute.bit1)
        assert t.dtype == brute.bit1
        assert torch.equal(base(t), raw)

    def test_eye_is_brute_tensor(self):
        t = brute.eye(3)
        assert isinstance(t, brute.Tensor)
        assert t.shape == torch.Size([3, 3])

    def test_linspace_brute_tensor(self):
        t = brute.linspace(0., 1., 11)
        assert isinstance(t, brute.Tensor)
        assert t.shape == torch.Size([11])

    def test_arange_brute_tensor(self):
        t = brute.arange(0, 10)
        assert isinstance(t, brute.Tensor)
        assert t.shape == torch.Size([10])

    def test_from_numpy_float32(self):
        import numpy as np
        a = np.ones((3, 3), dtype=np.float32)
        t = brute.from_numpy(a)
        assert t.shape == torch.Size([3, 3])
        assert isinstance(t, brute.Tensor)


# ── TestMultiDevice ───────────────────────────────────────────────────────────

_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


class TestMultiDevice:
    """CPU ↔ MPS transfers and on-device operations."""

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
        assert t_mps.dtype == brute.bit1
        assert t_mps.device.type == 'mps'
        assert torch.equal(base(t_mps).cpu(), base(t_cpu))

    @_mps
    def test_mps_to_cpu_round_trip(self):
        t = brute.tensor([True, False, True, False], dtype=brute.bit1)
        t2 = t.to('mps').to('cpu')
        assert t2.dtype == brute.bit1
        assert torch.equal(base(t2), base(t))

    @_mps
    def test_mps_bool_and(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1, device='mps')
        b = brute.tensor([True, True, False], dtype=brute.bit1, device='mps')
        r = a & b
        assert r.dtype == brute.bit1
        ref = torch.tensor([True, False, False])
        assert torch.equal(base(r).cpu(), ref)

    @_mps
    def test_mps_bool_or(self):
        a = brute.tensor([True, False, True], dtype=brute.bit1, device='mps')
        b = brute.tensor([False, False, False], dtype=brute.bit1, device='mps')
        r = a | b
        ref = torch.tensor([True, False, True])
        assert torch.equal(base(r).cpu(), ref)

    @_mps
    def test_mps_matmul_all_ones(self):
        K  = 16
        a  = brute.ones(3, K, dtype=brute.bit1, device='mps')
        b  = brute.ones(4, K, dtype=brute.bit1, device='mps')
        r  = a @ b
        assert r.shape == torch.Size([3, 4])
        assert (r == K).all()

    @_mps
    def test_mps_packed_buf_on_mps(self):
        t = brute.ones(4, 8, dtype=brute.bit1, device='mps')
        assert t._packed_buf.device.type == 'mps'
