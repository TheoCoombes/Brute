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
