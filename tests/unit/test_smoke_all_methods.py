"""Sweeping smoke tests across every category of torch method that a bit1
tensor might encounter. Every test pairs a bit1 result with a bool oracle and
asserts they materialise to the same bool tensor — catching any silent
miscompilation in our fast paths.

The cases are intentionally small and exhaustive rather than performance-y.
Edge cases (0-dim, empty, multi-dim, non-contiguous) get their own group.
"""
from __future__ import annotations

import pytest
import torch

import brute
from tests.helpers import assert_bit1_matches_bool, bit1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rand_bool(shape, device):
    if isinstance(shape, int):
        shape = (shape,)
    return torch.randint(0, 2, shape, dtype=torch.bool, device=device)


def _both(shape, device, pack_dtype=torch.uint8):
    """Return a (bit1, bool) pair with identical contents."""
    b = _rand_bool(shape, device)
    return bit1(b, pack_dtype=pack_dtype), b


# Common parametrisations
DEVICE_PD = pytest.mark.parametrize(
    "pack_dtype", [torch.uint8, torch.uint32, torch.uint64],
    ids=lambda d: str(d).replace("torch.", "")
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Construction
# ─────────────────────────────────────────────────────────────────────────────

@DEVICE_PD
def test_zeros(device, pack_dtype):
    a = brute.zeros(4, 8, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
    assert_bit1_matches_bool(a, torch.zeros(4, 8, dtype=torch.bool, device=device))


@DEVICE_PD
def test_ones(device, pack_dtype):
    a = brute.ones(4, 8, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
    assert_bit1_matches_bool(a, torch.ones(4, 8, dtype=torch.bool, device=device))


@DEVICE_PD
def test_full_true_false(device, pack_dtype):
    a_t = brute.full((3, 5), True, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
    a_f = brute.full((3, 5), False, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
    assert_bit1_matches_bool(a_t, torch.ones(3, 5, dtype=torch.bool, device=device))
    assert_bit1_matches_bool(a_f, torch.zeros(3, 5, dtype=torch.bool, device=device))


@DEVICE_PD
def test_empty_pad_bits_zero(device, pack_dtype):
    # Pick a last-dim that doesn't align with the pack width.
    a = brute.empty(3, 13, dtype=brute.bit1, device=device, pack_dtype=pack_dtype)
    # popcount == numel iff every bit is 1; can't assume contents. But the
    # invariant is that pad bits are 0, so packed_popcount <= numel.
    assert int(a.popcount()) <= a.numel()


def test_tensor_factory_list(device):
    a = brute.tensor([True, False, True], dtype=brute.bit1, device=device)
    assert_bit1_matches_bool(a, torch.tensor([True, False, True], device=device))


def test_tensor_factory_2d(device):
    a = brute.tensor([[1, 0], [0, 1]], dtype=brute.bit1, device=device)
    assert_bit1_matches_bool(a, torch.tensor([[True, False], [False, True]], device=device))


def test_as_tensor(device):
    src = torch.tensor([True, False, True, True], device=device)
    a = brute.as_tensor(src, dtype=brute.bit1)
    assert_bit1_matches_bool(a, src)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Shape / metadata accessors
# ─────────────────────────────────────────────────────────────────────────────

def test_shape_dim_numel(device):
    a = bit1(_rand_bool((4, 8), device))
    assert tuple(a.shape) == (4, 8)
    assert a.dim() == 2
    assert a.numel() == 32
    assert a.ndim == 2
    assert a.size() == torch.Size([4, 8])
    assert a.size(0) == 4 and a.size(1) == 8


def test_dtype_device_layout(device):
    a = bit1(_rand_bool(8, device))
    assert a.dtype == brute.bit1
    assert a.device.type == device
    assert a.layout == torch.strided


def test_is_contiguous_stride(device):
    a = bit1(_rand_bool((4, 8), device))
    assert a.is_contiguous()
    # Stride matches a contiguous bool tensor of the same shape.
    assert a.stride() == (8, 1)


def test_storage_offset(device):
    a = bit1(_rand_bool(16, device))
    assert a.storage_offset() == 0


def test_element_size_raises_for_bit1(device):
    a = bit1(_rand_bool(8, device))
    with pytest.raises(TypeError):
        _ = a.element_size()


def test_nbytes_reports_packed(device):
    a = brute.zeros(64, dtype=brute.bit1, device=device, pack_dtype=torch.uint8)
    # 64 bits / 8 = 8 bytes of packed storage.
    assert a.nbytes == 8


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Shape ops (view / reshape / squeeze / unsqueeze / etc.)
# ─────────────────────────────────────────────────────────────────────────────

@DEVICE_PD
def test_view_preserve_last_dim(device, pack_dtype):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src, pack_dtype=pack_dtype), src
    assert_bit1_matches_bool(a.view(2, 2, 8), b.view(2, 2, 8))


@DEVICE_PD
def test_view_flat(device, pack_dtype):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src, pack_dtype=pack_dtype), src
    assert_bit1_matches_bool(a.view(32), b.view(32))


def test_view_with_neg_one(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.view(-1), b.view(-1))
    assert_bit1_matches_bool(a.view(-1, 8), b.view(-1, 8))


@DEVICE_PD
def test_reshape_preserve_last_dim(device, pack_dtype):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src, pack_dtype=pack_dtype), src
    assert_bit1_matches_bool(a.reshape(2, 2, 8), b.reshape(2, 2, 8))


def test_reshape_flat(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.reshape(-1), b.reshape(-1))


def test_flatten(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.flatten(), b.flatten())


def test_flatten_partial(device):
    src = _rand_bool((2, 3, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.flatten(0, 1), b.flatten(0, 1))


def test_squeeze_default(device):
    src = _rand_bool((1, 4, 1, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.squeeze(), b.squeeze())


def test_squeeze_dim(device):
    src = _rand_bool((1, 4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.squeeze(0), b.squeeze(0))


def test_unsqueeze(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.unsqueeze(0), b.unsqueeze(0))
    assert_bit1_matches_bool(a.unsqueeze(1), b.unsqueeze(1))


def test_contiguous_already_contig(device):
    a = bit1(_rand_bool((4, 8), device))
    c = a.contiguous()
    assert c.is_contiguous()


def test_contiguous_after_transpose(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    c = a.t().contiguous()
    assert_bit1_matches_bool(c, b.t().contiguous())


def test_t_2d(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.t(), b.t())


def test_transpose(device):
    src = _rand_bool((2, 4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.transpose(0, 1), b.transpose(0, 1))


def test_transpose_leading_axes(device):
    src = _rand_bool((2, 3, 4, 8), device)
    a, b = bit1(src), src
    # Both transposed axes are NON-packed — should hit the leading-axis fast path.
    assert_bit1_matches_bool(a.transpose(0, 2), b.transpose(0, 2))


def test_permute(device):
    src = _rand_bool((2, 4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.permute(2, 0, 1), b.permute(2, 0, 1))


def test_expand(device):
    src = _rand_bool((1, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.expand(4, 8), b.expand(4, 8))


def test_repeat(device):
    src = _rand_bool((4,), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.repeat(3), b.repeat(3))


def test_narrow(device):
    src = _rand_bool((8, 16), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.narrow(0, 2, 4), b.narrow(0, 2, 4))


def test_select(device):
    src = _rand_bool((8, 16), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.select(0, 3), b.select(0, 3))


def test_flip(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.flip(0), b.flip(0))
    assert_bit1_matches_bool(a.flip(1), b.flip(1))


def test_roll(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a.roll(2, dims=0), b.roll(2, dims=0))


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Cat / stack / split
# ─────────────────────────────────────────────────────────────────────────────

def test_cat(device):
    s1 = _rand_bool((2, 8), device)
    s2 = _rand_bool((3, 8), device)
    a, b = bit1(s1), bit1(s2)
    ref = torch.cat([s1, s2], dim=0)
    out = torch.cat([a, b], dim=0)
    assert_bit1_matches_bool(out, ref)


def test_stack(device):
    s1 = _rand_bool((4, 8), device)
    s2 = _rand_bool((4, 8), device)
    ref = torch.stack([s1, s2], dim=0)
    out = torch.stack([bit1(s1), bit1(s2)], dim=0)
    assert_bit1_matches_bool(out, ref)


def test_split(device):
    src = _rand_bool((8, 8), device)
    parts_bool = src.split(4, dim=0)
    parts_bit1 = bit1(src).split(4, dim=0)
    assert len(parts_bit1) == len(parts_bool)
    for pa, pb in zip(parts_bit1, parts_bool):
        assert_bit1_matches_bool(pa, pb)


def test_chunk(device):
    src = _rand_bool((8, 8), device)
    parts_bool = src.chunk(4, dim=0)
    parts_bit1 = bit1(src).chunk(4, dim=0)
    for pa, pb in zip(parts_bit1, parts_bool):
        assert_bit1_matches_bool(pa, pb)


def test_unbind(device):
    src = _rand_bool((4, 8), device)
    parts_bool = src.unbind(0)
    parts_bit1 = bit1(src).unbind(0)
    for pa, pb in zip(parts_bit1, parts_bool):
        assert_bit1_matches_bool(pa, pb)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Indexing & __setitem__
# ─────────────────────────────────────────────────────────────────────────────

def test_getitem_int(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a[2], b[2])


def test_getitem_slice(device):
    src = _rand_bool((8, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a[2:6], b[2:6])


def test_getitem_negative_int(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a[-1], b[-1])


def test_getitem_ellipsis_last_int(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a[..., 3], b[..., 3])


def test_getitem_col_int_2d(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a[:, 3], b[:, 3])


def test_getitem_negative_col(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(a[:, -1], b[:, -1])


def test_getitem_mask(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src), src
    mask = torch.randint(0, 2, (4, 8), dtype=torch.bool, device=device)
    assert_bit1_matches_bool(a[mask], b[mask])


def test_getitem_int_tensor(device):
    src = _rand_bool((8, 8), device)
    a, b = bit1(src), src
    idx = torch.tensor([0, 2, 4], device=device)
    assert_bit1_matches_bool(a[idx], b[idx])


def test_setitem_int_scalar(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src.clone()), src.clone()
    a[1] = True
    b[1] = True
    assert_bit1_matches_bool(a, b)


def test_setitem_slice_scalar(device):
    src = _rand_bool((8, 8), device)
    a, b = bit1(src.clone()), src.clone()
    a[2:6] = False
    b[2:6] = False
    assert_bit1_matches_bool(a, b)


def test_setitem_col_scalar(device):
    src = _rand_bool((4, 8), device)
    a, b = bit1(src.clone()), src.clone()
    a[:, 3] = True
    b[:, 3] = True
    assert_bit1_matches_bool(a, b)


def test_setitem_ellipsis_col_scalar(device):
    src = _rand_bool((2, 3, 8), device)
    a, b = bit1(src.clone()), src.clone()
    a[..., 5] = True
    b[..., 5] = True
    assert_bit1_matches_bool(a, b)


def test_setitem_mask(device):
    src = _rand_bool((4, 8), device)
    mask = torch.randint(0, 2, (4, 8), dtype=torch.bool, device=device)
    a, b = bit1(src.clone()), src.clone()
    a[mask] = True
    b[mask] = True
    assert_bit1_matches_bool(a, b)


def test_index_select(device):
    src = _rand_bool((8, 8), device)
    a, b = bit1(src), src
    idx = torch.tensor([0, 3, 5], device=device)
    assert_bit1_matches_bool(torch.index_select(a, 0, idx),
                             torch.index_select(b, 0, idx))


def test_masked_select(device):
    src = _rand_bool((4, 8), device)
    mask = torch.randint(0, 2, (4, 8), dtype=torch.bool, device=device)
    a, b = bit1(src), src
    assert_bit1_matches_bool(torch.masked_select(a, mask),
                             torch.masked_select(b, mask))


def test_where(device):
    src_a = _rand_bool((4, 8), device)
    src_b = _rand_bool((4, 8), device)
    cond  = torch.randint(0, 2, (4, 8), dtype=torch.bool, device=device)
    a, b = bit1(src_a), bit1(src_b)
    assert_bit1_matches_bool(torch.where(cond, a, b),
                             torch.where(cond, src_a, src_b))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Bitwise / logical
# ─────────────────────────────────────────────────────────────────────────────

@DEVICE_PD
@pytest.mark.parametrize("op", [
    lambda a, b: a & b,
    lambda a, b: a | b,
    lambda a, b: a ^ b,
    lambda a, b: torch.bitwise_and(a, b),
    lambda a, b: torch.bitwise_or(a, b),
    lambda a, b: torch.bitwise_xor(a, b),
    lambda a, b: torch.logical_and(a, b),
    lambda a, b: torch.logical_or(a, b),
    lambda a, b: torch.logical_xor(a, b),
])
def test_binary_bitwise_logical(device, pack_dtype, op):
    sa = _rand_bool((4, 8), device)
    sb = _rand_bool((4, 8), device)
    a, b = bit1(sa, pack_dtype=pack_dtype), bit1(sb, pack_dtype=pack_dtype)
    assert_bit1_matches_bool(op(a, b), op(sa, sb))


@DEVICE_PD
@pytest.mark.parametrize("op", [lambda x: ~x,
                                lambda x: torch.bitwise_not(x),
                                lambda x: torch.logical_not(x)])
def test_unary_not(device, pack_dtype, op):
    sa = _rand_bool((4, 8), device)
    a = bit1(sa, pack_dtype=pack_dtype)
    assert_bit1_matches_bool(op(a), op(sa))


def test_inplace_bitwise_chain(device):
    src_a = _rand_bool((4, 8), device)
    src_b = _rand_bool((4, 8), device)
    src_c = _rand_bool((4, 8), device)
    a = bit1(src_a.clone()); b = bit1(src_b); c = bit1(src_c)
    a &= b
    a |= c
    a ^= b
    ref = src_a.clone() & src_b
    ref |= src_c
    ref ^= src_b
    assert_bit1_matches_bool(a, ref)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Comparison
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op_name", ["eq", "ne"])
def test_eq_ne_dunder_and_func(device, op_name):
    sa = _rand_bool((4, 8), device)
    sb = _rand_bool((4, 8), device)
    a, b = bit1(sa), bit1(sb)
    if op_name == "eq":
        assert_bit1_matches_bool(a == b, sa == sb)
        assert_bit1_matches_bool(torch.eq(a, b), torch.eq(sa, sb))
    else:
        assert_bit1_matches_bool(a != b, sa != sb)
        assert_bit1_matches_bool(torch.ne(a, b), torch.ne(sa, sb))


def test_torch_equal_true(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src); b = bit1(src.clone())
    assert torch.equal(a, b) is True


def test_torch_equal_false(device):
    src = _rand_bool((4, 8), device)
    mut = src.clone(); mut[0, 0] = ~mut[0, 0]
    a = bit1(src); b = bit1(mut)
    assert torch.equal(a, b) is False


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Reductions
# ─────────────────────────────────────────────────────────────────────────────

def test_sum_full(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    assert int(torch.sum(a)) == int(src.long().sum())
    assert int(a.sum()) == int(src.long().sum())


def test_count_nonzero_full(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    assert int(torch.count_nonzero(a)) == int(src.long().sum())


def test_all_any(device):
    s_zero = torch.zeros(8, dtype=torch.bool, device=device)
    s_one  = torch.ones(8, dtype=torch.bool, device=device)
    s_mix  = torch.tensor([True, False, True], device=device)
    for src in (s_zero, s_one, s_mix):
        a = bit1(src)
        assert bool(torch.all(a)) == bool(torch.all(src))
        assert bool(torch.any(a)) == bool(torch.any(src))


def test_sum_dim(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    for dim in (0, 1, -1):
        ref = src.long().sum(dim=dim)
        out = torch.sum(a, dim=dim)
        assert torch.equal(out.cpu(), ref.cpu())


def test_all_any_dim(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    for dim in (0, 1):
        assert torch.equal(torch.all(a, dim=dim).bool().cpu(),
                           torch.all(src, dim=dim).cpu())
        assert torch.equal(torch.any(a, dim=dim).bool().cpu(),
                           torch.any(src, dim=dim).cpu())


def test_argmax_argmin(device):
    src = _rand_bool((16,), device)
    a = bit1(src)
    assert int(torch.argmax(a)) == int(torch.argmax(src.long()))
    assert int(torch.argmin(a)) == int(torch.argmin(src.long()))


def test_nonzero(device):
    src = torch.tensor([True, False, True, True, False, True], device=device)
    a = bit1(src)
    bit_nz = torch.nonzero(a).cpu()
    ref_nz = torch.nonzero(src).cpu()
    assert torch.equal(bit_nz, ref_nz)


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Matmul (still bit1 only at 2-D)
# ─────────────────────────────────────────────────────────────────────────────

def test_matmul_2d(device):
    K = 64
    a_src = _rand_bool((4, K), device)
    b_src = _rand_bool((3, K), device)
    a = bit1(a_src); b = bit1(b_src)
    out = a @ b
    # Verify against the bool-style equivalent: out[i,j] = sum_k 1{a[i,k] == b[j,k]} - K + sum_k 1{a == b}
    # Easier: compare to ref via xnor-popcount semantics: C = K - 2*hamming
    # Here, just verify shape & dtype and that out matches the float-equivalent.
    pm1_a = (a_src.float() * 2 - 1)
    pm1_b = (b_src.float() * 2 - 1)
    ref = pm1_a @ pm1_b.t()
    assert torch.allclose(out.float().cpu(), ref.cpu(), atol=1e-3)


def test_matmul_a_at_bt(device):
    K = 32
    a_src = _rand_bool((4, K), device)
    b_src = _rand_bool((3, K), device)
    a = bit1(a_src); b = bit1(b_src)
    out_via_t = a @ b.t()
    out_direct = a @ b
    # Both forms compute the same logical product.
    assert torch.equal(out_via_t.float().cpu(), out_direct.float().cpu())


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — In-place ops
# ─────────────────────────────────────────────────────────────────────────────

def test_fill_true(device):
    a = bit1(_rand_bool((4, 8), device))
    a.fill_(True)
    assert_bit1_matches_bool(a, torch.ones(4, 8, dtype=torch.bool, device=device))


def test_fill_false(device):
    a = bit1(_rand_bool((4, 8), device))
    a.fill_(False)
    assert_bit1_matches_bool(a, torch.zeros(4, 8, dtype=torch.bool, device=device))


def test_zero_(device):
    a = bit1(torch.ones(8, dtype=torch.bool, device=device))
    a.zero_()
    assert_bit1_matches_bool(a, torch.zeros(8, dtype=torch.bool, device=device))


def test_copy_(device):
    a = bit1(torch.zeros(8, dtype=torch.bool, device=device))
    src = torch.tensor([True, False] * 4, device=device)
    a.as_subclass(torch.Tensor).copy_(src)
    assert_bit1_matches_bool(a, src)


def test_masked_fill_(device):
    src = _rand_bool((4, 8), device)
    mask = torch.randint(0, 2, (4, 8), dtype=torch.bool, device=device)
    a, b = bit1(src.clone()), src.clone()
    a.masked_fill_(mask, True)
    b.masked_fill_(mask, True)
    assert_bit1_matches_bool(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 — Type conversion
# ─────────────────────────────────────────────────────────────────────────────

def test_bool_method(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    out = a.bool().as_subclass(torch.Tensor)
    assert out.dtype == torch.bool
    assert torch.equal(out.cpu(), src.cpu())


def test_to_dtype_float(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    out = a.to(torch.float32).as_subclass(torch.Tensor)
    assert out.dtype == torch.float32
    assert torch.allclose(out.cpu(), src.float().cpu())


def test_to_dtype_int(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    out = a.to(torch.int64).as_subclass(torch.Tensor)
    assert out.dtype == torch.int64
    assert torch.equal(out.cpu(), src.long().cpu())


def test_to_device_roundtrip(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    # Round-trip via CPU to verify ops on different devices.
    cpu_copy = a.to('cpu')
    assert_bit1_matches_bool(cpu_copy, src.cpu())


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — Numpy / list / item interop
# ─────────────────────────────────────────────────────────────────────────────

def test_numpy(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    arr = a.cpu().numpy()
    assert arr.dtype.kind == 'b'
    import numpy as np
    assert np.array_equal(arr, src.cpu().numpy())


def test_tolist(device):
    src = torch.tensor([True, False, True], device=device)
    a = bit1(src)
    assert a.tolist() == src.tolist()


def test_item_0dim(device):
    a = bit1(torch.tensor(True, device=device))
    assert bool(a.item()) is True


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 — Clone / detach / new_*
# ─────────────────────────────────────────────────────────────────────────────

def test_clone(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    c = a.clone()
    assert_bit1_matches_bool(c, src)
    # Mutate clone; original must remain intact.
    c.fill_(False)
    assert_bit1_matches_bool(a, src)


def test_new_zeros_preserves_bit1(device):
    a = bit1(_rand_bool(8, device))
    z = a.new_zeros((4, 4))
    assert getattr(z, '_is_bit1', False)
    assert_bit1_matches_bool(z, torch.zeros(4, 4, dtype=torch.bool, device=device))


def test_new_ones_preserves_bit1(device):
    a = bit1(_rand_bool(8, device))
    o = a.new_ones((3, 5))
    assert getattr(o, '_is_bit1', False)
    assert_bit1_matches_bool(o, torch.ones(3, 5, dtype=torch.bool, device=device))


# ─────────────────────────────────────────────────────────────────────────────
# Section 14 — Brute-specific methods
# ─────────────────────────────────────────────────────────────────────────────

def test_popcount(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    assert int(a.popcount()) == int(src.long().sum())


def test_hamming(device):
    sa = _rand_bool((32,), device)
    sb = _rand_bool((32,), device)
    a, b = bit1(sa), bit1(sb)
    assert int(a.hamming(b)) == int((sa ^ sb).long().sum())


def test_word_popcount_sums_to_total(device):
    a = bit1(_rand_bool((4, 8), device))
    wp = a.word_popcount()
    assert int(wp.as_subclass(torch.Tensor).long().sum()) == int(a.popcount())


def test_randomize_preserves_invariants(device):
    a = brute.zeros(1003, dtype=brute.bit1, device=device)  # not pw-aligned
    a.randomize_()
    pc_packed = int(a.popcount())
    pc_bool   = int(a.bool().as_subclass(torch.Tensor).long().sum())
    assert pc_packed == pc_bool, "packed_popcount and bool.sum must agree"


def test_unpack_pm1_values(device):
    a = bit1(torch.tensor([True, False, True], device=device))
    pm = a.unpack_pm1()
    assert torch.allclose(pm.cpu(),
                          torch.tensor([1.0, -1.0, 1.0]).cpu())


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 — Edge cases (0-dim, empty, non-contiguous, multi-pack-dtype)
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_dim_bool(device):
    src = torch.tensor(True, device=device)
    a = bit1(src)
    assert a.dim() == 0
    assert bool(a.item()) is True


def test_zero_dim_clone(device):
    src = torch.tensor(True, device=device)
    a = bit1(src)
    c = a.clone()
    assert bool(c.item()) is True


def test_empty_tensor(device):
    a = brute.zeros(0, dtype=brute.bit1, device=device)
    assert a.numel() == 0
    assert int(a.popcount()) == 0


def test_non_contig_transpose_sum(device):
    src = _rand_bool((4, 8), device)
    a = bit1(src)
    # After transpose, packed buf needs to be rebuilt on next op.
    t_a = a.t()
    expected = int(src.t().long().sum())
    assert int(t_a.sum()) == expected


def test_multi_pack_dtype_interop(device):
    src = _rand_bool((16,), device)
    a8  = bit1(src, pack_dtype=torch.uint8)
    a32 = bit1(src, pack_dtype=torch.uint32)
    a64 = bit1(src, pack_dtype=torch.uint64)
    # Every flavour materialises to the same bool view.
    for x in (a8, a32, a64):
        assert_bit1_matches_bool(x, src)


def test_chain_through_every_path(device):
    """End-to-end chain exercising bitwise, view, indexing, reduction, eq."""
    src_a = _rand_bool((4, 8), device)
    src_b = _rand_bool((4, 8), device)
    a, b = bit1(src_a), bit1(src_b)

    bit_chain  = ((a & b) | ~a).view(2, 16)[0]
    bool_chain = ((src_a & src_b) | ~src_a).view(2, 16)[0]
    assert_bit1_matches_bool(bit_chain, bool_chain)

    # Reduction on top.
    assert int(bit_chain.sum()) == int(bool_chain.long().sum())
    # Equality on the chained result.
    assert torch.equal(bit_chain, bit1(bool_chain))
