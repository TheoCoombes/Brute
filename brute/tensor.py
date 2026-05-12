from __future__ import annotations

from typing import Optional
import torch
import copy

from brute.dtypes import (
    float32, float64, float16, bfloat16,
    complex64, complex128,
    float8_e4m3fn, float8_e5m2,
    # Signed int
    int8, int16, int32, int64,
    # Unsigned int  (uint8 / uint32 / uint64 also serve as bit1 pack dtypes)
    uint8, uint16, uint32, uint64,
    _Bit1DType, _resolve_pack_dtype,
    _PACK_BITS,
)


_MATMUL_FUNCS = frozenset([torch.matmul, torch.mm, torch.bmm])


# ── Internal helpers ───────────────────────────────────────────────────────────

def _to_bool(t: torch.Tensor) -> torch.Tensor:
    """Cast any tensor to bool (>0 for floats, standard .bool() otherwise)."""
    if t.dtype == torch.bool:
        return t
    if t.is_floating_point():
        return t > 0
    return t.bool()


def _pack_bool(bool_t: torch.Tensor, pack_dtype: torch.dtype) -> torch.Tensor:
    """Pack a bool tensor to uint storage (True→1 bit, False→0 bit, {+1,-1} encoding)."""
    pm1 = (bool_t.float() * 2 - 1).contiguous()
    return torch.ops.brute.pack_bits(pm1, _PACK_BITS[pack_dtype])


def _unpack_pm1(packed: torch.Tensor, logical_shape: list, pack_dtype: torch.dtype) -> torch.Tensor:
    """Unpack packed buffer → float32 (+1.0 = True, −1.0 = False)."""
    return torch.ops.brute.unpack_bits(packed, logical_shape, _PACK_BITS[pack_dtype])


# ── Tensor ─────────────────────────────────────────────────────────────────────

class Tensor(torch.Tensor):
    """
    torch.Tensor subclass with native 1-bit (bit1) support.

    For dtype=brute.bit1:
      • Base PyTorch storage is bool (full bool semantics for indexing and standard ops).
      • `_packed_buf` caches the uint-packed form used by xnor_popcount_matmul.
      • `_pack_dtype` (torch.dtype — one of uint8/uint32/uint64) records the pack width.
      • `element_size()` raises — a bit1 element is sub-byte.

    For dtype=torch.bool (brute.bool):
      • Standard torch.bool tensor wrapped in brute.Tensor.
      • `popcount()` counts True values.

    For all other dtypes: transparent thin wrapper; every result stays in brute.
    """

    # ── Construction ────────────────────────────────────────────────────────────

    @staticmethod
    def __new__(
        cls,
        data,
        *,
        dtype=None,
        pack_dtype: torch.dtype = torch.uint8,
        device=None,
    ):
        pack_dtype = _resolve_pack_dtype(pack_dtype)

        # Unwrap brute.Tensor inputs to their underlying torch.Tensor.
        raw = data.as_subclass(torch.Tensor) if isinstance(data, Tensor) else data

        if isinstance(dtype, _Bit1DType):
            bool_t = _to_bool(raw) if isinstance(raw, torch.Tensor) else \
                     torch.as_tensor(raw, dtype=torch.bool)
            if device is not None:
                bool_t = bool_t.to(device=device)
            instance = torch.Tensor._make_subclass(cls, bool_t)
            instance._is_bit1    = True
            instance._pack_dtype = pack_dtype
            instance._packed_buf = _pack_bool(bool_t, pack_dtype)
            return instance

        if isinstance(raw, torch.Tensor):
            t = raw.to(dtype=dtype, device=device) if (dtype is not None or device is not None) else raw
        elif isinstance(raw, (list, tuple)):
            t = torch.tensor(raw, dtype=dtype, device=device)
        else:
            t = torch.as_tensor(raw, dtype=dtype, device=device)
        instance = torch.Tensor._make_subclass(cls, t)
        instance._is_bit1    = False
        instance._pack_dtype = None
        instance._packed_buf = None
        return instance

    def __init__(self, data, *, dtype=None, pack_dtype=torch.uint8, device=None):
        pass

    @classmethod
    def _make_bit1(cls, bool_t: torch.Tensor, pack_dtype: torch.dtype = torch.uint8) -> 'Tensor':
        """Internal: wrap an already-prepared bool tensor as a bit1 brute.Tensor."""
        pack_dtype = _resolve_pack_dtype(pack_dtype)
        instance = torch.Tensor._make_subclass(cls, bool_t)
        instance._is_bit1    = True
        instance._pack_dtype = pack_dtype
        instance._packed_buf = _pack_bool(bool_t, pack_dtype)
        return instance

    @classmethod
    def _make_plain(cls, t: torch.Tensor) -> 'Tensor':
        """Internal: wrap a non-bit1 torch.Tensor as a brute.Tensor."""
        instance = torch.Tensor._make_subclass(cls, t)
        instance._is_bit1    = False
        instance._pack_dtype = None
        instance._packed_buf = None
        return instance

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def dtype(self):
        if getattr(self, '_is_bit1', False):
            return bit1
        return super().dtype

    @property
    def pack_dtype(self) -> Optional[torch.dtype]:
        """Pack dtype (torch.uint8/uint32/uint64) for bit1 tensors, None otherwise."""
        return getattr(self, '_pack_dtype', None)

    # ── element_size ──────────────────────────────────────────────────────────────

    def element_size(self) -> int:
        if getattr(self, '_is_bit1', False):
            raise TypeError(
                "element_size() is not defined for bit1 tensors: a logical element "
                "occupies less than one byte. Use tensor._packed_buf.nbytes for "
                "the total storage footprint of the packed buffer."
            )
        return super().element_size()

    # ── Conversion ──────────────────────────────────────────────────────────────

    def bool(self) -> 'Tensor':
        """
        Return a bool-typed brute.Tensor (dtype=torch.bool).

        For bit1: wraps the underlying bool storage as a plain brute.Tensor.
        For other dtypes: casts elements to bool and wraps.
        """
        base = self.as_subclass(torch.Tensor)
        if getattr(self, '_is_bit1', False):
            return Tensor._make_plain(base)
        return Tensor._make_plain(base.bool())

    def to(self, *args, **kwargs) -> 'Tensor':
        # Extract brute-specific pack_dtype before forwarding to torch.
        kwargs = dict(kwargs)
        pack_dtype_arg = kwargs.pop('pack_dtype', None)

        dtype_arg = kwargs.get('dtype')
        if dtype_arg is None:
            for a in args:
                if isinstance(a, _Bit1DType):
                    dtype_arg = a
                    break

        if isinstance(dtype_arg, _Bit1DType):
            base = self.as_subclass(torch.Tensor)
            if not getattr(self, '_is_bit1', False):
                base = _to_bool(base)
            device_arg = kwargs.get('device')
            if device_arg:
                base = base.to(device=device_arg)
            raw_pt = pack_dtype_arg or getattr(self, '_pack_dtype', None) or torch.uint8
            return Tensor._make_bit1(base, _resolve_pack_dtype(raw_pt))

        if getattr(self, '_is_bit1', False):
            new_base = self.as_subclass(torch.Tensor).to(*args, **kwargs)
            if new_base.dtype == torch.bool:
                return Tensor._make_bit1(new_base, self._pack_dtype)
            return Tensor._make_plain(new_base)

        return Tensor._make_plain(self.as_subclass(torch.Tensor).to(*args, **kwargs))

    def unpack_pm1(self) -> torch.Tensor:
        """Decode packed storage → float32 (+1.0 = True, −1.0 = False)."""
        if not getattr(self, '_is_bit1', False):
            raise TypeError("unpack_pm1() is only valid for bit1 tensors")
        return _unpack_pm1(self._packed_buf, list(self.shape), self._pack_dtype)

    def popcount(self) -> 'Tensor':
        """
        Count the number of 1-bits (True values) in this tensor.

        For bit1: uses libpopcnt via packed_popcount (no bool materialisation).
        For bool (torch.bool): equivalent to tensor.long().sum().
        Returns a 0-dim int64 brute.Tensor scalar.
        """
        if getattr(self, '_is_bit1', False):
            return Tensor._make_plain(
                torch.ops.brute.packed_popcount(self._packed_buf)
            )
        base = self.as_subclass(torch.Tensor)
        if base.dtype == torch.bool:
            return Tensor._make_plain(base.long().sum())
        raise TypeError(
            f"popcount() is only defined for bit1 and bool tensors; got dtype={self.dtype}"
        )

    # ── Matmul ──────────────────────────────────────────────────────────────────

    def __matmul__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.dim() == 2 and other.dim() == 2
                and self._packed_buf.shape[-1] == other._packed_buf.shape[-1]):
            return torch.ops.brute.xnor_popcount_matmul(
                self._packed_buf, other._packed_buf,
                self.shape[-1], _PACK_BITS[self._pack_dtype],
            )
        return super().__matmul__(other)

    # ── __torch_function__ ──────────────────────────────────────────────────────

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        flat = torch.utils._pytree.tree_leaves(list(args) + list(kwargs.values()))

        bit1_ins = [x for x in flat if isinstance(x, Tensor) and getattr(x, '_is_bit1', False)]

        # Track ALL bool-typed tensors that are NOT bit1: plain torch.Tensor bools
        # and non-bit1 brute.Tensor bools. Mixed bit1+bool operations must not
        # silently coerce the bool side to bit1.
        non_bit1_bool_ins = [
            x for x in flat
            if isinstance(x, torch.Tensor)
            and not getattr(x, '_is_bit1', False)
            and (
                x.as_subclass(torch.Tensor).dtype
                if isinstance(x, Tensor)
                else x.dtype
            ) == torch.bool
        ]

        any_bit1 = bool(bit1_ins)
        # Promote bool results to bit1 only when every bool input is already bit1.
        promote_to_bit1 = any_bit1 and not bool(non_bit1_bool_ins)
        pack_dtype = bit1_ins[0]._pack_dtype if bit1_ins else torch.uint8

        # Fast path: XNOR-popcount for 2-D bit1 × bit1 matmul.
        if func in _MATMUL_FUNCS and len(bit1_ins) >= 2:
            a, b = bit1_ins[0], bit1_ins[1]
            if (a.dim() == 2 and b.dim() == 2
                    and a._packed_buf.shape[-1] == b._packed_buf.shape[-1]):
                return torch.ops.brute.xnor_popcount_matmul(
                    a._packed_buf, b._packed_buf,
                    a.shape[-1], _PACK_BITS[a._pack_dtype],
                )

        def _unwrap(x):
            return x.as_subclass(torch.Tensor) if isinstance(x, Tensor) else x

        u_args   = torch.utils._pytree.tree_map(_unwrap, args)
        u_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
        result   = func(*u_args, **u_kwargs)

        def _rewrap(r):
            if not isinstance(r, torch.Tensor):
                return r
            if promote_to_bit1 and r.dtype == torch.bool:
                if r.dim() == 0:
                    # 0-dim bool scalars (all/any/scalar-index) can't be packed.
                    return Tensor._make_plain(r)
                return Tensor._make_bit1(r, pack_dtype)
            if not isinstance(r, cls):
                return Tensor._make_plain(r)
            return r

        if isinstance(result, torch.Tensor):
            return _rewrap(result)
        if isinstance(result, (tuple, list)):
            return type(result)(_rewrap(r) for r in result)
        return result

    # ── Serialisation ────────────────────────────────────────────────────────────

    def __deepcopy__(self, memo):
        new_t = super().__deepcopy__(memo)
        new_t._is_bit1    = getattr(self, '_is_bit1', False)
        new_t._pack_dtype = getattr(self, '_pack_dtype', None)
        new_t._packed_buf = copy.deepcopy(getattr(self, '_packed_buf', None), memo)
        return new_t

    # ── Repr ─────────────────────────────────────────────────────────────────────

    def __repr__(self):
        base = self.as_subclass(torch.Tensor).__repr__()
        if getattr(self, '_is_bit1', False):
            if 'dtype=torch.bool' in base:
                base = base.replace('dtype=torch.bool', 'dtype=brute.bit1')
            else:
                base = base[:-1] + ', dtype=brute.bit1)'
        return base.replace('tensor(', 'brute.Tensor(')
