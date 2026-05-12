from __future__ import annotations

from typing import Optional
import torch

from brute.dtype import (
    bit1,
    _Bit1DType,
    resolve_pack_dtype,
    get_optimal_pack_dtype,
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
        pack_dtype: torch.dtype = None,
        device=None,
    ):
        # Unwrap brute.Tensor inputs to their underlying torch.Tensor.
        raw = data.as_subclass(torch.Tensor) if isinstance(data, Tensor) else data

        if isinstance(dtype, _Bit1DType):
            bool_t = _to_bool(raw) if isinstance(raw, torch.Tensor) else \
                     torch.as_tensor(raw, dtype=torch.bool)
            if pack_dtype is None:
                pack_dtype = get_optimal_pack_dtype(device)
            pack_dtype = resolve_pack_dtype(pack_dtype)
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

    def __init__(self, data, *, dtype=None, pack_dtype=None, device=None):
        pass

    @classmethod
    def _make_bit1(cls, bool_t: torch.Tensor, pack_dtype: torch.dtype = None) -> 'Tensor':
        """Internal: wrap an already-prepared bool tensor as a bit1 brute.Tensor."""
        if pack_dtype is None:
            pack_dtype = get_optimal_pack_dtype(bool_t.device)
        pack_dtype = resolve_pack_dtype(pack_dtype)
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

    @property
    def _packed_buf(self) -> Optional[torch.Tensor]:
        """Packed bit buffer.

        Lazily recomputed whenever the underlying bool storage has been modified
        in-place (detected via PyTorch's per-storage version counter).  This makes
        in-place ops such as fill_(), copy_(), index_put_(), and __setitem__
        automatically correct without needing per-op overrides.
        """
        if not getattr(self, '_is_bit1', False):
            return self.__dict__.get('_packed_buf_cache')
        cache = self.__dict__.get('_packed_buf_cache')
        try:
            cur_ver = self._version
        except Exception:
            cur_ver = None
        if cache is None or self.__dict__.get('_packed_ver') != cur_ver:
            cache = _pack_bool(self.as_subclass(torch.Tensor), self._pack_dtype)
            self.__dict__['_packed_buf_cache'] = cache
            self.__dict__['_packed_ver'] = cur_ver
        return cache

    @_packed_buf.setter
    def _packed_buf(self, val: Optional[torch.Tensor]) -> None:
        self.__dict__['_packed_buf_cache'] = val
        if val is not None:
            try:
                self.__dict__['_packed_ver'] = self._version
            except Exception:
                self.__dict__['_packed_ver'] = None
        else:
            self.__dict__.pop('_packed_ver', None)

    # ── element_size ──────────────────────────────────────────────────────────────

    def element_size(self) -> int:
        if getattr(self, '_is_bit1', False):
            raise TypeError(
                "element_size() is not defined for bit1 tensors: a logical element "
                "occupies less than one byte. Use tensor._packed_buf.nbytes for "
                "the total storage footprint of the packed buffer."
            )
        return super().element_size()

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the packed storage (bit1) or the raw tensor data."""
        if getattr(self, '_is_bit1', False):
            pb = self._packed_buf
            return pb.nbytes if pb is not None else 0
        return super().nbytes

    @property
    def itemsize(self) -> int:
        if getattr(self, '_is_bit1', False):
            raise TypeError(
                "itemsize is not defined for bit1 tensors: a logical element "
                "occupies less than one byte."
            )
        return super().itemsize

    def type(self, dtype=None, non_blocking: bool = False, **kwargs):
        """Return the type string, or cast to *dtype* (mirrors torch.Tensor.type)."""
        if dtype is None:
            if getattr(self, '_is_bit1', False):
                return 'brute.Bit1Tensor'
            return super().type()
        result = self.as_subclass(torch.Tensor).type(dtype, non_blocking=non_blocking, **kwargs)
        return Tensor._make_plain(result)

    # ── Factory methods (preserve bit1 dtype) ───────────────────────────────────

    def new_tensor(self, data, *, dtype=None, device=None, **kwargs) -> 'Tensor':
        """Create a new tensor from *data* with the same dtype/device as self by default."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_tensor(data, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_tensor(data, dtype=dtype, device=device, **kwargs))

    def new_empty(self, size, *, dtype=None, device=None, **kwargs) -> 'Tensor':
        """Return an uninitialised tensor of *size* with the same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_empty(size, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_empty(size, dtype=dtype, device=device, **kwargs))

    def new_full(self, size, fill_value, *, dtype=None, device=None, **kwargs) -> 'Tensor':
        """Return a tensor of *size* filled with *fill_value*, same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_full(size, bool(fill_value), dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_full(size, fill_value, dtype=dtype, device=device, **kwargs))

    def new_ones(self, size, *, dtype=None, device=None, **kwargs) -> 'Tensor':
        """Return an all-ones tensor of *size* with the same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_ones(size, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_ones(size, dtype=dtype, device=device, **kwargs))

    def new_zeros(self, size, *, dtype=None, device=None, **kwargs) -> 'Tensor':
        """Return an all-zeros tensor of *size* with the same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_zeros(size, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_zeros(size, dtype=dtype, device=device, **kwargs))

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
            if pack_dtype_arg:
                raw_pt = resolve_pack_dtype(pack_dtype_arg)
            elif getattr(self, '_pack_dtype', None):
                raw_pt = self._pack_dtype
            else:
                raw_pt = get_optimal_pack_dtype(base.device)
            return Tensor._make_bit1(base, raw_pt)

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
                
                # Use existing bit1 pack_dtype, or detect optimal for new bit1 tensors
                if bit1_ins:
                    pack_dtype = bit1_ins[0]._pack_dtype
                else:
                    # Infer device from inputs if available
                    device_for_pack = None
                    if flat:
                        for x in flat:
                            if isinstance(x, torch.Tensor):
                                device_for_pack = x.device
                                break
                    pack_dtype = get_optimal_pack_dtype(device_for_pack)

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
        # _packed_buf is a lazy property backed by _packed_buf_cache / _packed_ver.
        # After deepcopy the underlying bool storage is a new tensor at version 0,
        # so the version check will recompute on first access — no explicit copy needed.
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
