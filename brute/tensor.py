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

# Reductions where the fast-path on bit1 is a direct call to packed_popcount.
_SUM_FUNCS = frozenset([
    torch.sum,        torch.Tensor.sum,
    torch.count_nonzero, torch.Tensor.count_nonzero,
])
_ALL_FUNCS   = frozenset([torch.all, torch.Tensor.all])
_ANY_FUNCS   = frozenset([torch.any, torch.Tensor.any])
_EQUAL_FUNCS = frozenset([torch.equal])

# Functional/method forms of binary bitwise & logical ops — `__and__`/etc. dunders
# are dispatched directly to our `Tensor.__and__` override, but the functional
# forms (`torch.bitwise_and`, `torch.logical_and`, `tensor.bitwise_and(...)`)
# come through `__torch_function__` and need their own packed fast path.
_BITWISE_BIN_FAST = {
    torch.bitwise_and:        'bitwise_and',
    torch.bitwise_or:         'bitwise_or',
    torch.bitwise_xor:        'bitwise_xor',
    torch.logical_and:        'bitwise_and',
    torch.logical_or:         'bitwise_or',
    torch.logical_xor:        'bitwise_xor',
    torch.Tensor.bitwise_and: 'bitwise_and',
    torch.Tensor.bitwise_or:  'bitwise_or',
    torch.Tensor.bitwise_xor: 'bitwise_xor',
    torch.Tensor.logical_and: 'bitwise_and',
    torch.Tensor.logical_or:  'bitwise_or',
    torch.Tensor.logical_xor: 'bitwise_xor',
}
_BITWISE_NOT_FAST = frozenset([
    torch.bitwise_not, torch.logical_not,
    torch.Tensor.bitwise_not, torch.Tensor.logical_not,
])

# Functions that only read tensor METADATA (shape/dim/dtype/device/etc.) and
# never the bool storage. We can skip `_ensure_bool_valid` for these — saving
# a full unpack on every metadata access for lazy-bool tensors.
#
# Property getters appear as the descriptor's `__get__` method-wrapper, so we
# unwrap them with `getattr(prop, '__get__', prop)`.
def _metadata_get(name: str):
    prop = getattr(torch.Tensor, name, None)
    return getattr(prop, '__get__', prop) if prop is not None else None

_BOOL_FREE_FUNCS = frozenset(
    f for f in [
        _metadata_get('shape'),    _metadata_get('ndim'),
        _metadata_get('dtype'),    _metadata_get('device'),
        _metadata_get('layout'),   _metadata_get('requires_grad'),
        _metadata_get('_version'),
        _metadata_get('is_cuda'),  _metadata_get('is_mps'),
        _metadata_get('is_cpu'),   _metadata_get('is_leaf'),
        torch.Tensor.dim, torch.Tensor.numel, torch.Tensor.size,
        torch.Tensor.stride, torch.Tensor.storage_offset,
        torch.Tensor.is_contiguous,
        torch.Tensor.__len__,
        torch.numel,
    ] if f is not None
)


# Internal helpers 

def _to_bool(t: torch.Tensor) -> torch.Tensor:
    """Cast any tensor to bool (>0 for floats, standard .bool() otherwise)."""
    if t.dtype == torch.bool:
        return t
    if t.is_floating_point():
        return t > 0
    return t.bool()

def _pack_bool(bool_t: torch.Tensor, pack_dtype: torch.dtype) -> torch.Tensor:
    """Pack a bool tensor to uint storage via `brute.pack_bool`.

    CPU + CUDA have native fast-path kernels; other backends (MPS, future)
    transparently fall through to the CompositeExplicitAutograd composite
    registered in ext.cpp, which decomposes to `(bool*2-1).float() → pack_bits`.
    """
    if bool_t.dtype != torch.bool:
        bool_t = bool_t.bool()
    if bool_t.dim() == 0:
        bool_t = bool_t.unsqueeze(0)
    return torch.ops.brute.pack_bool(bool_t.contiguous(), _PACK_BITS[pack_dtype])


def _unpack_pm1(packed: torch.Tensor, logical_shape: list, pack_dtype: torch.dtype) -> torch.Tensor:
    """Unpack packed buffer → float32 (+1.0 = True, −1.0 = False)."""
    if len(logical_shape) == 0:
        out = torch.ops.brute.unpack_bits(packed, [1], _PACK_BITS[pack_dtype])
        return out.squeeze(0)
    return torch.ops.brute.unpack_bits(packed, logical_shape, _PACK_BITS[pack_dtype])

def _rebuild_brute_tensor(plain_tensor: torch.Tensor, is_bit1: bool = False, pack_dtype: torch.dtype = None):
    """Reconstructs a brute.Tensor from a pickled base tensor."""
    t = plain_tensor.as_subclass(Tensor)
    t._is_bit1 = is_bit1
    t._pack_dtype = pack_dtype
    # The packed buffer will automatically lazy-load on next access
    t._packed_buf = None 
    return t

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([_rebuild_brute_tensor])

# Tensor 

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

    # Construction 

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
            
            instance = bool_t.as_subclass(cls)
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
            
        instance = t.as_subclass(cls)
        instance._is_bit1    = False
        instance._pack_dtype = None
        instance._packed_buf = None
        return instance
    
    def __init__(self, data, *, dtype=None, pack_dtype=None, device=None):
        pass

    @classmethod
    def _make_bit1(cls, bool_t: torch.Tensor, pack_dtype: torch.dtype = None) -> Tensor:
        # Unwrap brute.Tensor inputs to their underlying torch.Tensor.
        if pack_dtype is None:
            pack_dtype = get_optimal_pack_dtype(bool_t.device)
        pack_dtype = resolve_pack_dtype(pack_dtype)

        instance = bool_t.as_subclass(cls)
        instance._is_bit1    = True
        instance._pack_dtype = pack_dtype
        instance._packed_buf = _pack_bool(bool_t, pack_dtype)
        return instance

    @classmethod
    def _make_plain(cls, t: torch.Tensor) -> Tensor:
        """Internal: wrap a non-bit1 torch.Tensor as a brute.Tensor."""
        instance = t.as_subclass(cls)
        instance._is_bit1    = False
        instance._pack_dtype = None
        instance._packed_buf = None
        return instance

    @classmethod
    def _make_bit1_from_packed(cls, packed: torch.Tensor, logical_shape: list,
                               pack_dtype: torch.dtype) -> 'Tensor':
        """Create a bit1 tensor from a pre-computed packed buffer (lazy bool).

        The bool backing storage is allocated uninitialised — it is *not* a
        valid view of the bit1 contents. The packed buffer is the source of
        truth until `_ensure_bool_valid()` is called (lazily, from
        `__torch_function__` and any explicit bool-view accessor).
        """
        bool_t = torch.empty(logical_shape, dtype=torch.bool, device=packed.device)
        instance = bool_t.as_subclass(cls)
        instance._is_bit1    = True
        instance._pack_dtype = pack_dtype
        instance.__dict__['_packed_buf_cache'] = packed
        try:
            instance.__dict__['_packed_ver'] = instance._version
        except Exception:
            instance.__dict__['_packed_ver'] = None
        instance.__dict__['_bool_dirty'] = True
        return instance

    def _ensure_bool_valid(self) -> None:
        """Materialise the bool backing storage from the packed buffer.

        No-op if `_bool_dirty` is False (the common case after construction
        via `_make_bit1`). Called automatically from `__torch_function__` and
        any path that exposes the bool view to PyTorch ops or user code.

        Runs inside `DisableTorchFunctionSubclass`: attribute accesses on
        `self` (e.g. `.shape`, `._version`) would otherwise re-enter
        `__torch_function__` and recurse back here.
        """
        if not self.__dict__.get('_bool_dirty', False):
            return
        packed = self.__dict__.get('_packed_buf_cache')
        if packed is None:
            self.__dict__['_bool_dirty'] = False
            return
        with torch._C.DisableTorchFunctionSubclass():
            shape = list(self.shape)
            unpacked = torch.ops.brute.unpack_bool(
                packed, shape, _PACK_BITS[self._pack_dtype])
            # `copy_` bumps the storage version; immediately re-pin `_packed_ver`
            # to the new version so the packed cache stays valid in the getter.
            self.as_subclass(torch.Tensor).copy_(unpacked)
            ver = self._version
        self.__dict__['_bool_dirty'] = False
        self.__dict__['_packed_ver'] = ver

    # Properties 

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

        When `_bool_dirty` is set (after a packed-output op produced this tensor
        without unpacking) the cache is the *only* valid source — the version
        counter would re-pack from invalid bool storage, so we skip it.
        """
        if not getattr(self, '_is_bit1', False):
            return self.__dict__.get('_packed_buf_cache')
        if self.__dict__.get('_bool_dirty', False):
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

    # element_size 

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
        if getattr(self, '_is_bit1', False):
            self._ensure_bool_valid()
        result = self.as_subclass(torch.Tensor).type(dtype, non_blocking=non_blocking, **kwargs)
        return Tensor._make_plain(result)

    # Factory methods (preserve bit1 dtype) 

    def new_tensor(self, data, *, dtype=None, device=None, **kwargs) -> Tensor:
        """Create a new tensor from *data* with the same dtype/device as self by default."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_tensor(data, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_tensor(data, dtype=dtype, device=device, **kwargs))

    def new_empty(self, size, *, dtype=None, device=None, **kwargs) -> Tensor:
        """Return an uninitialised tensor of *size* with the same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_empty(size, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_empty(size, dtype=dtype, device=device, **kwargs))

    def new_full(self, size, fill_value, *, dtype=None, device=None, **kwargs) -> Tensor:
        """Return a tensor of *size* filled with *fill_value*, same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_full(size, bool(fill_value), dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_full(size, fill_value, dtype=dtype, device=device, **kwargs))

    def new_ones(self, size, *, dtype=None, device=None, **kwargs) -> Tensor:
        """Return an all-ones tensor of *size* with the same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_ones(size, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_ones(size, dtype=dtype, device=device, **kwargs))

    def new_zeros(self, size, *, dtype=None, device=None, **kwargs) -> Tensor:
        """Return an all-zeros tensor of *size* with the same dtype/device as self."""
        eff_dtype = dtype if dtype is not None else self.dtype
        base = self.as_subclass(torch.Tensor)
        if isinstance(eff_dtype, _Bit1DType):
            bool_t = base.new_zeros(size, dtype=torch.bool, device=device)
            pd = getattr(self, '_pack_dtype', None) or get_optimal_pack_dtype(
                torch.device(device) if device else base.device)
            return Tensor._make_bit1(bool_t, pd)
        return Tensor._make_plain(base.new_zeros(size, dtype=dtype, device=device, **kwargs))

    # Conversion 

    def bool(self) -> Tensor:
        """
        Return a bool-typed brute.Tensor (dtype=torch.bool).

        For bit1: wraps the underlying bool storage as a plain brute.Tensor.
        For other dtypes: casts elements to bool and wraps.
        """
        if getattr(self, '_is_bit1', False):
            self._ensure_bool_valid()
            return Tensor._make_plain(self.as_subclass(torch.Tensor))
        return Tensor._make_plain(self.as_subclass(torch.Tensor).bool())

    def to(self, *args, **kwargs) -> Tensor:
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
            if getattr(self, '_is_bit1', False):
                self._ensure_bool_valid()
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
            # Detect whether the caller explicitly asked for a non-bit1 dtype
            # (e.g. .to(torch.bool)) — in that case we must NOT re-promote to bit1.
            explicit_dtype = kwargs.get('dtype')
            if explicit_dtype is None:
                for a in args:
                    if isinstance(a, torch.dtype):
                        explicit_dtype = a
                        break
            self._ensure_bool_valid()
            new_base = self.as_subclass(torch.Tensor).to(*args, **kwargs)
            if explicit_dtype is None and new_base.dtype == torch.bool:
                # Device-only / format-only conversion — preserve bit1.
                return Tensor._make_bit1(new_base, self._pack_dtype)
            return Tensor._make_plain(new_base)

        return Tensor._make_plain(self.as_subclass(torch.Tensor).to(*args, **kwargs))

    def unpack_pm1(self) -> torch.Tensor:
        """Decode packed storage → float32 (+1.0 = True, −1.0 = False)."""
        if not getattr(self, '_is_bit1', False):
            raise TypeError("unpack_pm1() is only valid for bit1 tensors")
        return _unpack_pm1(self._packed_buf, list(self.shape), self._pack_dtype)

    # Binary ops on packed buffers 
    # For same-shape bit1 tensors we operate directly on the packed integer
    # buffers — XOR/AND/OR of two identically-packed buffers gives the correct
    # packed result because the pad bits are 0 in both operands:
    #   0 XOR 0 = 0,  0 AND 0 = 0,  0 OR 0 = 0  ✓
    # Falls back to bool-level ops for broadcasting / mixed types.

    def __and__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.shape == other.shape
                and self._pack_dtype == other._pack_dtype
                and self.dim() >= 1):
            return Tensor._make_bit1_from_packed(
                torch.ops.brute.bitwise_and(self._packed_buf, other._packed_buf),
                list(self.shape), self._pack_dtype,
            )
        return super().__and__(other)

    def __or__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.shape == other.shape
                and self._pack_dtype == other._pack_dtype
                and self.dim() >= 1):
            return Tensor._make_bit1_from_packed(
                torch.ops.brute.bitwise_or(self._packed_buf, other._packed_buf),
                list(self.shape), self._pack_dtype,
            )
        return super().__or__(other)

    def __xor__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.shape == other.shape
                and self._pack_dtype == other._pack_dtype
                and self.dim() >= 1):
            return Tensor._make_bit1_from_packed(
                torch.ops.brute.bitwise_xor(self._packed_buf, other._packed_buf),
                list(self.shape), self._pack_dtype,
            )
        return super().__xor__(other)

    def __invert__(self):
        # Operate directly on the packed buffer: ~A then mask the pad bits in
        # the tail word so popcount/sum stay correct. The bool view stays
        # lazy — never materialised unless a non-bitwise op demands it.
        if getattr(self, '_is_bit1', False):
            new_packed = torch.ops.brute.bit1_not_packed(
                self._packed_buf, self.numel(), _PACK_BITS[self._pack_dtype],
            )
            return Tensor._make_bit1_from_packed(
                new_packed, list(self.shape), self._pack_dtype,
            )
        return super().__invert__()

    def _iop_packed(self, other, op):
        """Shared body for __iand__/__ior__/__ixor__.

        Updates only the packed cache and marks bool dirty — the bool view is
        materialised lazily on next access. This avoids the unpack+copy that
        previously dominated in-place bitwise cost.
        """
        new_packed = op(self._packed_buf, other._packed_buf)
        self.__dict__['_packed_buf_cache'] = new_packed
        self.__dict__['_bool_dirty'] = True
        try:
            self.__dict__['_packed_ver'] = self._version
        except Exception:
            self.__dict__['_packed_ver'] = None
        return self

    def __iand__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.shape == other.shape
                and self._pack_dtype == other._pack_dtype
                and self.dim() >= 1):
            return self._iop_packed(other, torch.ops.brute.bitwise_and)
        return super().__iand__(other)

    def __ior__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.shape == other.shape
                and self._pack_dtype == other._pack_dtype
                and self.dim() >= 1):
            return self._iop_packed(other, torch.ops.brute.bitwise_or)
        return super().__ior__(other)

    def __ixor__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.shape == other.shape
                and self._pack_dtype == other._pack_dtype
                and self.dim() >= 1):
            return self._iop_packed(other, torch.ops.brute.bitwise_xor)
        return super().__ixor__(other)

    # Indexing fast paths (leading-axis only) 
    # `x[i]`, `x[a:b]`, `x[i] = scalar`, `x[a:b] = scalar` on a bit1 tensor of
    # dim >= 2 can be served by slicing the packed buffer directly — rows are
    # contiguous in packed storage. The last axis (packed axis) requires bit
    # gather/scatter and falls through to the bool path.

    def _leading_axis_fast(self, idx) -> bool:
        """True iff `idx` only touches axes other than the last (packed) axis."""
        if not getattr(self, '_is_bit1', False) or self.dim() < 2:
            return False
        if isinstance(idx, int):
            return True
        if isinstance(idx, slice):
            # Reject negative step; need contiguous packed slice.
            return (idx.step is None or idx.step == 1)
        return False

    def __getitem__(self, idx):
        if self._leading_axis_fast(idx):
            packed = self._packed_buf
            sub = packed[idx]
            # `x[int]` drops the leading axis; `x[slice]` keeps it.
            new_shape = list(self.shape[1:]) if isinstance(idx, int) else \
                        [sub.shape[0]] + list(self.shape[1:])
            if not isinstance(sub, torch.Tensor) or sub.numel() == 0:
                # Empty or unexpected — fall through to safe path.
                return super().__getitem__(idx)
            return Tensor._make_bit1_from_packed(
                sub.contiguous(), new_shape, self._pack_dtype,
            )
        return super().__getitem__(idx)

    def _row_pattern(self, value: bool) -> torch.Tensor:
        """One row of packed words representing `value` across the last dim.

        For `True`: all live bits set, pad bits zero.
        For `False`: all bits zero.
        """
        pw     = _PACK_BITS[self._pack_dtype]
        K      = int(self.shape[-1])
        n_words = (K + pw - 1) // pw
        valid_bits = K % pw
        # Always materialise via the packed cache so dtype/device match exactly.
        ref = self._packed_buf
        row = torch.zeros(n_words, dtype=ref.dtype, device=ref.device)
        if not value:
            return row
        # ~0 in the unsigned dtype: pre-zero then bitwise_not_ avoids the
        # uint64-overflow issue when passing 2**64-1 as a Python int.
        row.bitwise_not_()
        if valid_bits != 0:
            mask = (1 << valid_bits) - 1
            row[-1] = mask
        return row

    def __setitem__(self, idx, value):
        # Fast path: scalar bool value on a leading axis. Mutates the packed
        # buffer directly and marks bool dirty; the bool view is unpacked
        # lazily on next read.
        if (self._leading_axis_fast(idx)
                and isinstance(value, (bool, int))):
            pb = self._packed_buf
            row = self._row_pattern(bool(value))
            if isinstance(idx, int):
                pb[idx].copy_(row)
            else:
                # slice — broadcast the row into every selected row.
                pb[idx].copy_(row.unsqueeze(0).expand(pb[idx].shape[0], -1))
            self.__dict__['_packed_buf_cache'] = pb
            self.__dict__['_bool_dirty'] = True
            try:
                self.__dict__['_packed_ver'] = self._version
            except Exception:
                self.__dict__['_packed_ver'] = None
            return
        return super().__setitem__(idx, value)

    def popcount(self) -> Tensor:
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

    def hamming(self, other: 'Tensor') -> 'Tensor':
        """Total Hamming distance — number of bit positions where self and other differ.

        Uses the fused XOR+popcount kernel with no intermediate allocation.
        Both tensors must be bit1 with identical shape and pack_dtype.
        Returns a 0-dim int64 brute.Tensor scalar.
        """
        if not (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)):
            raise TypeError("hamming() is only defined for bit1 tensors")
        if self.shape != other.shape:
            raise ValueError(
                f"hamming() requires matching shapes; got {tuple(self.shape)} vs {tuple(other.shape)}"
            )
        if self._pack_dtype != other._pack_dtype:
            raise ValueError(
                f"hamming() requires matching pack_dtype; got {self._pack_dtype} vs {other._pack_dtype}"
            )
        return Tensor._make_plain(
            torch.ops.brute.bit1_hamming_total(self._packed_buf, other._packed_buf)
        )

    def word_popcount(self) -> 'Tensor':
        """Per-packed-word popcount of the bit1 storage buffer.

        Returns an int32 tensor with the same shape as the packed buffer, where
        each element is the popcount of the corresponding packed integer word.
        Useful for density analysis and debugging.
        """
        if not getattr(self, '_is_bit1', False):
            raise TypeError("word_popcount() is only defined for bit1 tensors")
        return Tensor._make_plain(torch.ops.brute.popcount(self._packed_buf))

    def randomize_(self) -> 'Tensor':
        """Fill this bit1 tensor with uniformly random bits, in place.

        Operates directly on the packed buffer; bool view is marked dirty and
        unpacked lazily on first access.
        """
        if not getattr(self, '_is_bit1', False):
            raise TypeError("randomize_() is only defined for bit1 tensors")
        pb = self._packed_buf.clone()
        torch.ops.brute.randomize_bits(pb)
        # Mask off pad bits so packed_popcount / sum etc. stay correct.
        pw = _PACK_BITS[self._pack_dtype]
        numel = self.numel()
        valid_bits = numel % pw
        if valid_bits != 0 and pb.numel() > 0:
            mask = (1 << valid_bits) - 1
            flat = pb.flatten()
            flat[-1] = flat[-1] & mask
        self.__dict__['_packed_buf_cache'] = pb
        self.__dict__['_bool_dirty'] = True
        try:
            self.__dict__['_packed_ver'] = self._version
        except Exception:
            self.__dict__['_packed_ver'] = None
        return self

    # Transpose (preserves _t_source for fast A @ B.t() matmul) 

    def t(self) -> 'Tensor':
        """2-D transpose preserving bit1 dtype and enabling the A @ B.t() fast path.

        The returned tensor keeps a _t_source back-reference to self so that
        __matmul__ can read self's packed buffer (N×K form) instead of repacking
        the transposed K×N tensor, which would break the xnor_popcount_matmul layout.
        """
        if getattr(self, '_is_bit1', False):
            # The transpose returns a *view* of self's bool storage. If we're
            # still dirty, that view would be invalid; force materialisation
            # before slicing. Note: matmul against t() goes through _t_source
            # and reads self._packed_buf directly, which is dirty-safe — but
            # any other code touching the transposed bool view needs valid data.
            self._ensure_bool_valid()
        base   = self.as_subclass(torch.Tensor)
        result = base.t()
        if not getattr(self, '_is_bit1', False):
            return Tensor._make_plain(result)
        t_inst = result.as_subclass(type(self))
        t_inst._is_bit1    = True
        t_inst._pack_dtype = self._pack_dtype
        t_inst._packed_buf = None
        if self.dim() == 2:
            t_inst.__dict__['_t_source'] = self
        return t_inst

    # Matmul 

    def __matmul__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.dim() == 2 and other.dim() == 2):
            K = self.shape[-1]

            # A (M×K) @ B (N×K) — bit1 convention: rows of B are output features.
            # Guard on logical K equality, not just packed-word count, to avoid
            # false positives when two tensors share the same ceil(K/pw) by accident.
            if (other.shape[-1] == K
                    and self._packed_buf.shape[-1] == other._packed_buf.shape[-1]):
                return torch.ops.brute.xnor_popcount_matmul(
                    self._packed_buf, other._packed_buf,
                    K, _PACK_BITS[self._pack_dtype],
                )

            # A (M×K) @ B.t() (K×N) — standard matmul shape.
            # B.t() carries _t_source = B (N×K), so we use B's packed buffer
            # directly instead of repacking the transposed K×N form.
            t_src = other.__dict__.get('_t_source')
            if (t_src is not None
                    and isinstance(t_src, Tensor)
                    and getattr(t_src, '_is_bit1', False)
                    and t_src.shape[-1] == K
                    and t_src._pack_dtype == self._pack_dtype
                    and self._packed_buf.shape[-1] == t_src._packed_buf.shape[-1]):
                return torch.ops.brute.xnor_popcount_matmul(
                    self._packed_buf, t_src._packed_buf,
                    K, _PACK_BITS[self._pack_dtype],
                )

        return super().__matmul__(other)

    # __torch_function__ 

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
            if a.dim() == 2 and b.dim() == 2:
                K = a.shape[-1]
                # A (M×K) @ B (N×K)
                if (b.shape[-1] == K
                        and a._packed_buf.shape[-1] == b._packed_buf.shape[-1]):
                    return torch.ops.brute.xnor_popcount_matmul(
                        a._packed_buf, b._packed_buf,
                        K, _PACK_BITS[a._pack_dtype],
                    )
                # A (M×K) @ B.t() (K×N) via _t_source
                b_src = b.__dict__.get('_t_source')
                if (b_src is not None
                        and isinstance(b_src, Tensor)
                        and getattr(b_src, '_is_bit1', False)
                        and b_src.shape[-1] == K
                        and b_src._pack_dtype == a._pack_dtype
                        and a._packed_buf.shape[-1] == b_src._packed_buf.shape[-1]):
                    return torch.ops.brute.xnor_popcount_matmul(
                        a._packed_buf, b_src._packed_buf,
                        K, _PACK_BITS[a._pack_dtype],
                    )

        # Fast path: full-tensor reductions on bit1 driven by packed_popcount.
        # Pad bits in the packed buffer are zero, so the total popcount equals
        # the count of True logical bits — no correction needed.
        if bit1_ins and not non_bit1_bool_ins:
            # Only the full-reduction form is short-circuited. If `dim` is
            # provided (positionally or as kw), fall through to the bool path.
            no_dim = (len(args) <= 1 and 'dim' not in kwargs)

            if func in _SUM_FUNCS and no_dim and len(bit1_ins) == 1:
                a = bit1_ins[0]
                return Tensor._make_plain(
                    torch.ops.brute.packed_popcount(a._packed_buf)
                )

            if func in _ANY_FUNCS and no_dim and len(bit1_ins) == 1:
                a = bit1_ins[0]
                total = torch.ops.brute.packed_popcount(a._packed_buf).item()
                return Tensor._make_plain(torch.tensor(bool(total), dtype=torch.bool))

            if func in _ALL_FUNCS and no_dim and len(bit1_ins) == 1:
                a = bit1_ins[0]
                total = torch.ops.brute.packed_popcount(a._packed_buf).item()
                return Tensor._make_plain(
                    torch.tensor(int(total) == a.numel(), dtype=torch.bool)
                )

            # torch.equal(a, b) -> Python bool. Fused XOR+popcount, no temp.
            if func in _EQUAL_FUNCS and len(bit1_ins) >= 2:
                a, b = bit1_ins[0], bit1_ins[1]
                if (a.shape == b.shape
                        and a._pack_dtype == b._pack_dtype
                        and a._packed_buf.shape == b._packed_buf.shape):
                    return torch.ops.brute.bit1_hamming_total(
                        a._packed_buf, b._packed_buf
                    ).item() == 0

            # torch.bitwise_*(a, b) / torch.logical_*(a, b) and the equivalent
            # Tensor methods. Identical shape + pack_dtype → operate on packed.
            if func in _BITWISE_BIN_FAST and len(bit1_ins) >= 2:
                a, b = bit1_ins[0], bit1_ins[1]
                if (a.shape == b.shape
                        and a._pack_dtype == b._pack_dtype
                        and a.dim() >= 1):
                    op_name = _BITWISE_BIN_FAST[func]
                    op_fn = getattr(torch.ops.brute, op_name)
                    return Tensor._make_bit1_from_packed(
                        op_fn(a._packed_buf, b._packed_buf),
                        list(a.shape), a._pack_dtype,
                    )

            # torch.bitwise_not(a) / torch.logical_not(a) — pad-safe packed NOT.
            # 0-dim scalars fall through to the bool path (the lazy-bool
            # unpack helper requires dim >= 1).
            if func in _BITWISE_NOT_FAST and len(bit1_ins) == 1 \
                    and bit1_ins[0].dim() >= 1:
                a = bit1_ins[0]
                return Tensor._make_bit1_from_packed(
                    torch.ops.brute.bit1_not_packed(
                        a._packed_buf, a.numel(), _PACK_BITS[a._pack_dtype]
                    ),
                    list(a.shape), a._pack_dtype,
                )

            # torch.eq(a, b) / a == b — bit1-bit1 same shape. Equivalent to
            # ~(a ^ b): a single packed XOR then a pad-masked NOT.
            if func in (torch.eq, torch.Tensor.eq, torch.Tensor.__eq__) \
                    and len(bit1_ins) >= 2:
                a, b = bit1_ins[0], bit1_ins[1]
                if (a.shape == b.shape
                        and a._pack_dtype == b._pack_dtype
                        and a.dim() >= 1):
                    xored = torch.ops.brute.bitwise_xor(a._packed_buf, b._packed_buf)
                    inverted = torch.ops.brute.bit1_not_packed(
                        xored, a.numel(), _PACK_BITS[a._pack_dtype],
                    )
                    return Tensor._make_bit1_from_packed(
                        inverted, list(a.shape), a._pack_dtype,
                    )

            # torch.ne(a, b) / a != b — bit1-bit1 same shape. Identical to XOR.
            if func in (torch.ne, torch.Tensor.ne, torch.Tensor.__ne__) \
                    and len(bit1_ins) >= 2:
                a, b = bit1_ins[0], bit1_ins[1]
                if (a.shape == b.shape
                        and a._pack_dtype == b._pack_dtype
                        and a.dim() >= 1):
                    return Tensor._make_bit1_from_packed(
                        torch.ops.brute.bitwise_xor(a._packed_buf, b._packed_buf),
                        list(a.shape), a._pack_dtype,
                    )

        # No fast path matched — we are about to expose the bit1 inputs as
        # torch.Tensor bool views to the generic fallback. Any input that was
        # produced via `_make_bit1_from_packed` (lazy bool) must materialise
        # its bool storage now, otherwise the underlying op would read garbage.
        #
        # Skip this for pure-metadata ops (`.shape`, `.dim()`, `.dtype`, ...)
        # which never touch bool data — paying a full unpack on every shape
        # access would be absurd.
        if func not in _BOOL_FREE_FUNCS:
            for inp in bit1_ins:
                if inp.__dict__.get('_bool_dirty', False):
                    inp._ensure_bool_valid()

        def _unwrap(x):
            return x.as_subclass(torch.Tensor) if isinstance(x, Tensor) else x

        u_args   = torch.utils._pytree.tree_map(_unwrap, args)
        u_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
        result   = func(*u_args, **u_kwargs)

        def _rewrap(r):
            if not isinstance(r, torch.Tensor):
                return r
            if promote_to_bit1 and r.dtype == torch.bool:
                # 0-dim bool results from reductions (all/any) should NOT be promoted
                # back to bit1 — they are logical scalars. Only promote when the
                # operation preserves the bit1 element-wise structure (e.g. clone).
                if r.dim() == 0 and func in (
                    torch.all, torch.any, torch.Tensor.all, torch.Tensor.any,
                ):
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

    # Serialisation 

    def __deepcopy__(self, memo):
        new_t = super().__deepcopy__(memo)
        new_t._is_bit1    = getattr(self, '_is_bit1', False)
        new_t._pack_dtype = getattr(self, '_pack_dtype', None)
        # _packed_buf is a lazy property backed by _packed_buf_cache / _packed_ver.
        # After deepcopy the underlying bool storage is a new tensor at version 0,
        # so the version check will recompute on first access — no explicit copy needed.
        return new_t

    def __reduce_ex__(self, _):
        """Tells pickle how to serialize and reconstruct this subclass."""
        # Demote to plain tensor to prevent infinite recursion during serialization.
        # Pickled state must be self-contained, so we materialise the bool view.
        if getattr(self, '_is_bit1', False):
            self._ensure_bool_valid()
        plain_t = self.as_subclass(torch.Tensor)
        return (
            _rebuild_brute_tensor,
            (plain_t, getattr(self, '_is_bit1', False), getattr(self, '_pack_dtype', None))
        )

    def __array__(self, dtype=None, copy=None):
        """NumPy 2.0+ interop hook."""
        if getattr(self, '_is_bit1', False):
            self._ensure_bool_valid()
        base_t = self.as_subclass(torch.Tensor)

        # For bit1, base_t is already torch.bool storage, so just convert to numpy natively
        arr = base_t.numpy(force=True)

        if dtype is not None:
            return arr.astype(dtype, copy=copy if copy is not None else False)
        if copy:
            return arr.copy()
        return arr

    def __tensor_flatten__(self):
        """Tells torch.compile how to extract standard tensors from this subclass."""
        # AOTAutograd needs string attribute names to extract tensors.
        # Since we use .as_subclass(), we temporarily attach the plain tensor data.
        if getattr(self, '_is_bit1', False):
            self._ensure_bool_valid()
        self._base_data = self.as_subclass(torch.Tensor)
        
        tensor_attrs = ["_base_data"]
        
        # Dynamo MUST know about any cached tensors hiding in __dict__
        if self.__dict__.get('_packed_buf_cache') is not None:
            tensor_attrs.append("_packed_buf_cache")
            
        metadata = {
            "is_bit1": getattr(self, '_is_bit1', False),
            "pack_dtype": getattr(self, '_pack_dtype', None),
            "packed_ver": self.__dict__.get('_packed_ver')
        }
        return tensor_attrs, metadata

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        """Tells torch.compile how to rebuild this subclass from the standard tensors."""
        base_data = inner_tensors["_base_data"]
        
        # Rebuild using your internal factories
        if metadata["is_bit1"]:
            # We don't want to re-trigger _pack_bool if we already have the cache, 
            # so we manually rebuild the instance state.
            instance = base_data.as_subclass(cls)
            instance._is_bit1 = True
            instance._pack_dtype = metadata["pack_dtype"]
        else:
            instance = cls._make_plain(base_data)
            
        # Restore the cache if it existed in the graph
        if "_packed_buf_cache" in inner_tensors:
            instance.__dict__['_packed_buf_cache'] = inner_tensors["_packed_buf_cache"]
            instance.__dict__['_packed_ver'] = metadata["packed_ver"]
            
        return instance

    # Repr 

    def __repr__(self):
        if getattr(self, '_is_bit1', False):
            self._ensure_bool_valid()
        base = self.as_subclass(torch.Tensor).__repr__()
        if getattr(self, '_is_bit1', False):
            if 'dtype=torch.bool' in base:
                base = base.replace('dtype=torch.bool', 'dtype=brute.bit1')
            else:
                base = base[:-1] + ', dtype=brute.bit1)'
        return base.replace('tensor(', 'brute.Tensor(')
