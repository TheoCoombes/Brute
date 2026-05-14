from __future__ import annotations

from math import prod as _prod
from typing import Optional
import torch

from brute.dtype import (
    bit1,
    _Bit1DType,
    resolve_pack_dtype,
    get_optimal_pack_dtype,
    _PACK_BITS,
)


_MATMUL_FUNCS = frozenset([
    torch.matmul, torch.mm, torch.bmm,
    torch.Tensor.matmul, torch.Tensor.mm, torch.Tensor.bmm,
    torch.Tensor.__matmul__,
])

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

# Funcs that take a bool MASK / CONDITION as one of their inputs. The bool
# input here is a selector, not a value, so it must NOT block promotion of
# a bit1 value-input back to bit1.
_MASK_USING_FUNCS = frozenset([
    torch.where,
    torch.masked_select, torch.Tensor.masked_select,
    torch.masked_fill,   torch.Tensor.masked_fill,
    torch.Tensor.__getitem__,
])

# Funcs that PyTorch does not implement on bool — we must transparently
# cast the bit1 value to uint8 before delegating.
_BOOL_UNSUPPORTED_FUNCS = frozenset([
    torch.argmax, torch.Tensor.argmax,
    torch.argmin, torch.Tensor.argmin,
])

# Movement/shape ops that change only metadata (no bit-pattern change). We can
# rebuild the packed buffer with a single `reshape`/`transpose` view instead of
# unpacking → bool op → re-packing.
_RESHAPE_FUNCS  = frozenset([torch.Tensor.view, torch.Tensor.reshape, torch.reshape])
_FLATTEN_FUNCS  = frozenset([torch.Tensor.flatten, torch.flatten])
_TRANSPOSE_FUNCS = frozenset([torch.Tensor.transpose, torch.transpose])
_CONTIG_FUNCS   = frozenset([torch.Tensor.contiguous])
_CLONE_FUNCS    = frozenset([torch.Tensor.clone, torch.clone])
_FILL_FUNCS     = frozenset([torch.Tensor.fill_])
_ZERO_FUNCS     = frozenset([torch.Tensor.zero_])

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

# Dynamo's _automatic_dynamic walks the view chain via _is_view() / _base.
# brute.Tensor is implemented with `as_subclass`, which makes every instance a
# C++-level view: _is_view() is True, _base is the plain backing tensor.
#
# If the generic __torch_function__ fallback handles _base, _rewrap wraps the
# plain tensor back into a brute.Tensor — whose _base is again the same plain
# tensor — and Dynamo walks the chain forever. So intercept _base here and
# return the C++ result directly (a plain torch.Tensor, whose _is_view() is
# False), which lets Dynamo's traversal terminate after one step.
_BASE_FUNCS = frozenset(
    f for f in [_metadata_get('_base')] if f is not None
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
        if getattr(self, '_is_bit1', False) and self.dim() >= 1:
            new_packed = torch.ops.brute.bit1_not_packed(
                self._packed_buf, int(self.shape[-1]),
                _PACK_BITS[self._pack_dtype],
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
        # Integer-tensor / long-list gather along axis 0.
        if isinstance(idx, torch.Tensor) and idx.dim() == 1 \
                and idx.dtype in (torch.int64, torch.int32, torch.long):
            return True
        return False

    def _last_axis_int_index(self, idx) -> Optional[int]:
        """Return the integer index targeting the last (packed) axis, else None.

        Recognises:
          - `x[..., j]`  → (Ellipsis, int)
          - `x[i_0, ..., i_{n-1}, j]` where every i_k is `:` and the int is on
            the last axis (e.g. `x[:, j]` on a 2-D tensor).
        Only recognises the packed-axis-int pattern; everything else returns None.
        """
        if not (getattr(self, '_is_bit1', False) and self.dim() >= 1):
            return None
        last_axis = self.dim() - 1
        if isinstance(idx, tuple):
            # (Ellipsis, int)
            if len(idx) == 2 and idx[0] is Ellipsis and isinstance(idx[1], int):
                return idx[1]
            # (slice(None), ..., slice(None), int) of length == dim
            if len(idx) == self.dim() and isinstance(idx[-1], int):
                lead_ok = all(
                    isinstance(s, slice) and s == slice(None, None, None)
                    for s in idx[:-1]
                )
                if lead_ok:
                    return idx[-1]
        return None

    def _column_gather(self, j: int) -> 'Tensor':
        """Extract the column `j` (along the packed axis) of a bit1 tensor.

        Reads bit `j` from each row's packed word and re-packs the result.
        Returns a bit1 tensor of shape `self.shape[:-1]`.
        """
        pw = _PACK_BITS[self._pack_dtype]
        K  = int(self.shape[-1])
        if j < 0:
            j += K
        if not (0 <= j < K):
            raise IndexError(
                f"index {j} is out of bounds for axis {self.dim() - 1} with size {K}"
            )
        word_idx   = j // pw
        bit_offset = j % pw
        # `select` along packed axis → shape self.shape[:-1] (uint-typed words)
        word_col = self._packed_buf.select(-1, word_idx)
        # Extract bit `bit_offset` → uint with 0 or 1 per cell.
        bool_col = ((word_col >> bit_offset) & 1).bool().contiguous()
        return Tensor._make_bit1(bool_col, self._pack_dtype)

    def _column_scatter(self, j: int, value: bool) -> None:
        """In-place set the column `j` of a bit1 tensor to a scalar bool.

        Two regimes:
        * Bool is **dirty** (lazy result from a packed-op chain): touch ONLY
          the packed buffer. Bool stays dirty; the eventual `_ensure_bool_valid`
          will materialise the post-write state in one pass. This avoids a
          full unpack just to overwrite one column.
        * Bool is **valid**: update bool storage (for view-alias correctness)
          AND the single packed-word column — both updates are O(M) and the
          full re-pack from the bool fallback is avoided.
        """
        pw = _PACK_BITS[self._pack_dtype]
        K  = int(self.shape[-1])
        if j < 0:
            j += K
        if not (0 <= j < K):
            raise IndexError(
                f"index {j} is out of bounds for axis {self.dim() - 1} with size {K}"
            )
        pb = self.__dict__.get('_packed_buf_cache')
        if pb is None:
            # No packed cache yet — fall back to the bool path.
            return super().__setitem__((slice(None),) * (self.dim() - 1) + (j,),
                                       bool(value))
        # Build the single-bit mask in packed dtype.
        one  = torch.ones((), dtype=pb.dtype, device=pb.device)
        mask = one << (j % pw)
        word_col = pb.select(-1, j // pw)
        if value:
            word_col.bitwise_or_(mask)
        else:
            word_col.bitwise_and_(mask.bitwise_not())
        self.__dict__['_packed_buf_cache'] = pb

        if self.__dict__.get('_bool_dirty', False):
            # Bool storage is uninit garbage — defer materialisation.
            # `_ensure_bool_valid` will pick up the updated packed buffer next
            # time something reads the bool view.
            try:
                self.__dict__['_packed_ver'] = self._version
            except Exception:
                self.__dict__['_packed_ver'] = None
            return
        # Bool is valid — keep aliasing views in sync by also writing the
        # column. `copy_` from a broadcast scalar dodges MPS's strided
        # `fill_` bug.
        plain = self.as_subclass(torch.Tensor)
        col_view = plain.select(-1, j)
        col_view.copy_(
            torch.tensor(bool(value), dtype=torch.bool,
                         device=plain.device).expand_as(col_view)
        )
        try:
            self.__dict__['_packed_ver'] = self._version
        except Exception:
            self.__dict__['_packed_ver'] = None

    def __getitem__(self, idx):
        if self._leading_axis_fast(idx):
            packed = self._packed_buf
            if isinstance(idx, torch.Tensor):
                # Row gather: packed.index_select(0, idx) preserves last dim.
                sub = packed.index_select(0, idx)
                new_shape = [int(idx.numel())] + list(self.shape[1:])
            else:
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
        # Last-axis column reads (`x[:, j]`, `x[..., j]`) currently fall
        # through to the bool path — the bool fallback is one strided view +
        # one `pack_bool` kernel, which is faster than a hand-rolled
        # shift-mask-cast-pack sequence (extra ATen launches dominate at
        # the typical workloads). Re-enable a custom path here if/when a
        # fused packed-gather kernel exists.
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
        # Column scatter: `x[:, j] = scalar` / `x[..., j] = scalar`.
        last_int = self._last_axis_int_index(idx)
        if last_int is not None and isinstance(value, (bool, int)):
            self._column_scatter(last_int, bool(value))
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
        # Mask off pad bits in EVERY row's tail word — multi-row packed
        # tensors have shape (..., ceil(K / pw)) and each row carries its
        # own (pw - K % pw) pad bits.
        pw = _PACK_BITS[self._pack_dtype]
        K  = int(self.shape[-1]) if self.dim() >= 1 else 0
        valid_bits = K % pw
        if valid_bits != 0 and pb.numel() > 0:
            mask = (1 << valid_bits) - 1
            last_col = pb.select(-1, pb.size(-1) - 1)
            last_col.bitwise_and_(
                torch.tensor(mask, dtype=pb.dtype, device=pb.device))
        self.__dict__['_packed_buf_cache'] = pb
        self.__dict__['_bool_dirty'] = True
        try:
            self.__dict__['_packed_ver'] = self._version
        except Exception:
            self.__dict__['_packed_ver'] = None
        return self

    # ── Direct method overrides (bypass __torch_function__ dispatch overhead) ──

    def _reshape_packed(self, new_shape: list) -> Optional['Tensor']:
        """Helper: try the packed-buffer reshape. Returns None on fall-through."""
        total = self.numel()
        if -1 in new_shape and new_shape.count(-1) == 1:
            known = 1
            for d in new_shape:
                if d != -1:
                    known *= d
            if known > 0 and total % known == 0:
                new_shape = [(total // known) if d == -1 else d for d in new_shape]
        if not new_shape or any(d < 0 for d in new_shape):
            return None
        prod_new = 1
        for d in new_shape:
            prod_new *= d
        if prod_new != total:
            return None
        pw = _PACK_BITS[self._pack_dtype]
        old_last = int(self.shape[-1]) if self.dim() >= 1 else 0
        new_last = new_shape[-1]
        packed = self._packed_buf
        if old_last == new_last and self.dim() >= 1:
            new_packed = packed.reshape(new_shape[:-1] + [packed.shape[-1]])
        elif old_last % pw == 0 and new_last % pw == 0:
            new_packed = packed.reshape(new_shape[:-1] + [new_last // pw])
        else:
            return None
        return Tensor._make_bit1_from_packed(new_packed, new_shape, self._pack_dtype)

    def view(self, *size):
        if getattr(self, '_is_bit1', False) and self.dim() >= 1:
            if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
                new_shape = list(size[0])
            else:
                new_shape = list(size)
            r = self._reshape_packed(new_shape)
            if r is not None:
                return r
        return super().view(*size)

    def reshape(self, *size):
        if getattr(self, '_is_bit1', False) and self.dim() >= 1:
            if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
                new_shape = list(size[0])
            else:
                new_shape = list(size)
            r = self._reshape_packed(new_shape)
            if r is not None:
                return r
        return super().reshape(*size)

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        if getattr(self, '_is_bit1', False) and self.dim() >= 1:
            nd = self.dim()
            sd = start_dim if start_dim >= 0 else nd + start_dim
            ed = end_dim   if end_dim   >= 0 else nd + end_dim
            # Full flatten: equivalent to reshape((-1,)) when start==0, end==nd-1.
            if sd == 0 and ed == nd - 1:
                pw = _PACK_BITS[self._pack_dtype]
                if self.shape[-1] % pw == 0 and self.shape[-1] > 0:
                    new_packed = self._packed_buf.reshape(-1)
                    return Tensor._make_bit1_from_packed(
                        new_packed, [self.numel()], self._pack_dtype,
                    )
        return super().flatten(start_dim, end_dim)

    def contiguous(self, memory_format=torch.contiguous_format):
        # `contiguous()` is a no-op when self's bool view is already contiguous;
        # the packed buffer is always contiguous (alloc'd with `at::empty`).
        if getattr(self, '_is_bit1', False):
            if self.as_subclass(torch.Tensor).is_contiguous(memory_format=memory_format):
                return self
        return super().contiguous(memory_format=memory_format)

    def transpose(self, dim0: int, dim1: int):
        if getattr(self, '_is_bit1', False) and self.dim() >= 2:
            nd = self.dim()
            d0 = dim0 if dim0 >= 0 else nd + dim0
            d1 = dim1 if dim1 >= 0 else nd + dim1
            if d0 == d1:
                return self
            last_axis = nd - 1
            if d0 != last_axis and d1 != last_axis:
                # Both leading — rearrange packed buffer too.
                new_packed = self._packed_buf.transpose(d0, d1).contiguous()
                new_shape  = list(self.shape)
                new_shape[d0], new_shape[d1] = new_shape[d1], new_shape[d0]
                return Tensor._make_bit1_from_packed(
                    new_packed, new_shape, self._pack_dtype,
                )
            # Involves the packed axis — bool view + lazy packed rebuild.
            if self.__dict__.get('_bool_dirty', False):
                self._ensure_bool_valid()
            new_bool = self.as_subclass(torch.Tensor).transpose(d0, d1)
            inst = new_bool.as_subclass(type(self))
            inst._is_bit1    = True
            inst._pack_dtype = self._pack_dtype
            inst._packed_buf = None
            if nd == 2:
                inst.__dict__['_t_source'] = self
            return inst
        return super().transpose(dim0, dim1)

    def unsqueeze(self, dim: int):
        if getattr(self, '_is_bit1', False) and self.dim() >= 1:
            nd = self.dim()
            d  = dim if dim >= 0 else nd + 1 + dim
            # Inserting at any position that doesn't shift the packed axis past
            # the new last position keeps the packed buffer trivially reusable.
            if d <= nd - 1:
                new_packed = self._packed_buf.unsqueeze(d)
                new_shape  = list(self.shape)
                new_shape.insert(d, 1)
                return Tensor._make_bit1_from_packed(
                    new_packed, new_shape, self._pack_dtype,
                )
        return super().unsqueeze(dim)

    def squeeze(self, *args, **kwargs):
        if getattr(self, '_is_bit1', False) and self.dim() >= 1:
            # Resolve `dim` argument (optional).
            dim = None
            if args:
                dim = args[0]
            elif 'dim' in kwargs:
                dim = kwargs['dim']
            if dim is None:
                # Squeeze every size-1 axis. Don't touch the packed axis (last).
                squeezable = [i for i, s in enumerate(self.shape[:-1]) if s == 1]
                if squeezable:
                    new_shape = [s for i, s in enumerate(self.shape) if not (i in squeezable)]
                    new_packed = self._packed_buf
                    # Squeeze each axis in reverse so indices stay valid.
                    for i in sorted(squeezable, reverse=True):
                        new_packed = new_packed.squeeze(i)
                    if not new_shape:
                        return super().squeeze(*args, **kwargs)
                    return Tensor._make_bit1_from_packed(
                        new_packed, new_shape, self._pack_dtype,
                    )
                if self.shape[-1] == 1:
                     # The packed axis itself is size-1 — we deliberately keep
                     # the packed axis intact to stay within the parity envelope.
                     # Return self unchanged as a bit1 tensor.
                     return self
                # Nothing to squeeze.
                return self
            nd = self.dim()
            d  = dim if dim >= 0 else nd + dim
            if 0 <= d < nd - 1 and self.shape[d] == 1:
                new_packed = self._packed_buf.squeeze(d)
                new_shape  = [s for i, s in enumerate(self.shape) if i != d]
                return Tensor._make_bit1_from_packed(
                    new_packed, new_shape, self._pack_dtype,
                )
        return super().squeeze(*args, **kwargs)

    def clone(self, *, memory_format=torch.preserve_format):
        if getattr(self, '_is_bit1', False) and self.dim() >= 1:
            return Tensor._make_bit1_from_packed(
                self._packed_buf.clone(), list(self.shape), self._pack_dtype,
            )
        return super().clone(memory_format=memory_format)

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
        # Exception: for mask-using ops (where, masked_select, masked_fill,
        # __getitem__), bool inputs may be MASKS rather than values — promote
        # purely on whether any bit1 value-input is present.
        if func in _MASK_USING_FUNCS:
            promote_to_bit1 = any_bit1
        else:
            promote_to_bit1 = any_bit1 and not bool(non_bit1_bool_ins)

        # argmax/argmin on bool aren't implemented in PyTorch — cast our bit1
        # value to uint8 transparently and dispatch to the underlying op.
        if func in _BOOL_UNSUPPORTED_FUNCS and bit1_ins:
            a = bit1_ins[0]
            if a.__dict__.get('_bool_dirty', False):
                a._ensure_bool_valid()
            with torch._C.DisableTorchFunctionSubclass():
                u8 = a.as_subclass(torch.Tensor).to(torch.uint8)
            new_args = list(args)
            # The bit1 value is positional arg 0 for argmax/argmin.
            new_args[0] = u8
            return Tensor._make_plain(func(*new_args, **kwargs))

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

            # any(dim=-1) / all(dim=-1) — reduce over the packed last dim
            # without unpacking the full bool storage. The packed buffer has
            # shape (..., ceil(K/pw)). Pad bits are zero, so:
            #   any(row) = ((packed_row != 0)).any(dim=-1)
            #   all(row) = popcount(packed_row).sum(dim=-1) == K
            # Both avoid materialising a (..., K) bool temporary.
            if (func in _ANY_FUNCS or func in _ALL_FUNCS) \
                    and len(bit1_ins) == 1 and not no_dim:
                a = bit1_ins[0]
                dim_arg = args[1] if len(args) > 1 else kwargs.get('dim')
                keepdim = (args[2] if len(args) > 2
                           else kwargs.get('keepdim', False))
                if (a.dim() >= 1 and isinstance(dim_arg, int)
                        and isinstance(keepdim, bool)):
                    ndim = a.dim()
                    dim_p = dim_arg if dim_arg >= 0 else ndim + dim_arg
                    if dim_p == ndim - 1:
                        packed = a._packed_buf
                        if func in _ANY_FUNCS:
                            result = (packed != 0).any(dim=-1,
                                                       keepdim=keepdim)
                        else:
                            K = int(a.shape[-1])
                            pops = torch.ops.brute.popcount(packed)
                            row_total = pops.to(torch.int64).sum(
                                dim=-1, keepdim=keepdim)
                            result = (row_total == K)
                        # Match the generic-fallback wrap: non-scalar bool
                        # reductions over a bit1 input get promoted back to
                        # bit1; scalar results stay plain (no shape).
                        if result.dim() == 0:
                            return Tensor._make_plain(result)
                        return Tensor._make_bit1(result, a._pack_dtype)

            # torch.equal(a, b) -> Python bool. Pad bits are zero in both
            # operands, so packed equality ⇔ logical equality. `torch.equal`
            # on uint packed buffers short-circuits on the first mismatching
            # byte — much faster than the full XOR+popcount scan for unequal
            # tensors, and identical for equal ones.
            if func in _EQUAL_FUNCS and len(bit1_ins) >= 2:
                a, b = bit1_ins[0], bit1_ins[1]
                if (a.shape == b.shape
                        and a._pack_dtype == b._pack_dtype
                        and a._packed_buf.shape == b._packed_buf.shape):
                    return torch.equal(a._packed_buf, b._packed_buf)

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
                        a._packed_buf, int(a.shape[-1]),
                        _PACK_BITS[a._pack_dtype],
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
                        xored, int(a.shape[-1]), _PACK_BITS[a._pack_dtype],
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

            # ── Movement / shape ops ────────────────────────────────────────
            # These preserve bit ordering in row-major storage, so we can just
            # reshape the packed buffer instead of unpacking → bool op → pack.

            # contiguous(): no-op only when the tensor is already contiguous.
            # After a transpose / strided slice the bool view is non-contiguous,
            # and we must actually materialise a contiguous copy — fall through.
            if func in _CONTIG_FUNCS and len(bit1_ins) == 1:
                a = bit1_ins[0]
                if a.as_subclass(torch.Tensor).is_contiguous():
                    return a

            # view / reshape — rearrange packed if last-dim alignment matches.
            if func in _RESHAPE_FUNCS and len(bit1_ins) == 1:
                a = bit1_ins[0]
                # Extract new shape from args/kwargs.
                if 'shape' in kwargs:
                    raw = kwargs['shape']
                    new_shape = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                elif len(args) == 2 and isinstance(args[1], (list, tuple, torch.Size)):
                    new_shape = list(args[1])
                else:
                    new_shape = list(args[1:])
                # Resolve -1.
                total = a.numel()
                if -1 in new_shape and new_shape.count(-1) == 1:
                    known = _prod([d for d in new_shape if d != -1]) or 1
                    if total % known == 0:
                        new_shape = [(total // known) if d == -1 else d for d in new_shape]
                if new_shape and all(d >= 0 for d in new_shape) \
                        and _prod(new_shape) == total:
                    pw = _PACK_BITS[a._pack_dtype]
                    old_last = int(a.shape[-1]) if a.dim() >= 1 else 0
                    new_last = new_shape[-1]
                    packed = a._packed_buf
                    new_packed = None
                    if old_last == new_last and a.dim() >= 1:
                        # Just rearrange leading dims of packed.
                        new_packed_shape = new_shape[:-1] + [packed.shape[-1]]
                        new_packed = packed.reshape(new_packed_shape)
                    elif old_last % pw == 0 and new_last % pw == 0:
                        new_packed_shape = new_shape[:-1] + [new_last // pw]
                        new_packed = packed.reshape(new_packed_shape)
                    if new_packed is not None:
                        return Tensor._make_bit1_from_packed(
                            new_packed, new_shape, a._pack_dtype,
                        )

            # flatten() — equivalent to reshape((-1,)) for the common case.
            if func in _FLATTEN_FUNCS and len(bit1_ins) == 1:
                a = bit1_ins[0]
                start = args[1] if len(args) > 1 else kwargs.get('start_dim', 0)
                end   = args[2] if len(args) > 2 else kwargs.get('end_dim', -1)
                if start == 0 and end in (-1, a.dim() - 1) and a.dim() >= 1:
                    pw = _PACK_BITS[a._pack_dtype]
                    last = int(a.shape[-1])
                    if last % pw == 0 and last > 0:
                        new_packed = a._packed_buf.reshape(-1)
                        return Tensor._make_bit1_from_packed(
                            new_packed, [a.numel()], a._pack_dtype,
                        )

            # transpose(d0, d1) — only fast when neither dim is the packed axis.
            # Otherwise return a lazy bit1 with packed=None (rebuilt on demand).
            if func in _TRANSPOSE_FUNCS and len(bit1_ins) == 1:
                a = bit1_ins[0]
                if a.dim() >= 2:
                    d0 = args[1] if len(args) > 1 else kwargs.get('dim0', 0)
                    d1 = args[2] if len(args) > 2 else kwargs.get('dim1', 1)
                    ndim = a.dim()
                    last_axis = ndim - 1
                    d0p = d0 if d0 >= 0 else ndim + d0
                    d1p = d1 if d1 >= 0 else ndim + d1
                    if d0p == d1p:
                        return a
                    if d0p != last_axis and d1p != last_axis:
                        # Both leading — rearrange packed too.
                        new_packed = a._packed_buf.transpose(d0p, d1p).contiguous()
                        new_shape = list(a.shape)
                        new_shape[d0p], new_shape[d1p] = new_shape[d1p], new_shape[d0p]
                        return Tensor._make_bit1_from_packed(
                            new_packed, new_shape, a._pack_dtype,
                        )
                    # Involves the packed axis — bool view + lazy packed rebuild.
                    if a.__dict__.get('_bool_dirty', False):
                        a._ensure_bool_valid()
                    new_bool = a.as_subclass(torch.Tensor).transpose(d0p, d1p)
                    inst = new_bool.as_subclass(type(a))
                    inst._is_bit1    = True
                    inst._pack_dtype = a._pack_dtype
                    inst._packed_buf = None
                    if ndim == 2:
                        inst.__dict__['_t_source'] = a
                    return inst

            # clone() — copy packed buffer, allocate empty bool, mark dirty.
            if func in _CLONE_FUNCS and len(bit1_ins) == 1:
                a = bit1_ins[0]
                if a.dim() >= 1:
                    return Tensor._make_bit1_from_packed(
                        a._packed_buf.clone(), list(a.shape), a._pack_dtype,
                    )

            # torch.index_select(x, dim, idx) on axis 0 — gather rows of packed.
            if func in (torch.index_select, torch.Tensor.index_select) \
                    and len(bit1_ins) == 1:
                a = bit1_ins[0]
                # Args: (input, dim, index)
                dim = args[1] if len(args) > 1 else kwargs.get('dim')
                idx = args[2] if len(args) > 2 else kwargs.get('index')
                if (a.dim() >= 2 and dim in (0,) and isinstance(idx, torch.Tensor)
                        and idx.dim() == 1):
                    new_packed = a._packed_buf.index_select(0, idx)
                    new_shape = [int(idx.numel())] + list(a.shape[1:])
                    return Tensor._make_bit1_from_packed(
                        new_packed.contiguous(), new_shape, a._pack_dtype,
                    )

            # fill_(value) / zero_() — write bool storage AND packed pattern,
            # skipping the read-back pack from bool. Bool must stay in sync
            # because callers may hold view aliases that observe the bool storage.
            if (func in _FILL_FUNCS or func in _ZERO_FUNCS) and len(bit1_ins) == 1:
                a = bit1_ins[0]
                if func in _ZERO_FUNCS:
                    value = False
                else:
                    value = args[1] if len(args) > 1 else kwargs.get('value')
                if isinstance(value, (bool, int)) and a.dim() >= 1:
                    # Mutate bool storage via copy_ from a broadcast scalar.
                    # `.fill_()` on MPS has a known bug for non-zero-offset
                    # bool views (writes to the wrong indices); copy_ from a
                    # 0-dim expanded tensor is the safe equivalent and adds
                    # no extra allocation (broadcasting stride is 0).
                    plain = a.as_subclass(torch.Tensor)
                    plain.copy_(
                        torch.tensor(bool(value), dtype=torch.bool,
                                     device=a.device).expand_as(plain)
                    )
                    # Write the packed pattern directly — skips the pack_bool
                    # kernel that would otherwise read the entire bool buffer.
                    pw = _PACK_BITS[a._pack_dtype]
                    K = int(a.shape[-1])
                    pb = a.__dict__.get('_packed_buf_cache')
                    if pb is None:
                        packed_shape = list(a.shape[:-1]) + [(K + pw - 1) // pw]
                        pb = torch.empty(packed_shape, dtype=a._pack_dtype,
                                         device=a.device)
                    if bool(value):
                        # Single byte-wise fill — works for any unsigned pack
                        # dtype without going through (zero_ + bitwise_not_),
                        # which used two passes over packed memory. view as
                        # uint8 because uint32/uint64 fill_ doesn't accept
                        # negative literals in pytorch.
                        pb.view(torch.uint8).fill_(0xFF)
                        if K > 0 and K % pw != 0:
                            valid_bits = K % pw
                            mask = (1 << valid_bits) - 1
                            pb[..., -1] = mask
                    else:
                        pb.zero_()
                    a.__dict__['_packed_buf_cache'] = pb
                    a.__dict__['_bool_dirty'] = False
                    a.__dict__['_packed_ver'] = a._version
                    return a

        # Dynamo's _automatic_dynamic walks `e._base` recursively. The generic
        # fallback would rewrap our plain-tensor _base back into brute.Tensor —
        # whose _base is the same plain tensor again — and that loops forever.
        # Bypass the fallback for _base: return the raw C++ result (a plain
        # torch.Tensor whose own _is_view() is False), terminating the walk.
        if func in _BASE_FUNCS:
            with torch._C.DisableTorchFunctionSubclass():
                return args[0]._base

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

        # CUDA has no addmm/bmm kernel for integer dtypes — promote int tensor
        # operands to float for the matmul and demote the result back to the
        # promoted integer dtype. CPU/MPS handle int matmul natively, so guard
        # on device.
        if func in _MATMUL_FUNCS and not bit1_ins:
            cuda_int_tensors = [
                x for x in flat
                if isinstance(x, torch.Tensor)
                and x.device.type == 'cuda'
                and not x.is_floating_point()
                and not x.is_complex()
                and x.dtype != torch.bool
            ]
            if cuda_int_tensors:
                target_int = cuda_int_tensors[0].dtype
                for x in cuda_int_tensors[1:]:
                    target_int = torch.promote_types(target_int, x.dtype)
                # int64 needs float64 to preserve precision; everything else
                # fits in float32.
                compute_dtype = (torch.float64 if target_int == torch.int64
                                 else torch.float32)
                def _to_compute(x):
                    if (isinstance(x, torch.Tensor)
                            and x.device.type == 'cuda'
                            and not x.is_floating_point()
                            and not x.is_complex()
                            and x.dtype != torch.bool):
                        return x.to(compute_dtype)
                    return x
                u_args   = torch.utils._pytree.tree_map(_to_compute, u_args)
                u_kwargs = torch.utils._pytree.tree_map(_to_compute, u_kwargs)
                result   = func(*u_args, **u_kwargs)
                if isinstance(result, torch.Tensor):
                    result = result.to(target_int)
                else:
                    result = type(result)(
                        r.to(target_int) if isinstance(r, torch.Tensor) else r
                        for r in result
                    )
            else:
                result = func(*u_args, **u_kwargs)
        else:
            result = func(*u_args, **u_kwargs)

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

    # NOTE: We deliberately do NOT implement __tensor_flatten__ /
    # __tensor_unflatten__. Those methods opt into PyTorch's "traceable wrapper
    # subclass" protocol, which assumes the OUTER tensor is metadata and the
    # INNER tensors hold the storage. brute.Tensor uses `as_subclass` instead —
    # outer and inner share the same TensorImpl — so the wrapper protocol is
    # the wrong shape for us: FakeTensorMode tries to re-subclass an existing
    # FakeTensor and crashes ("raw Tensor object is already associated to a
    # python object of type FakeTensor"). Without these methods, Dynamo treats
    # us as a plain tensor subclass and the compile path works end-to-end.

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
