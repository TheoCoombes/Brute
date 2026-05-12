from __future__ import annotations

import copy
from typing import Optional

import torch


# ── Dtype sentinel ─────────────────────────────────────────────────────────────

class _Bit1DType:
    """Sentinel dtype for 1-bit packed tensors.

    Compares equal to torch.bool so that downstream code checking
    `tensor.dtype == torch.bool` still works for bit1 tensors.
    """
    def __repr__(self): return "brute.bit1"
    def __str__(self):  return "bit1"
    def __eq__(self, o):
        return isinstance(o, _Bit1DType) or o is torch.bool
    def __hash__(self):
        return hash(torch.bool)


bit1 = _Bit1DType()

# Dtype aliases
float32  = torch.float32
float16  = torch.float16
bfloat16 = torch.bfloat16
int8     = torch.int8
int32    = torch.int32
int64    = torch.int64

_PACK_WIDTH = {'uint8': 8, 'uint32': 32, 'uint64': 64}

# Functions that implement matrix multiplication
_MATMUL_FUNCS = frozenset([torch.matmul, torch.mm, torch.bmm])


# ── Internal packing helpers ────────────────────────────────────────────────────

def _to_bool(data: torch.Tensor) -> torch.Tensor:
    """Convert any tensor to bool using sign convention for floats."""
    if data.dtype == torch.bool:
        return data
    if data.is_floating_point():
        return data > 0  # sign convention: >0 → True (matches pack_bits kernel)
    return data.bool()


def _pack_bool(bool_t: torch.Tensor, pack_dtype: str) -> torch.Tensor:
    """Pack a bool tensor into a uint buffer using {−1, +1} encoding."""
    pw = _PACK_WIDTH[pack_dtype]
    # True→+1.0, False→−1.0; pack_bits treats ≥0 as bit=1
    return torch.ops.brute.pack_bits((bool_t.float() * 2 - 1).contiguous(), pw)


def _unpack_pm1(packed: torch.Tensor, logical_shape: list, pack_dtype: str) -> torch.Tensor:
    """Unpack packed buffer → float32 with values +1 / −1."""
    return torch.ops.brute.unpack_bits(packed, logical_shape, _PACK_WIDTH[pack_dtype])


# ── Tensor ─────────────────────────────────────────────────────────────────────

class Tensor(torch.Tensor):
    """torch.Tensor subclass with native bit1 support.

    For dtype=bit1
    ─────────────
    The base PyTorch storage is a BoolTensor (correct logical shape, full bool
    semantics). An additional ``_packed_buf`` caches the {−1,+1}-encoded uint
    representation used by ``xnor_popcount_matmul``.

    All standard boolean ops (``&``, ``|``, ``^``, ``~``, indexing, reductions,
    shape ops, ``torch.cat``, etc.) work natively through the bool base.  Only
    matmul is overridden to use the packed kernel.

    For other dtypes
    ────────────────
    Transparent torch.Tensor subclass.  Every op behaves identically to plain
    torch; the only effect is that the returned tensor is a ``brute.Tensor``.
    """

    # ── Construction ────────────────────────────────────────────────────────────

    @staticmethod
    def __new__(cls, data, *, dtype=None, pack_dtype: str = 'uint8', device=None):
        if isinstance(dtype, _Bit1DType):
            if isinstance(data, torch.Tensor):
                bool_t = _to_bool(data)
            else:
                bool_t = torch.as_tensor(data, dtype=torch.bool)
            if device is not None:
                bool_t = bool_t.to(device=device)
            instance = torch.Tensor._make_subclass(cls, bool_t)
            instance._is_bit1   = True
            instance._pack_str  = pack_dtype
            instance._packed_buf = _pack_bool(bool_t, pack_dtype)
            return instance
        else:
            if isinstance(data, torch.Tensor):
                t = data.to(dtype=dtype, device=device) if (dtype is not None or device is not None) else data
            elif isinstance(data, (list, tuple)):
                t = torch.tensor(data, dtype=dtype, device=device)
            else:
                t = torch.as_tensor(data, dtype=dtype, device=device)
            instance = torch.Tensor._make_subclass(cls, t)
            instance._is_bit1    = False
            instance._pack_str   = None
            instance._packed_buf = None
            return instance

    def __init__(self, data, *, dtype=None, pack_dtype: str = 'uint8', device=None):
        pass  # torch.Tensor.__init__ takes no arguments

    @classmethod
    def _make_bit1(cls, bool_t: torch.Tensor, pack_dtype: str = 'uint8') -> Tensor:
        """Internal: wrap an already-prepared bool tensor as bit1."""
        instance = torch.Tensor._make_subclass(cls, bool_t)
        instance._is_bit1    = True
        instance._pack_str   = pack_dtype
        instance._packed_buf = _pack_bool(bool_t, pack_dtype)
        return instance

    @classmethod
    def _make_plain(cls, t: torch.Tensor) -> Tensor:
        """Internal: wrap a non-bit1 torch.Tensor."""
        instance = torch.Tensor._make_subclass(cls, t)
        instance._is_bit1    = False
        instance._pack_str   = None
        instance._packed_buf = None
        return instance

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def dtype(self):
        if getattr(self, '_is_bit1', False):
            return bit1
        return super().dtype

    @property
    def pack_dtype(self) -> Optional[str]:
        return getattr(self, '_pack_str', None)

    # ── Conversion ──────────────────────────────────────────────────────────────

    def to(self, *args, **kwargs):
        # Detect dtype=brute.bit1 in positional or keyword args
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
            pack_str = kwargs.get('pack_dtype', getattr(self, '_pack_str', None) or 'uint8')
            return Tensor._make_bit1(base, pack_str)

        # For bit1 tensors: apply to() on the underlying bool base
        if getattr(self, '_is_bit1', False):
            new_base = self.as_subclass(torch.Tensor).to(*args, **kwargs)
            if new_base.dtype == torch.bool:
                return Tensor._make_bit1(new_base, self._pack_str)
            return Tensor._make_plain(new_base)

        return Tensor._make_plain(self.as_subclass(torch.Tensor).to(*args, **kwargs))

    def bool(self) -> torch.Tensor:
        """Return the underlying bool tensor (for bit1) or convert (for others)."""
        if getattr(self, '_is_bit1', False):
            return self.as_subclass(torch.Tensor)
        return self.as_subclass(torch.Tensor).bool()

    def unpack_pm1(self) -> torch.Tensor:
        """Decode packed storage → float32 with values +1 / −1 (BitNet convention)."""
        assert getattr(self, '_is_bit1', False), "unpack_pm1() is only valid for bit1 tensors"
        return _unpack_pm1(self._packed_buf, list(self.shape), self._pack_str)

    # ── Matmul ──────────────────────────────────────────────────────────────────

    def __matmul__(self, other):
        if (getattr(self, '_is_bit1', False)
                and isinstance(other, Tensor)
                and getattr(other, '_is_bit1', False)
                and self.dim() == 2 and other.dim() == 2
                and self._packed_buf.shape[-1] == other._packed_buf.shape[-1]):
            return torch.ops.brute.xnor_popcount_matmul(
                self._packed_buf, other._packed_buf,
                self.shape[-1], _PACK_WIDTH[self._pack_str],
            )
        return super().__matmul__(other)

    # ── __torch_function__ ──────────────────────────────────────────────────────

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Collect bit1 inputs
        flat = torch.utils._pytree.tree_leaves(list(args) + list(kwargs.values()))
        bit1_ins = [x for x in flat if isinstance(x, Tensor) and getattr(x, '_is_bit1', False)]
        any_bit1 = bool(bit1_ins)
        pack_str = bit1_ins[0]._pack_str if any_bit1 else 'uint8'

        # XNOR-popcount for 2D bit1 × bit1 matmul
        if func in _MATMUL_FUNCS and len(bit1_ins) >= 2:
            a, b = bit1_ins[0], bit1_ins[1]
            if (a.dim() == 2 and b.dim() == 2
                    and a._packed_buf.shape[-1] == b._packed_buf.shape[-1]):
                return torch.ops.brute.xnor_popcount_matmul(
                    a._packed_buf, b._packed_buf,
                    a.shape[-1], _PACK_WIDTH[a._pack_str],
                )

        # Unwrap all brute.Tensors to plain torch.Tensors
        def _unwrap(x):
            return x.as_subclass(torch.Tensor) if isinstance(x, Tensor) else x

        u_args   = torch.utils._pytree.tree_map(_unwrap, args)
        u_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
        result   = func(*u_args, **u_kwargs)

        # Re-wrap outputs
        def _rewrap(r):
            if not isinstance(r, torch.Tensor):
                return r
            if any_bit1 and r.dtype == torch.bool:
                return Tensor._make_bit1(r, pack_str)
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
        new_t._pack_str   = getattr(self, '_pack_str', None)
        new_t._packed_buf = copy.deepcopy(getattr(self, '_packed_buf', None), memo)
        return new_t

    # ── Repr ─────────────────────────────────────────────────────────────────────

    def __repr__(self):
        base = self.as_subclass(torch.Tensor).__repr__()
        if getattr(self, '_is_bit1', False):
            base = base.replace('dtype=torch.bool', 'dtype=brute.bit1')
        return base.replace('tensor(', 'brute.Tensor(')
