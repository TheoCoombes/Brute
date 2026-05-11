import torch

import brute._C as _C # pyright: ignore[reportMissingImports]
from brute._C import Bit1Tensor, PackDType # pyright: ignore[reportMissingImports]

# Sentinel dtype object for 1-bit tensors
class _Bit1DType(object):
    def __repr__(self):  return "brute.bit1"
    def __str__(self):   return "bit1"
    def __eq__(self, o): return isinstance(o, _Bit1DType)
    def __hash__(self):  return hash("bit1")

# Export dtypes
bit1 = _Bit1DType()
float32  = torch.float32
float16  = torch.float16
bfloat16 = torch.bfloat16
int8     = torch.int8
int32    = torch.int32
int64    = torch.int64

_PW   = {'uint8': 8,  'uint32': 32,  'uint64': 64}
_PDT  = {'uint8': PackDType.U8, 'uint32': PackDType.U32, 'uint64': PackDType.U64}
_PSTR = {PackDType.U8: 'uint8', PackDType.U32: 'uint32', PackDType.U64: 'uint64'}


class Tensor(object):
    """
    Unified tensor: either a packed 1-bit tensor or a standard torch.Tensor.

    For bit1 dtype, the packed buffer is stored in _inner (a Bit1Tensor).
    For all other dtypes, the raw torch.Tensor is in _data.

    Bit operations route to torch.ops.brute.*; everything else falls through
    to the underlying torch.Tensor via __torch_function__ or __getattr__.
    """

    def __init__(self, data, *, dtype=None, logical_shape=None, pack_dtype='uint8'):
        if isinstance(dtype, _Bit1DType):
            self._is_bit1  = True
            self._pack_str = pack_dtype
            pw = _PW[pack_dtype]
            if isinstance(data, torch.Tensor):
                ls = list(logical_shape) if logical_shape is not None else list(data.shape)
                packed = torch.ops.brute.pack_bits(data.float(), pw)
                self._inner = Bit1Tensor(packed, ls, _PDT[pack_dtype])
            elif isinstance(data, Bit1Tensor):
                self._inner    = data
                self._pack_str = _PSTR[data.pack_dtype()]
            else:
                raise TypeError(f"Cannot create bit1 Tensor from {type(data)}")
            self._data = None
        else:
            self._is_bit1 = False
            self._inner   = None
            self._pack_str = None
            if isinstance(data, torch.Tensor):
                self._data = data if dtype is None else data.to(dtype)
            elif isinstance(data, (list, tuple)):
                self._data = torch.tensor(data, dtype=dtype)
            else:
                self._data = torch.as_tensor(data, dtype=dtype)

    # ── Properties ──────────────────────────────────────────

    @property
    def dtype(self):
        return bit1 if self._is_bit1 else self._data.dtype

    @property
    def shape(self):
        if self._is_bit1:
            return torch.Size(self._inner.logical_shape())
        return self._data.shape

    @property
    def data(self):
        """Packed buffer (bit1) or raw tensor (others)."""
        return self._inner.data() if self._is_bit1 else self._data

    @property
    def pack_dtype(self):
        return self._pack_str

    # ── Conversion ──────────────────────────────────────────

    def unpack(self) -> torch.Tensor:
        """Decode bit1 storage → float32 tensor with values +1 / −1."""
        assert self._is_bit1, "unpack() is only valid for bit1 tensors"
        return torch.ops.brute.unpack_bits(
            self._inner.data(),
            list(self._inner.logical_shape()),
            _PW[self._pack_str],
        )

    def to(self, device=None, dtype=None):
        if self._is_bit1:
            new_packed = self._inner.data().to(device=device)
            new_inner  = Bit1Tensor(new_packed, list(self._inner.logical_shape()),
                                    self._inner.pack_dtype())
            t = object.__new__(Tensor)
            t._is_bit1  = True
            t._pack_str = self._pack_str
            t._inner    = new_inner
            t._data     = None
            return t
        return Tensor(self._data.to(device=device, dtype=dtype))

    def cpu(self):  return self.to(device='cpu')
    def mps(self):  return self.to(device='mps')
    def cuda(self): return self.to(device='cuda')

    # ── Ops ─────────────────────────────────────────────────

    def __matmul__(self, other):
        """XNOR-popcount matmul for two bit1 tensors; falls back to torch otherwise."""
        if self._is_bit1 and isinstance(other, Tensor) and other._is_bit1:
            K  = self.shape[-1]
            pw = _PW[self._pack_str]
            return torch.ops.brute.xnor_popcount_matmul(
                self._inner.data(), other._inner.data(), K, pw
            )
        a = self.unpack()  if self._is_bit1 else self._data
        b = other.unpack() if (isinstance(other, Tensor) and other._is_bit1) \
            else (other._data if isinstance(other, Tensor) else other)
        return torch.matmul(a, b)

    # ── torch protocol ──────────────────────────────────────

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Let torch.* functions work on brute.Tensor by unwrapping to torch.Tensor."""
        if kwargs is None:
            kwargs = {}
        def _unwrap(x):
            if isinstance(x, Tensor):
                return x.unpack() if x._is_bit1 else x._data
            return x
        new_args   = torch.utils._pytree.tree_map(_unwrap, args)
        new_kwargs = torch.utils._pytree.tree_map(_unwrap, kwargs)
        return func(*new_args, **new_kwargs)

    # ── Attribute forwarding ─────────────────────────────────

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        underlying = self._inner.data() if self._is_bit1 else self._data
        return getattr(underlying, name)

    # ── Repr ────────────────────────────────────────────────

    def __repr__(self):
        if self._is_bit1:
            d = self._inner.data().device
            return (f"brute.Tensor(shape={list(self.shape)}, dtype=bit1, "
                    f"pack_dtype={self._pack_str}, device={d})")
        return f"brute.Tensor({str(self._data).split('(', 1)[1][:-1]})"
