"""
brute.functional — factory and utility functions.

Only overrides a torch function when bit1 requires special handling.
For all other dtypes these are thin wrappers that keep results inside brute.
"""

from __future__ import annotations

import torch

from brute.tensor import Tensor, _Bit1DType, resolve_pack_dtype, bit1
from brute.dtype import _PACK_BITS


# Internal normaliser

def _norm_size(size) -> tuple:
    """Accept zeros(2,3), zeros((2,3)), or zeros([2,3])."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        return tuple(size[0])
    return size


def _packed_shape(size: tuple, pack_dtype: torch.dtype) -> tuple:
    """Compute the packed buffer shape for a given logical shape."""
    pw = _PACK_BITS[pack_dtype]
    if len(size) == 0:
        return (1,)
    last = int(size[-1])
    n_words = (last + pw - 1) // pw if last > 0 else 0
    return tuple(size[:-1]) + (n_words,)


def _make_bit1_eager(packed: torch.Tensor, size: tuple, pack_dtype: torch.dtype,
                     device) -> Tensor:
    """Wrap a freshly-allocated packed buffer as a bit1 tensor with empty
    (uninitialised) bool storage. The bool view is left dirty — it is unpacked
    lazily on first access.
    """
    bool_t = torch.empty(size, dtype=torch.bool, device=device)
    instance = bool_t.as_subclass(Tensor)
    instance._is_bit1    = True
    instance._pack_dtype = pack_dtype
    instance.__dict__['_packed_buf_cache'] = packed
    instance.__dict__['_packed_ver']       = instance._version
    instance.__dict__['_bool_dirty']       = True
    return instance


# Creators: bit1-aware

def zeros(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.zeros(size, dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.zeros(size, dtype=dtype, device=device, **kwargs))


def ones(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.ones(size, dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.ones(size, dtype=dtype, device=device, **kwargs))


def empty(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    size = _norm_size(size)
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.empty(size, dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.empty(size, dtype=dtype, device=device, **kwargs))


def full(size, fill_value, *, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    if isinstance(size, int):
        size = (size,)
    if isinstance(dtype, _Bit1DType):
        # Delegate to zeros/ones — both already skip the bool init.
        return ones(*size, dtype=dtype, device=device, pack_dtype=pack_dtype) \
            if bool(fill_value) else \
            zeros(*size, dtype=dtype, device=device, pack_dtype=pack_dtype)
    return Tensor._make_plain(torch.full(size, fill_value, dtype=dtype, device=device, **kwargs))


def tensor(data, *, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    if isinstance(dtype, _Bit1DType):
        return Tensor._make_bit1(torch.tensor(data, dtype=torch.bool, device=device),
                                 resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.tensor(data, dtype=dtype, device=device, **kwargs))


def as_tensor(data, *, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    if isinstance(dtype, _Bit1DType):
        if isinstance(data, torch.Tensor):
            bool_t = data.as_subclass(torch.Tensor) if isinstance(data, Tensor) else data
            bool_t = bool_t if bool_t.dtype == torch.bool else bool_t.bool()
        else:
            bool_t = torch.as_tensor(data, dtype=torch.bool, device=device)
        if device is not None:
            bool_t = bool_t.to(device=device)
        return Tensor._make_bit1(bool_t, resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.as_tensor(data, dtype=dtype, device=device))


def from_numpy(ndarray) -> Tensor:
    return Tensor._make_plain(torch.from_numpy(ndarray))


# Random creators 

def rand(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8,
         generator=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.rand(size, device=device, **gen_kw) >= 0.5
        return Tensor._make_bit1(bool_t, resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.rand(size, dtype=dtype, device=device, **gen_kw, **kwargs))


def randn(*size, dtype=None, device=None, pack_dtype: torch.dtype = torch.uint8,
          generator=None, **kwargs) -> Tensor:
    size = _norm_size(size)
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        bool_t = torch.randn(size, device=device, **gen_kw) >= 0
        return Tensor._make_bit1(bool_t, resolve_pack_dtype(pack_dtype))
    return Tensor._make_plain(torch.randn(size, dtype=dtype, device=device, **gen_kw, **kwargs))


def randint(low_or_high, high=None, size=None, *, dtype=None, device=None,
            pack_dtype: torch.dtype = torch.uint8, generator=None, **kwargs) -> Tensor:
    gen_kw = {'generator': generator} if generator is not None else {}
    if isinstance(dtype, _Bit1DType):
        s = size if size is not None else (low_or_high,)
        bool_t = torch.randint(0, 2, s, device=device, **gen_kw).bool()
        return Tensor._make_bit1(bool_t, resolve_pack_dtype(pack_dtype))
    if high is None:
        return Tensor._make_plain(torch.randint(low_or_high, size, dtype=dtype, device=device, **gen_kw, **kwargs))
    return Tensor._make_plain(torch.randint(low_or_high, high, size, dtype=dtype, device=device, **gen_kw, **kwargs))


def _inherit_pack_dtype(input: Tensor, pack_dtype: torch.dtype) -> torch.dtype:
    """Return the caller-supplied pack_dtype if explicitly set, else inherit from input."""
    if pack_dtype is not torch.uint8:
        return resolve_pack_dtype(pack_dtype)
    inherited = getattr(input, '_pack_dtype', None)
    return inherited if inherited is not None else torch.uint8


def rand_like(input: Tensor, *, dtype=None, device=None,
              pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.rand_like(input.as_subclass(torch.Tensor).float()) >= 0.5
        dev = device or input.device
        return Tensor._make_bit1(bool_t.to(device=dev), _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.rand_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def randn_like(input: Tensor, *, dtype=None, device=None,
               pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.randn_like(input.as_subclass(torch.Tensor).float()) >= 0
        dev = device or input.device
        return Tensor._make_bit1(bool_t.to(device=dev), _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.randn_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def zeros_like(input: Tensor, *, dtype=None, device=None,
               pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.zeros_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.zeros_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def ones_like(input: Tensor, *, dtype=None, device=None,
              pack_dtype: torch.dtype = torch.uint8) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.ones_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.ones_like(input.as_subclass(torch.Tensor), dtype=dtype, device=device))


def full_like(input: Tensor, fill_value, *, dtype=None, device=None,
              pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.full_like(input.as_subclass(torch.Tensor), bool(fill_value),
                                 dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.full_like(
        input.as_subclass(torch.Tensor) if isinstance(input, Tensor) else input,
        fill_value, dtype=dtype, device=device, **kwargs,
    ))


def empty_like(input: Tensor, *, dtype=None, device=None,
               pack_dtype: torch.dtype = torch.uint8, **kwargs) -> Tensor:
    eff_dtype = dtype if dtype is not None else input.dtype
    if isinstance(eff_dtype, _Bit1DType):
        bool_t = torch.empty_like(input.as_subclass(torch.Tensor), dtype=torch.bool, device=device)
        return Tensor._make_bit1(bool_t, _inherit_pack_dtype(input, pack_dtype))
    return Tensor._make_plain(torch.empty_like(
        input.as_subclass(torch.Tensor) if isinstance(input, Tensor) else input,
        dtype=dtype, device=device, **kwargs,
    ))


# Arange / linspace / eye 

def arange(start, end=None, step=1, *, dtype=None, device=None, **kwargs) -> Tensor:
    if end is None:
        return Tensor._make_plain(torch.arange(start, dtype=dtype, device=device, **kwargs))
    return Tensor._make_plain(torch.arange(start, end, step, dtype=dtype, device=device, **kwargs))


def linspace(start, end, steps, *, dtype=None, device=None, **kwargs) -> Tensor:
    return Tensor._make_plain(torch.linspace(start, end, steps, dtype=dtype, device=device, **kwargs))


def eye(n, m=None, *, dtype=None, device=None, **kwargs) -> Tensor:
    if m is None:
        return Tensor._make_plain(torch.eye(n, dtype=dtype, device=device, **kwargs))
    return Tensor._make_plain(torch.eye(n, m, dtype=dtype, device=device, **kwargs))


# Re-exports: work correctly via __torch_function__ 
# All of the ops below unwrap brute.Tensor inputs to their bool/underlying base,
# execute the standard torch op, and re-wrap results (including bit1 promotion)
# via brute.Tensor.__torch_function__. No custom implementation is needed.

# Reductions
all   = torch.all
any   = torch.any
sum   = torch.sum
max   = torch.max
min   = torch.min
mean  = torch.mean
prod  = torch.prod
amax  = torch.amax
amin  = torch.amin
aminmax       = torch.aminmax
argmax        = torch.argmax
argmin        = torch.argmin
count_nonzero = torch.count_nonzero
nonzero       = torch.nonzero
argwhere      = torch.argwhere
nansum        = torch.nansum
nanmean       = torch.nanmean
logsumexp     = torch.logsumexp

# Combining / stacking
cat   = torch.cat
stack = torch.stack

# Shape manipulation
reshape           = torch.reshape
flatten           = torch.flatten
squeeze           = torch.squeeze
unsqueeze         = torch.unsqueeze
permute           = torch.permute
transpose         = torch.transpose
t                 = torch.t
movedim           = torch.movedim
moveaxis          = torch.moveaxis
swapaxes          = torch.swapaxes
swapdims          = torch.swapdims
broadcast_to      = torch.broadcast_to
broadcast_tensors = torch.broadcast_tensors
narrow            = torch.narrow
select            = torch.select
atleast_1d        = torch.atleast_1d
atleast_2d        = torch.atleast_2d
atleast_3d        = torch.atleast_3d

# Splitting
split         = torch.split
chunk         = torch.chunk
unbind        = torch.unbind
tensor_split  = torch.tensor_split
hsplit        = torch.hsplit
vsplit        = torch.vsplit
dsplit        = torch.dsplit

# Clone / detach
clone  = torch.clone
detach = torch.detach

# Logical (particularly meaningful for bit1)
where         = torch.where
logical_and   = torch.logical_and
logical_or    = torch.logical_or
logical_xor   = torch.logical_xor
logical_not   = torch.logical_not

# Bitwise (meaningful for bool/bit1)
bitwise_and         = torch.bitwise_and
bitwise_or          = torch.bitwise_or
bitwise_xor         = torch.bitwise_xor
bitwise_not         = torch.bitwise_not
bitwise_left_shift  = torch.bitwise_left_shift
bitwise_right_shift = torch.bitwise_right_shift

# Comparison
eq            = torch.eq
ne            = torch.ne
lt            = torch.lt
le            = torch.le
gt            = torch.gt
ge            = torch.ge
equal         = torch.equal
allclose      = torch.allclose
isclose       = torch.isclose
isnan         = torch.isnan
isinf         = torch.isinf
isfinite      = torch.isfinite
isneginf      = torch.isneginf
isposinf      = torch.isposinf

# Arithmetic
add      = torch.add
sub      = torch.sub
mul      = torch.mul
div      = torch.div
divide   = torch.divide
neg      = torch.neg
negative = torch.negative
abs      = torch.abs
absolute = torch.absolute
sign     = torch.sign
sgn      = torch.sgn
clamp    = torch.clamp
clip     = torch.clamp
pow      = torch.pow
sqrt     = torch.sqrt
rsqrt    = torch.rsqrt

# Cumulative
cumsum    = torch.cumsum
cumprod   = torch.cumprod
cummax    = torch.cummax
cummin    = torch.cummin

# Sorting / selection
sort              = torch.sort
argsort           = torch.argsort
topk              = torch.topk
kthvalue          = torch.kthvalue
msort             = torch.msort
unique            = torch.unique
unique_consecutive = torch.unique_consecutive

# Indexing / masking
gather        = torch.gather
index_select  = torch.index_select
masked_select = torch.masked_select
take          = torch.take
scatter       = torch.scatter

# Rearrangement / replication
roll              = torch.roll
flip              = torch.flip
fliplr            = torch.fliplr
flipud            = torch.flipud
rot90             = torch.rot90
tile              = torch.tile
repeat_interleave = torch.repeat_interleave

# Matrix / linear algebra
mm      = torch.mm
bmm     = torch.bmm
matmul  = torch.matmul
mv      = torch.mv
dot     = torch.dot
inner   = torch.inner
outer   = torch.outer
vdot    = torch.vdot
cross   = torch.cross

# Diagonal / triangular
diagonal   = torch.diagonal
diag       = torch.diag
diag_embed = torch.diag_embed
diagflat   = torch.diagflat
tril       = torch.tril
triu       = torch.triu
trace      = torch.trace

# Misc
nan_to_num        = torch.nan_to_num
diff              = torch.diff
block_diag        = torch.block_diag
cartesian_prod    = torch.cartesian_prod
