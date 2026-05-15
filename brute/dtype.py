import torch

_bool = bool

class _Bit1DType:
    """Sentinel dtype object for 1-bit packed tensors."""
    def __repr__(self) -> str: return "brute.bit1"
    def __str__(self)  -> str: return "bit1"
    def __eq__(self, other) -> _bool: return isinstance(other, _Bit1DType)
    def __hash__(self) -> int: return hash('brute._Bit1DType_singleton')


# Re-export data types for convenience (see `__init__.py`).
bit1            = _Bit1DType()
bool            = torch.bool
float32         = torch.float32
float64         = torch.float64
float16         = torch.float16
bfloat16        = torch.bfloat16
int8            = torch.int8
int16           = torch.int16
int32           = torch.int32
int64           = torch.int64
uint8           = torch.uint8
uint16          = torch.uint16
uint32          = torch.uint32
uint64          = torch.uint64
complex64       = torch.complex64
complex128      = torch.complex128
float8_e4m3fn   = torch.float8_e4m3fn
float8_e5m2     = torch.float8_e5m2

_PACK_WIDTH: int = 64
_PACK_STORAGE_DTYPE: torch.dtype = torch.int64