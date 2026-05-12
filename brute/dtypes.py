import torch

# ── Pack-dtype helpers ─────────────────────────────────────────────────────────
# A pack_dtype is a plain torch.dtype — torch.uint8, torch.uint32, or torch.uint64.
# No custom enum needed: brute.uint8 is torch.uint8.

_PACK_BITS: dict[torch.dtype, int] = {
    torch.uint8:  8,
    torch.uint32: 32,
    torch.uint64: 64,
}

def _resolve_pack_dtype(pt) -> torch.dtype:
    """Validate and return a pack dtype (torch.uint8 / uint32 / uint64)."""
    if pt in _PACK_BITS:
        return pt
    raise TypeError(
        f"pack_dtype must be brute.uint8, brute.uint32, or brute.uint64 "
        f"(i.e. torch.uint8/uint32/uint64); got {pt!r}"
    )


# ── Dtype sentinel ─────────────────────────────────────────────────────────────

class _Bit1DType:
    """Sentinel dtype object for 1-bit packed tensors. Strictly distinct from torch.bool."""

    def __repr__(self) -> str: return "brute.bit1"
    def __str__(self)  -> str: return "bit1"
    def __eq__(self, other) -> bool: return isinstance(other, _Bit1DType)
    def __hash__(self) -> int: return hash('brute._Bit1DType_singleton')


bit1 = _Bit1DType()
float32  = torch.float32
float64  = torch.float64
float16  = torch.float16
bfloat16 = torch.bfloat16
int8     = torch.int8
int16    = torch.int16
int32    = torch.int32
int64    = torch.int64
uint8    = torch.uint8
uint16   = torch.uint16
uint32   = torch.uint32
uint64   = torch.uint64
complex64  = torch.complex64
complex128 = torch.complex128
float8_e4m3fn = torch.float8_e4m3fn
float8_e5m2   = torch.float8_e5m2