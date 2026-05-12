import torch

class _Bit1DType:
    """Sentinel dtype object for 1-bit packed tensors."""
    def __repr__(self) -> str: return "brute.bit1"
    def __str__(self)  -> str: return "bit1"
    def __eq__(self, other) -> bool: return isinstance(other, _Bit1DType)
    def __hash__(self) -> int: return hash('brute._Bit1DType_singleton')

bit1 = _Bit1DType()

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
        f"(eqv. torch.uint8/uint32/uint64); got {pt!r}"
    )