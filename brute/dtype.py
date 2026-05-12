import torch
import struct

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

def get_optimal_pack_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """
    Detect the optimal pack dtype for a given device.

    Strategy:
    - 64-bit systems (most modern): uint64 (best packing efficiency)
    - 32-bit systems: uint32
    - CUDA: uint32 for older GPUs (pre-Maxwell), uint64 for modern
    - CPU: uint64 (modern systems are 64-bit)
    - Fallback: uint64
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Check system pointer size (64-bit vs 32-bit architecture).
    is_64bit = struct.calcsize("P") == 8

    if device.type == "cuda":
        # For CUDA, prefer uint32 for older GPUs (pre-Maxwell, CC < 5.2)
        # and uint64 for modern GPUs. If we can't determine capability,
        # default to uint64 (most common in practice).
        try:
            props = torch.cuda.get_device_properties(device.index or 0)
            cc = props.major, props.minor
            # Maxwell (CC 5.2+) and newer handle uint64 well
            return torch.uint64 if cc >= (5, 2) else torch.uint32
        except RuntimeError:
            # If we can't get properties, default to uint64
            return torch.uint64

    # CPU, MPS, and other backends: use uint64 on 64-bit systems
    if is_64bit:
        return torch.uint64
    else:
        # 32-bit systems: use uint32 for alignment and address space
        return torch.uint32

def resolve_pack_dtype(pt) -> torch.dtype:
    """Validate and return a pack dtype (torch.uint8 / uint32 / uint64)."""
    if pt in _PACK_BITS:
        return pt
    raise TypeError(
        f"pack_dtype must be brute.uint8, brute.uint32, or brute.uint64 "
        f"(eqv. torch.uint8/uint32/uint64); got {pt!r}"
    )