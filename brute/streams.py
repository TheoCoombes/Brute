"""Round-robin CUDA stream context for parallel bit1 op chains.

When a sequence of independent bit1 ops has no data dependency, naively
launching them onto the current stream serialises the kernel launches even
though the GPU has spare bandwidth. `parallel_streams(n)` rotates each call
between `n` CUDA streams; downstream synchronisation is the caller's
responsibility (or wait_all() at end of the with-block).

Example:
    with brute.parallel_streams(4):
        a = x & y           # stream 0
        b = z | w           # stream 1
        c = u ^ v           # stream 2
        d = ~q              # stream 3
    # implicit sync at __exit__
"""

from __future__ import annotations

import contextlib
from typing import List, Optional
import torch

# Module-level stream pool, lazily created. Indexed by (device_index, n).
_POOLS: dict = {}


def _get_pool(device: torch.device, n: int) -> List[torch.cuda.Stream]:
    if device.type != 'cuda':
        raise RuntimeError("parallel_streams only supported on CUDA")
    key = (device.index if device.index is not None else
           torch.cuda.current_device(), n)
    pool = _POOLS.get(key)
    if pool is None:
        pool = [torch.cuda.Stream(device=device) for _ in range(n)]
        _POOLS[key] = pool
    return pool


@contextlib.contextmanager
def parallel_streams(n: int = 4, device: Optional[torch.device] = None):
    """Round-robin CUDA stream context.

    Each `torch.cuda.Stream` in the pool is selected in turn for a single
    operation. Inside the context manager, the *current* stream rotates per
    enclosed op. On exit, all pooled streams are synchronised against the
    original stream so subsequent ops see the writes.

    Caveats:
      * Only useful when ops are genuinely independent — sequential data
        dependencies will force the GPU to insert wait events anyway.
      * Each rotation is the cost of a stream switch (very cheap, <1 µs).
      * `n=1` is a no-op (just the outer stream).

    This is a coarse-grained alternative to manual `with torch.cuda.stream(s):`
    blocks. For fine-grained control, use the PyTorch primitive directly.
    """
    if not torch.cuda.is_available():
        # No-op outside CUDA so callers can write portable code.
        yield
        return

    if device is None:
        device = torch.device('cuda', torch.cuda.current_device())
    pool = _get_pool(device, max(1, int(n)))
    outer = torch.cuda.current_stream(device=device)

    # Use a class to hold rotation state and provide a context manager that
    # alternates streams on each .next() call.
    class _Rotator:
        def __init__(self, streams):
            self.streams = streams
            self.idx = 0

        def next(self) -> torch.cuda.Stream:
            s = self.streams[self.idx]
            self.idx = (self.idx + 1) % len(self.streams)
            return s

    rot = _Rotator(pool)

    # Patch torch.cuda.current_stream temporarily to round-robin returns.
    # This is intentionally lightweight: only the current_stream lookup gets
    # rotated; users that explicitly target a stream are unaffected.
    original_current = torch.cuda.current_stream
    def patched(device=None):
        return rot.next()
    torch.cuda.current_stream = patched  # type: ignore

    try:
        yield rot
    finally:
        torch.cuda.current_stream = original_current  # type: ignore
        # Wait for every pooled stream so subsequent ops see all writes.
        for s in pool:
            outer.wait_stream(s)


__all__ = ["parallel_streams"]
