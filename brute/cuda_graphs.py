"""CUDA Graphs helpers for amortising bit1 dispatch overhead.

The dispatch column in the benchmark report is 0.05–0.20× across nearly every
op on CUDA. That's almost entirely Python `__torch_function__` + PyTorch
dispatcher + cudaLaunchKernel overhead. For inference loops where the same
shape sequence runs every step, capturing once with CUDA Graphs replaces all
of it with a single `cudaGraphLaunch` (~3 µs).

Usage:

    from brute.cuda_graphs import capture, replay

    # Warm-up phase — compiler picks up shapes, allocator stabilises.
    for _ in range(3):
        out = my_inference_function(x_packed)

    # Capture once.
    g = capture(lambda: my_inference_function(x_packed))

    # Replay many times — almost zero per-call overhead.
    for _ in range(1000):
        replay(g)

`capture` runs the closure once inside a `torch.cuda.graph(g)` context. The
returned object is the recorded `torch.cuda.CUDAGraph`; the closure's outputs
are accessible via `g.outputs`.

This is a thin convenience wrapper — for advanced control (stream pools,
input-buffer mutation), use `torch.cuda.graph` directly.
"""

from __future__ import annotations

from typing import Callable, Any
import torch


def capture(fn: Callable[[], Any], warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Capture `fn()` into a CUDA graph after `warmup` warm runs.

    The returned graph carries `.outputs` — the result of the captured `fn()`.
    Subsequent `replay(g)` calls re-execute the captured kernel sequence with
    no Python or dispatcher overhead.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Graphs require a CUDA build of PyTorch")

    # Warm-up to stabilise the allocator / autotuner / shape specialisations.
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            out = fn()
    torch.cuda.current_stream().wait_stream(s)
    g.outputs = out  # type: ignore[attr-defined]
    return g


def replay(g: torch.cuda.CUDAGraph) -> Any:
    """Replay a captured graph and return the recorded outputs."""
    g.replay()
    return getattr(g, 'outputs', None)


__all__ = ["capture", "replay"]
