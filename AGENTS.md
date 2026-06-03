# Brute Agent Guide

## Project

`brute` is a packed 1-bit tensor library built on PyTorch. It provides a
`brute.Tensor` subclass with `brute.bit1` dtype semantics and a compiled
`_cbrute` extension for CPU, Apple MPS/Metal, and CUDA backends.

The root project is the library. The `model/` directory is a consumer project
for the Haemmr binary language model and should be treated as separate
application code.

## Layout

- `brute/`: Python tensor subclass, dtype, factories, direct fast-path helpers,
  streams, and CUDA graph utilities.
- `cbrute/ext.cpp`: PyTorch operator schemas and per-device dispatch
  registration.
- `cbrute/cpu/`: CPU drivers and Highway kernels.
- `cbrute/metal/`: MPS drivers and Metal kernels embedded into the extension at
  build time.
- `cbrute/cuda/`: CUDA drivers plus fallback/CUTLASS kernels.
- `tests/unit/`: main unit and parity coverage.
- `tests/fuzz/`, `tests/stress/`: broader randomized and long-running coverage.
- `tests/benchmarks/`: benchmark harness and report generation.
- `examples/`: older binary-network experiments that are not part of the core
  extension API.
- `third_party/`: vendored Highway and CUTLASS dependencies.

## Build

Use the workspace virtual environment from the repository root:

```bash
./.venv/bin/python -m pip install --no-build-isolation -ve .
```

The macOS build compiles Metal kernels in `cbrute/metal/kernels/*.metal`,
links them into a metallib, and embeds the bytes into the extension. CUDA is
enabled when the local PyTorch/CMake toolchain exposes CUDA.

## Tests

Fast validation:

```bash
./.venv/bin/python -m pytest tests/unit -q
```

Focused kernel validation:

```bash
./.venv/bin/python -m pytest tests/unit/test_packed_kernels.py -q
```

The test suite parametrizes over CPU, CUDA, and MPS when available. CUDA is not
available on this Mac workspace, so CUDA changes must compile by inspection
locally and be tested on a CUDA host later.

## Kernel API Notes

All bit1 buffers use `int64` storage interpreted as unsigned 64-bit packed
words, with the logical last dimension packed LSB-first. Padding bits in the
last word of each row must remain zero.

Current model-facing fused ops:

- `pack_bool`: bool bytes to packed bit1 storage.
- `pack_sign`: numeric `x >= 0` threshold directly to packed bit1 storage.
- `xnor_popcount_matmul`: packed XNOR/popcount matrix multiply returning signed
  int32 similarities.
- `bsr_scan`: packed BSR forward scan returning packed read/state/gate buffers.

When adding an op, update all applicable backend headers, drivers, schemas in
`cbrute/ext.cpp`, Python fast-path helpers if needed, and tests. Avoid Python
fallbacks for hot paths unless they are explicitly outside the packed contract.

## Editing Rules

Do not rewrite the tensor subclass casually. Many operations rely on lazy bool
materialisation and packed-cache versioning. Prefer narrow changes and add
targeted tests under `tests/unit/` for every new packed behavior.
