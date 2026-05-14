# Brute

An ultra-fast, lightweight binary deep learning framework extension for PyTorch.

### Features

- Native 1-bit tensors, with optimised CPU / CUDA / Apple Silicon kernels.
- Complete Torch backwards compatibility, `brute.Tensor` inherits from `torch.Tensor` and keeps all existing data types / hardware optimisations.
- Native binary support for `autograd`.
- Implementations of some recent BNN papers, including BOLD.

## Installation

```bash
pip install brute
```

### From Source

```bash
git clone --recurse-submodules -j8 https://github.com/TheoCoombes/Brute.git
cd Brute
pip install --no-build-isolation -ve .
```

## Tests

A single op catalog (`tests/ops_catalog.py`) drives three parity test
files and the benchmark suite. Adding a new op means adding one `OpSpec`.

```bash
# Unit parity (fast, every op × device × pack-dtype)
pytest tests/unit/test_parity_catalog.py

# Edge cases (empty / 0-dim / non-contig / broadcast / tail-misalign)
pytest tests/unit/test_edge_cases_catalog.py

# Hypothesis fuzz (BRUTE_FUZZ_EXAMPLES=N to widen)
pytest tests/unit/test_fuzz_catalog.py

# All unit tests
pytest tests/unit/
```

## Benchmarks

```bash
pytest tests/benchmarks --benchmark-only \
        --benchmark-group-by=group \
        --benchmark-sort=name \
        --benchmark-json=.benchmarks/$(date +%Y%m%d_%H%M).json

# Render markdown report (single run)
python -m tests.benchmarks.make_report

# Or, compare two runs (head vs base) to track per-op deltas
python -m tests.benchmarks.make_report base.json head.json -o diff.md
```

