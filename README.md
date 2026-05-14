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
git clone https://github.com/TheoCoombes/Brute.git
cd Brute
pip install --no-build-isolation -ve .
```

## Benchmarks

```bash
pytest tests/benchmarks --benchmark-only \
        --benchmark-group-by=group \
        --benchmark-sort=name \
        --benchmark-json=.benchmarks/full.json

python -m tests.benchmarks.make_report .benchmarks/full.json
```

