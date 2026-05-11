# Brute

An ultra-fast, lightweight binary deep learning framework with full PyTorch compatibility.

### Features

- Native 1-bit tensors, with an optimised CPU implementation and custom kernels for CUDA/Metal.
- Built using LibTorch, hence has full PyTorch compatibility and pre-existing hardware optimisations for other dtypes.
- ...

### Example
```py
import brute

x = brute.Tensor([1, 0, 1], dtype=brute.bit1, device='cuda:0')
y = brute.Tensor([0, 0, 1], dtype=brute.bit1, device='cuda:0')

# XNOR-popcount matmul
x @ y
>>> brute.Tensor([1, 0, 1], dtype=brute.bit1, device='cuda:0')

# Native popcount
x.popcount()
>>> 2
```

## Installation

```bash
pip install --no-build-isolation -ve .
```