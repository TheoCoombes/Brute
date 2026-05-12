#include <torch/extension.h>

PYBIND11_MODULE(_cbrute, m) {
    m.doc() = "brute C++ backend — 1-bit tensors on a PyTorch foundation";

    // Torch ops are registered via TORCH_LIBRARY in ops.cpp
}
