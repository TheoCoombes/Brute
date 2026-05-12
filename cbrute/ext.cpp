#include <torch/extension.h>
#include "ops_cpu.h"
#ifdef HAVE_MPS
#include "ops_metal.h"
#endif
#ifdef HAVE_CUDA
#include "ops_cuda.h"
#endif

PYBIND11_MODULE(_cbrute, m) {
    m.doc() = "brute C++ backend — 1-bit tensors on a PyTorch foundation";
}

// ── Schema ────────────────────────────────────────────────
TORCH_LIBRARY(brute, m) {
    m.def("pack_bits(Tensor input, int pack_width) -> Tensor");
    m.def("unpack_bits(Tensor packed, int[] logical_shape, int pack_width) -> Tensor");
    m.def("xnor_popcount_matmul(Tensor A, Tensor B, int K, int pack_width) -> Tensor");
    m.def("popcount(Tensor packed) -> Tensor");
    m.def("packed_popcount(Tensor packed) -> Tensor");
    m.def("hamming_distance(Tensor A, Tensor B) -> Tensor");
    m.def("bitwise_and(Tensor A, Tensor B) -> Tensor");
    m.def("bitwise_or(Tensor A, Tensor B) -> Tensor");
    m.def("bitwise_xor(Tensor A, Tensor B) -> Tensor");
    m.def("bitwise_not(Tensor A) -> Tensor");
    m.def("randomize_bits(Tensor(a!) out) -> Tensor(a!)");
}

// ── CPU ───────────────────────────────────────────────────
TORCH_LIBRARY_IMPL(brute, CPU, m) {
    m.impl("pack_bits",            cbrute::cpu::pack_bits);
    m.impl("unpack_bits",          cbrute::cpu::unpack_bits);
    m.impl("xnor_popcount_matmul", cbrute::cpu::xnor_popcount_matmul);
    m.impl("popcount",             cbrute::cpu::popcount);
    m.impl("packed_popcount",      cbrute::cpu::packed_popcount);
    m.impl("hamming_distance",     cbrute::cpu::hamming_distance);
    m.impl("bitwise_and",          cbrute::cpu::bitwise_and);
    m.impl("bitwise_or",           cbrute::cpu::bitwise_or);
    m.impl("bitwise_xor",          cbrute::cpu::bitwise_xor);
    m.impl("bitwise_not",          cbrute::cpu::bitwise_not);
    m.impl("randomize_bits",       cbrute::cpu::randomize_bits);
}

// ── Metal/MPS ─────────────────────────────────────────────
#ifdef HAVE_MPS
TORCH_LIBRARY_IMPL(brute, MPS, m) {
    m.impl("pack_bits",            cbrute::mps::pack_bits);
    m.impl("unpack_bits",          cbrute::mps::unpack_bits);
    m.impl("xnor_popcount_matmul", cbrute::mps::xnor_popcount_matmul);
    m.impl("popcount",             cbrute::mps::popcount);
    m.impl("packed_popcount",      cbrute::mps::packed_popcount);
    m.impl("hamming_distance",     cbrute::mps::hamming_distance);
    m.impl("bitwise_and",          cbrute::mps::bitwise_and);
    m.impl("bitwise_or",           cbrute::mps::bitwise_or);
    m.impl("bitwise_xor",          cbrute::mps::bitwise_xor);
    m.impl("bitwise_not",          cbrute::mps::bitwise_not);
    m.impl("randomize_bits",       cbrute::mps::randomize_bits);
}
#endif

// ── CUDA ──────────────────────────────────────────────────
#ifdef HAVE_CUDA
TORCH_LIBRARY_IMPL(brute, CUDA, m) {
    m.impl("pack_bits",            cbrute::cuda::pack_bits);
    m.impl("unpack_bits",          cbrute::cuda::unpack_bits);
    m.impl("xnor_popcount_matmul", cbrute::cuda::xnor_popcount_matmul);
    m.impl("popcount",             cbrute::cuda::popcount);
    m.impl("packed_popcount",      cbrute::cuda::packed_popcount);
    m.impl("hamming_distance",     cbrute::cuda::hamming_distance);
    m.impl("bitwise_and",          cbrute::cuda::bitwise_and);
    m.impl("bitwise_or",           cbrute::cuda::bitwise_or);
    m.impl("bitwise_xor",          cbrute::cuda::bitwise_xor);
    m.impl("bitwise_not",          cbrute::cuda::bitwise_not);
    m.impl("randomize_bits",       cbrute::cuda::randomize_bits);
}
#endif
