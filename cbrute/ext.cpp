#include <torch/extension.h>
#include <ATen/core/dispatch/Dispatcher.h>
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

// brute ops always operate on uint64-packed (int64-stored) buffers: pack
// width is fixed at 64. Bitwise AND/OR/XOR/NOT are not exposed here — Python
// callers reach `_packed_buf.bitwise_*` directly, and PyTorch's native
// implementations on int64 are already SIMD-vectorised on every backend.

namespace {

// Pad-safe inversion of a packed bit1 buffer. `~A` over the whole buffer
// would flip the (zero) padding bits in every row's tail word, corrupting
// downstream popcount/sum/all/any.
//
// `last_dim_bits` = the LAST LOGICAL DIMENSION (`tensor.shape[-1]`).
at::Tensor bit1_not_packed_composite(const at::Tensor& A, int64_t last_dim_bits) {
    auto out = at::bitwise_not(A);
    if (out.numel() == 0 || last_dim_bits == 0) return out;
    constexpr int64_t kPackWidth = 64;
    int64_t valid_bits = last_dim_bits % kPackWidth;
    if (valid_bits == 0) return out;
    const uint64_t mask = (valid_bits == 64)
        ? ~uint64_t(0)
        : ((uint64_t(1) << valid_bits) - 1);
    auto last_col = out.select(-1, out.size(-1) - 1);
    last_col.bitwise_and_(at::scalar_tensor((int64_t)mask, last_col.options()));
    return out;
}

}  // anon

//  Schema
TORCH_LIBRARY(brute, m) {
    // packing / unpacking
    m.def("pack_bool(Tensor input) -> Tensor");
    m.def("unpack_bits(Tensor packed, int[] logical_shape) -> Tensor");
    m.def("unpack_bool(Tensor packed, int[] logical_shape) -> Tensor");

    // matmul
    m.def("xnor_popcount_matmul(Tensor A, Tensor B, int K) -> Tensor");

    // signed vote bundle: int8-weight × bit1 → bit1 (register-bound reduce)
    m.def("signed_bundle(Tensor W, Tensor V, int D) -> Tensor");

    // popcount / hamming
    m.def("popcount(Tensor x) -> Tensor");
    m.def("packed_popcount(Tensor x) -> Tensor");
    m.def("hamming_distance(Tensor A, Tensor B) -> Tensor");
    m.def("bit1_hamming_total(Tensor A, Tensor B) -> Tensor");

    // pad-safe NOT on the packed buffer of a bit1 tensor.
    m.def("bit1_not_packed(Tensor A, int last_dim_bits) -> Tensor");

    m.def("randomize_bits(Tensor(a!) out) -> Tensor(a!)");
}

//  Composite (any-backend) fallbacks — only `bit1_not_packed` needs one;
//  every other op has native impls on every backend we ship.
TORCH_LIBRARY_IMPL(brute, CompositeExplicitAutograd, m) {
    m.impl("bit1_not_packed", bit1_not_packed_composite);
}

//  CPU
TORCH_LIBRARY_IMPL(brute, CPU, m) {
    m.impl("pack_bool",            cbrute::cpu::pack_bool);
    m.impl("unpack_bits",          cbrute::cpu::unpack_bits);
    m.impl("unpack_bool",          cbrute::cpu::unpack_bool);
    m.impl("xnor_popcount_matmul", cbrute::cpu::xnor_popcount_matmul);
    m.impl("signed_bundle",        cbrute::cpu::signed_bundle);
    m.impl("popcount",             cbrute::cpu::popcount);
    m.impl("packed_popcount",      cbrute::cpu::packed_popcount);
    m.impl("hamming_distance",     cbrute::cpu::hamming_distance);
    m.impl("bit1_hamming_total",   cbrute::cpu::bit1_hamming_total);
    m.impl("randomize_bits",       cbrute::cpu::randomize_bits);
}

//  Metal/MPS
#ifdef HAVE_MPS
TORCH_LIBRARY_IMPL(brute, MPS, m) {
    m.impl("pack_bool",            cbrute::mps::pack_bool);
    m.impl("unpack_bits",          cbrute::mps::unpack_bits);
    m.impl("unpack_bool",          cbrute::mps::unpack_bool);
    m.impl("xnor_popcount_matmul", cbrute::mps::xnor_popcount_matmul);
    m.impl("signed_bundle",        cbrute::mps::signed_bundle);
    m.impl("popcount",             cbrute::mps::popcount);
    m.impl("packed_popcount",      cbrute::mps::packed_popcount);
    m.impl("hamming_distance",     cbrute::mps::hamming_distance);
    m.impl("bit1_hamming_total",   cbrute::mps::bit1_hamming_total);
    m.impl("randomize_bits",       cbrute::mps::randomize_bits);
}
#endif

//  CUDA
#ifdef HAVE_CUDA
TORCH_LIBRARY_IMPL(brute, CUDA, m) {
    m.impl("pack_bool",            cbrute::cuda::pack_bool);
    m.impl("unpack_bits",          cbrute::cuda::unpack_bits);
    m.impl("unpack_bool",          cbrute::cuda::unpack_bool);
    m.impl("xnor_popcount_matmul", cbrute::cuda::xnor_popcount_matmul);
    m.impl("signed_bundle",        cbrute::cuda::signed_bundle);
    m.impl("popcount",             cbrute::cuda::popcount);
    m.impl("packed_popcount",      cbrute::cuda::packed_popcount);
    m.impl("hamming_distance",     cbrute::cuda::hamming_distance);
    m.impl("bit1_hamming_total",   cbrute::cuda::bit1_hamming_total);
    m.impl("randomize_bits",       cbrute::cuda::randomize_bits);
}
#endif
