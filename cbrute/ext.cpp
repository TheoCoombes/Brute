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

// ── Device-agnostic fallbacks ─────────────────────────────────────────────────
//
// The new ops `pack_bool`, `unpack_bool`, `bit1_hamming_total` ship with native
// CPU + CUDA kernels but no Metal implementation. Rather than refusing on MPS
// (or any future backend that misses a kernel), we register a *composite*
// fallback that decomposes each into ops that already have per-backend impls.
// The CPU / CUDA native registrations still win for those devices because
// per-backend impls have higher priority than CompositeExplicitAutograd.
//
// Decompositions:
//   pack_bool(b, pw)          ≡ pack_bits((b.float() * 2 − 1), pw)
//   unpack_bool(p, shape, pw) ≡ (unpack_bits(p, shape, pw) > 0)
//   bit1_hamming_total(A, B)  ≡ popcount(bitwise_xor(A, B)).sum()

namespace {

template <typename Sig>
auto _brute_op(const char* name) {
    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("brute::", name)
        .typed<Sig>();
}

at::Tensor pack_bool_composite(const at::Tensor& input, int64_t pack_width) {
    auto pm1 = (input.to(at::kFloat) * 2.0f - 1.0f).contiguous();
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("brute::pack_bits", "")
        .typed<at::Tensor(const at::Tensor&, int64_t)>();
    return op.call(pm1, pack_width);
}

at::Tensor unpack_bool_composite(const at::Tensor& packed,
                                 at::IntArrayRef logical_shape, int64_t pw) {
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("brute::unpack_bits", "")
        .typed<at::Tensor(const at::Tensor&, at::IntArrayRef, int64_t)>();
    auto pm1 = op.call(packed, logical_shape, pw);
    return pm1.gt(0.0f);
}

at::Tensor bit1_hamming_total_composite(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.sizes()       == B.sizes(),       "bit1_hamming_total: shape mismatch");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "bit1_hamming_total: dtype mismatch");
    static auto popcnt = c10::Dispatcher::singleton()
        .findSchemaOrThrow("brute::popcount", "")
        .typed<at::Tensor(const at::Tensor&)>();
    auto xored = at::bitwise_xor(A, B);
    return popcnt.call(xored).to(at::kLong).sum();
}

} // anon

// ── Schema ────────────────────────────────────────────────
TORCH_LIBRARY(brute, m) {
    // packing / unpacking
    m.def("pack_bits(Tensor input, int pack_width) -> Tensor");
    m.def("pack_bool(Tensor input, int pack_width) -> Tensor");
    m.def("unpack_bits(Tensor packed, int[] logical_shape, int pack_width) -> Tensor");
    m.def("unpack_bool(Tensor packed, int[] logical_shape, int pack_width) -> Tensor");

    // matmul
    m.def("xnor_popcount_matmul(Tensor A, Tensor B, int K, int pack_width) -> Tensor");

    // popcount / hamming
    m.def("popcount(Tensor x) -> Tensor");
    m.def("packed_popcount(Tensor x) -> Tensor");
    m.def("hamming_distance(Tensor A, Tensor B) -> Tensor");
    m.def("bit1_hamming_total(Tensor A, Tensor B) -> Tensor");

    // bitwise (pass-through to at:: for non-bit1 paths)
    m.def("bitwise_and(Tensor A, Tensor B) -> Tensor");
    m.def("bitwise_or(Tensor A, Tensor B) -> Tensor");
    m.def("bitwise_xor(Tensor A, Tensor B) -> Tensor");
    m.def("bitwise_not(Tensor A) -> Tensor");

    m.def("randomize_bits(Tensor(a!) out) -> Tensor(a!)");
}

// ── Composite (any-backend) fallbacks ─────────────────────
TORCH_LIBRARY_IMPL(brute, CompositeExplicitAutograd, m) {
    m.impl("pack_bool",          pack_bool_composite);
    m.impl("unpack_bool",        unpack_bool_composite);
    m.impl("bit1_hamming_total", bit1_hamming_total_composite);
}

// ── CPU ───────────────────────────────────────────────────
TORCH_LIBRARY_IMPL(brute, CPU, m) {
    m.impl("pack_bits",            cbrute::cpu::pack_bits);
    m.impl("pack_bool",            cbrute::cpu::pack_bool);
    m.impl("unpack_bits",          cbrute::cpu::unpack_bits);
    m.impl("unpack_bool",          cbrute::cpu::unpack_bool);
    m.impl("xnor_popcount_matmul", cbrute::cpu::xnor_popcount_matmul);
    m.impl("popcount",             cbrute::cpu::popcount);
    m.impl("packed_popcount",      cbrute::cpu::packed_popcount);
    m.impl("hamming_distance",     cbrute::cpu::hamming_distance);
    m.impl("bit1_hamming_total",   cbrute::cpu::bit1_hamming_total);
    m.impl("bitwise_and",          cbrute::cpu::bitwise_and);
    m.impl("bitwise_or",           cbrute::cpu::bitwise_or);
    m.impl("bitwise_xor",          cbrute::cpu::bitwise_xor);
    m.impl("bitwise_not",          cbrute::cpu::bitwise_not);
    m.impl("randomize_bits",       cbrute::cpu::randomize_bits);
}

// ── Metal/MPS native impls — full op coverage with simdgroup kernels. ─────
// The 3 ops previously falling through to CompositeExplicitAutograd
// (pack_bool, unpack_bool, bit1_hamming_total) now have native MSL kernels.
#ifdef HAVE_MPS
TORCH_LIBRARY_IMPL(brute, MPS, m) {
    m.impl("pack_bits",            cbrute::mps::pack_bits);
    m.impl("pack_bool",            cbrute::mps::pack_bool);
    m.impl("unpack_bits",          cbrute::mps::unpack_bits);
    m.impl("unpack_bool",          cbrute::mps::unpack_bool);
    m.impl("xnor_popcount_matmul", cbrute::mps::xnor_popcount_matmul);
    m.impl("popcount",             cbrute::mps::popcount);
    m.impl("packed_popcount",      cbrute::mps::packed_popcount);
    m.impl("hamming_distance",     cbrute::mps::hamming_distance);
    m.impl("bit1_hamming_total",   cbrute::mps::bit1_hamming_total);
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
    m.impl("pack_bool",            cbrute::cuda::pack_bool);
    m.impl("unpack_bits",          cbrute::cuda::unpack_bits);
    m.impl("unpack_bool",          cbrute::cuda::unpack_bool);
    m.impl("xnor_popcount_matmul", cbrute::cuda::xnor_popcount_matmul);
    m.impl("popcount",             cbrute::cuda::popcount);
    m.impl("packed_popcount",      cbrute::cuda::packed_popcount);
    m.impl("hamming_distance",     cbrute::cuda::hamming_distance);
    m.impl("bit1_hamming_total",   cbrute::cuda::bit1_hamming_total);
    m.impl("bitwise_and",          cbrute::cuda::bitwise_and);
    m.impl("bitwise_or",           cbrute::cuda::bitwise_or);
    m.impl("bitwise_xor",          cbrute::cuda::bitwise_xor);
    m.impl("bitwise_not",          cbrute::cuda::bitwise_not);
    m.impl("randomize_bits",       cbrute::cuda::randomize_bits);
}
#endif
