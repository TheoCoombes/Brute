#pragma once
#ifdef HAVE_MPS

#include <torch/torch.h>

namespace cbrute { namespace mps {

// ── Packing / unpacking ──────────────────────────────────────────────────────
at::Tensor  pack_bits   (const at::Tensor& input,  int64_t pack_width);
at::Tensor  pack_bool   (const at::Tensor& input,  int64_t pack_width);
at::Tensor  unpack_bits (const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw);
at::Tensor  unpack_bool (const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw);

// ── Matmul ──────────────────────────────────────────────────────────────────
at::Tensor  xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B,
                                 int64_t K, int64_t pack_width);

// ── Popcount / hamming ──────────────────────────────────────────────────────
at::Tensor  popcount         (const at::Tensor& x);
at::Tensor  packed_popcount  (const at::Tensor& x);
at::Tensor  hamming_distance (const at::Tensor& A, const at::Tensor& B);
at::Tensor  bit1_hamming_total(const at::Tensor& A, const at::Tensor& B);

// ── Bitwise ─────────────────────────────────────────────────────────────────
at::Tensor  bitwise_and(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_or (const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_xor(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_not(const at::Tensor& A);

at::Tensor& randomize_bits(at::Tensor& out);

}} // cbrute::mps
#endif // HAVE_MPS
