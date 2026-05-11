#pragma once

#include <torch/torch.h>

namespace cbrute { namespace cpu {

at::Tensor  pack_bits(const at::Tensor& input, int64_t pack_width);
at::Tensor  unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pack_width);
at::Tensor  xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K, int64_t pack_width);
at::Tensor  popcount(const at::Tensor& packed);
at::Tensor  hamming_distance(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_and(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_or(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_xor(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_not(const at::Tensor& A);
at::Tensor& randomize_bits(at::Tensor& out);

}} // cbrute::cpu
