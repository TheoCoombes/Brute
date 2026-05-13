#pragma once

#include <torch/torch.h>

namespace cbrute { namespace cpu {

//  Packing / unpacking 
at::Tensor  pack_bits   (const at::Tensor& input,  int64_t pack_width);                       // float→packed
at::Tensor  pack_bool   (const at::Tensor& input,  int64_t pack_width);                       // bool →packed
at::Tensor  unpack_bits (const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw);// packed→±1 f32
at::Tensor  unpack_bool (const at::Tensor& packed, at::IntArrayRef logical_shape, int64_t pw);// packed→bool

//  Matmul 
at::Tensor  xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B,
                                 int64_t K, int64_t pack_width);

//  Popcount / hamming (generic int dtype + bit1) 
at::Tensor  popcount         (const at::Tensor& x);             // per-element, int32 out
at::Tensor  packed_popcount  (const at::Tensor& x);             // total bits over buffer, int64
at::Tensor  hamming_distance (const at::Tensor& A,
                              const at::Tensor& B);              // per-element popcount(A^B)
at::Tensor  bit1_hamming_total(const at::Tensor& A,
                               const at::Tensor& B);             // total popcount(A^B), no broadcast

//  Bitwise (delegate to at:: which is already SIMD-vectorized) 
at::Tensor  bitwise_and(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_or (const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_xor(const at::Tensor& A, const at::Tensor& B);
at::Tensor  bitwise_not(const at::Tensor& A);

//  Misc 
at::Tensor& randomize_bits(at::Tensor& out);

}} // cbrute::cpu
