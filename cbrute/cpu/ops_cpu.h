#pragma once

#include <torch/torch.h>

namespace cbrute { namespace cpu {

//  Packing / unpacking — bit1 is uint64-packed (64 bits per int64 word).
at::Tensor  pack_bool   (const at::Tensor& input);                               // bool →packed
at::Tensor  unpack_bits (const at::Tensor& packed, at::IntArrayRef logical_shape); // packed→±1 f32
at::Tensor  unpack_bool (const at::Tensor& packed, at::IntArrayRef logical_shape); // packed→bool

//  Matmul: A,B are uint64-packed (M,Kp),(N,Kp); K is the logical last dim.
at::Tensor  xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K);

//  Signed vote bundle: out = sign(Σ_n W[...,n]·pm1(V[...,n,:])).
//    W: int8 (B,M,N) integer weights;  V: uint64-packed bit1 (B,N,Dp);
//    returns uint64-packed bit1 (B,M,Dp).  D = logical last dim of V.
at::Tensor  signed_bundle(const at::Tensor& W, const at::Tensor& V, int64_t D);

//  Popcount / hamming (generic int dtype + bit1)
at::Tensor  popcount         (const at::Tensor& x);             // per-element, int32 out
at::Tensor  packed_popcount  (const at::Tensor& x);             // total bits over buffer, int64
at::Tensor  hamming_distance (const at::Tensor& A,
                              const at::Tensor& B);              // per-element popcount(A^B)
at::Tensor  bit1_hamming_total(const at::Tensor& A,
                               const at::Tensor& B);             // total popcount(A^B), no broadcast

//  Misc
at::Tensor& randomize_bits(at::Tensor& out);

}} // cbrute::cpu
