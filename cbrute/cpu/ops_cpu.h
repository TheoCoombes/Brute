#pragma once

#include <torch/torch.h>

namespace cbrute { namespace cpu {

//  Packing / unpacking — bit1 is uint64-packed (64 bits per int64 word).
at::Tensor  pack_bool   (const at::Tensor& input);                               // bool →packed
at::Tensor  pack_sign   (const at::Tensor& input);                               // sign(x)>=0 →packed
at::Tensor  unpack_bits (const at::Tensor& packed, at::IntArrayRef logical_shape); // packed→±1 f32
at::Tensor  unpack_bool (const at::Tensor& packed, at::IntArrayRef logical_shape); // packed→bool

//  Matmul: A,B are uint64-packed (M,Kp),(N,Kp); K is the logical last dim.
at::Tensor  xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K);

//  Sequence kernels.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
bsr_scan(const at::Tensor& q, const at::Tensor& assoc,
         const at::Tensor& decay_shifts, int64_t D);

//  Popcount / hamming (generic int dtype + bit1)
at::Tensor  popcount         (const at::Tensor& x);             // per-element, int32 out
at::Tensor  packed_popcount  (const at::Tensor& x);             // total bits over buffer, int64
at::Tensor  hamming_distance (const at::Tensor& A,
                              const at::Tensor& B);              // per-element popcount(A^B)
at::Tensor  bit1_hamming_total(const at::Tensor& A,
                               const at::Tensor& B);             // total popcount(A^B), no broadcast

//  Fused sign ops
at::Tensor  xnor_popcount_matmul_sign(const at::Tensor& A, const at::Tensor& B, int64_t K);
at::Tensor  packed_majority(const at::Tensor& rows, int64_t k, int64_t D);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
episodic_causal_search(const at::Tensor& qc, const at::Tensor& kc_buf,
                        const at::Tensor& qp, const at::Tensor& pos_buf,
                        const at::Tensor& payload, const at::Tensor& cnt, int64_t D);

//  Misc
at::Tensor& randomize_bits(at::Tensor& out);

}} // cbrute::cpu
