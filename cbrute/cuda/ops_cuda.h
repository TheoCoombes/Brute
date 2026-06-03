#pragma once
#ifdef HAVE_CUDA

#include <torch/torch.h>

namespace cbrute { namespace cuda {

//  Packing / unpacking — bit1 is uint64-packed (64 bits per int64 word).
at::Tensor  pack_bool   (const at::Tensor& input);
at::Tensor  pack_sign   (const at::Tensor& input);
at::Tensor  unpack_bits (const at::Tensor& packed, at::IntArrayRef logical_shape);
at::Tensor  unpack_bool (const at::Tensor& packed, at::IntArrayRef logical_shape);

//  Matmul (CUTLASS B1 XOR-popc on sm_80+, hand kernel otherwise)
at::Tensor  xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
bsr_scan(const at::Tensor& q, const at::Tensor& assoc,
         const at::Tensor& decay_shifts, int64_t D);

//  Fused sign ops
at::Tensor  xnor_popcount_matmul_sign(const at::Tensor& A, const at::Tensor& B, int64_t K);
at::Tensor  packed_majority(const at::Tensor& rows, int64_t k, int64_t D);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
episodic_causal_search(const at::Tensor& qc, const at::Tensor& kc_buf,
                        const at::Tensor& qp, const at::Tensor& pos_buf,
                        const at::Tensor& payload, const at::Tensor& cnt, int64_t D);

//  Popcount / hamming
at::Tensor  popcount         (const at::Tensor& x);
at::Tensor  packed_popcount  (const at::Tensor& x);
at::Tensor  hamming_distance (const at::Tensor& A, const at::Tensor& B);
at::Tensor  bit1_hamming_total(const at::Tensor& A, const at::Tensor& B);

at::Tensor& randomize_bits(at::Tensor& out);

}} // cbrute::cuda
#endif // HAVE_CUDA
