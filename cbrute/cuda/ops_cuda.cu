// brute CUDA backend — driver layer.
//
// All bit1 packed buffers are uint64 (int64 storage, 64 bits per word).
// Drivers do tensor validation, output allocation, and kernel launches on
// the current CUDA stream; all heavy lifting lives in `kernels/`.
//
// Invariants:
//   * Zero device-side allocations inside any kernel.
//   * All kernels see contiguous, type-stable raw pointers.
//   * Output is zero-initialised when pad bytes/bits must be 0; otherwise
//     allocated with at::empty.

#ifdef HAVE_CUDA
#include "ops_cuda.h"
#include "kernels/packing.cuh"
#include "kernels/popcount.cuh"
#include "kernels/matmul_fallback.cuh"
#include "kernels/matmul_cutlass.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

namespace cbrute { namespace cuda {

namespace {

constexpr int64_t PACK_WIDTH = 64;

inline cudaStream_t cur_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

inline int compute_capability() {
    int dev = -1;
    cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major * 10 + minor;
}

inline const uint64_t* as_u64(const at::Tensor& t) {
    return reinterpret_cast<const uint64_t*>(t.data_ptr());
}
inline uint8_t*        as_u8 (at::Tensor& t)       { return reinterpret_cast<uint8_t*>(t.data_ptr()); }
inline const uint8_t*  as_u8 (const at::Tensor& t) { return reinterpret_cast<const uint8_t*>(t.data_ptr()); }

}  // anon

//  pack_bool: bool input → int64 packed buffer.
at::Tensor pack_bool(const at::Tensor& input) {
    TORCH_CHECK(input.dim() >= 1, "pack_bool: input must have >= 1 dim");
    TORCH_CHECK(input.scalar_type() == at::kBool,
                "pack_bool: input must be torch.bool");

    const auto inp = input.contiguous();
    const int64_t ld = inp.size(-1);

    auto out_shape = inp.sizes().vec();
    if (ld == 0) {
        out_shape.back() = 0;
        return at::zeros(out_shape, inp.options().dtype(at::kLong));
    }

    const int64_t pd_words  = (ld + PACK_WIDTH - 1) / PACK_WIDTH;
    const int64_t row_bytes = pd_words * (PACK_WIDTH / 8);
    const int64_t batch     = inp.numel() / ld;
    out_shape.back() = pd_words;

    auto output = at::zeros(out_shape, inp.options().dtype(at::kLong));
    const int chunks = (int)((ld + 31) / 32);

    // Persistent kernel for large inputs amortises launch overhead.
    constexpr int64_t PERSISTENT_THRESHOLD = 32 * 1024;
    const int64_t total_chunks = batch * (int64_t)chunks;
    if (total_chunks >= PERSISTENT_THRESHOLD) {
        kernels::k_pack_bool_persistent<<<256, 256, 0, cur_stream()>>>(
            reinterpret_cast<const uint8_t*>(inp.data_ptr<bool>()),
            as_u8(output), ld, row_bytes, batch, chunks);
    } else {
        dim3 grid((unsigned)chunks, (unsigned)batch);
        kernels::k_pack_bool_warp<<<grid, 32, 0, cur_stream()>>>(
            reinterpret_cast<const uint8_t*>(inp.data_ptr<bool>()),
            as_u8(output), ld, row_bytes);
    }
    return output;
}

//  unpack_bits: packed → float32 ±1.0.
at::Tensor unpack_bits(const at::Tensor& packed, at::IntArrayRef logical_shape) {
    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0)
        return at::empty(logical_shape.vec(), p.options().dtype(at::kFloat));

    const int64_t pl_words  = p.size(-1);
    const int64_t row_bytes = pl_words * (PACK_WIDTH / 8);
    const int64_t batch     = p.numel() / pl_words;

    auto output = at::empty(logical_shape.vec(), p.options().dtype(at::kFloat));
    const int chunks = (int)((ll + 31) / 32);
    dim3 grid((unsigned)chunks, (unsigned)batch);
    kernels::k_unpack_pm1_warp<<<grid, 32, 0, cur_stream()>>>(
        as_u8(p), output.data_ptr<float>(), ll, row_bytes);
    return output;
}

//  unpack_bool: packed → bool.
at::Tensor unpack_bool(const at::Tensor& packed, at::IntArrayRef logical_shape) {
    const auto p = packed.contiguous();
    const int64_t ll = logical_shape.back();
    if (ll == 0 || p.numel() == 0)
        return at::empty(logical_shape.vec(), p.options().dtype(at::kBool));

    const int64_t pl_words  = p.size(-1);
    const int64_t row_bytes = pl_words * (PACK_WIDTH / 8);
    const int64_t batch     = p.numel() / pl_words;

    auto output = at::empty(logical_shape.vec(), p.options().dtype(at::kBool));
    const int chunks = (int)((ll + 31) / 32);
    const int64_t total_chunks = batch * (int64_t)chunks;
    constexpr int64_t PERSISTENT_THRESHOLD = 32 * 1024;
    if (total_chunks >= PERSISTENT_THRESHOLD) {
        kernels::k_unpack_bool_persistent<<<256, 256, 0, cur_stream()>>>(
            as_u8(p), reinterpret_cast<uint8_t*>(output.data_ptr<bool>()),
            ll, row_bytes, batch, chunks);
    } else {
        dim3 grid((unsigned)chunks, (unsigned)batch);
        kernels::k_unpack_bool_warp<<<grid, 32, 0, cur_stream()>>>(
            as_u8(p), reinterpret_cast<uint8_t*>(output.data_ptr<bool>()),
            ll, row_bytes);
    }
    return output;
}

//  xnor_popcount_matmul: bit1 GEMM.
//  Strategy:
//    1. CUTLASS B1 GEMM with `OpXorPopc` + fused K-2H epilogue on sm_80+ when
//       K_bits (= Kp * 64) ≥ 256. Pad M/N to 8 and K_bits to 256 with zeros.
//    2. Hand-tuned XOR-popc fallback on sm_70 / sm_75 or when K_bits < 256.
at::Tensor xnor_popcount_matmul(const at::Tensor& A, const at::Tensor& B, int64_t K) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "xnor_popcount_matmul: inputs must be 2-D");
    const int64_t M = A.size(0), N = B.size(0), Kp = A.size(1);
    TORCH_CHECK(Kp == B.size(1), "xnor_popcount_matmul: packed K mismatch");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    const int64_t Kp_bits = Kp * PACK_WIDTH;
    const int     cc      = compute_capability();

    if (cc >= 80 && Kp_bits >= 256) {
        constexpr int64_t M_ALIGN = 8;
        constexpr int64_t N_ALIGN = 8;
        constexpr int64_t K_BIT_ALIGN = 256;
        const int64_t M_pad      = (M + M_ALIGN - 1) / M_ALIGN * M_ALIGN;
        const int64_t N_pad      = (N + N_ALIGN - 1) / N_ALIGN * N_ALIGN;
        const int64_t Kp_bits_pad =
            (Kp_bits + K_BIT_ALIGN - 1) / K_BIT_ALIGN * K_BIT_ALIGN;
        const int64_t Kp32_pad   = Kp_bits_pad / 32;

        const bool needs_pad = (M_pad != M) || (N_pad != N)
                            || (Kp_bits_pad != Kp_bits);

        // Allocate padded uint32-viewed working tensors when padding is needed.
        auto opts_u32 = A.options().dtype(at::kInt);
        at::Tensor A_u32, B_u32;
        if (!needs_pad) {
            // Zero-copy reinterpret of the int64 packed buffer as int32 view.
            A_u32 = Ac.view(at::kInt);
            B_u32 = Bc.view(at::kInt);
        } else {
            A_u32 = at::zeros({M_pad, Kp32_pad}, opts_u32);
            B_u32 = at::zeros({N_pad, Kp32_pad}, opts_u32);
            const int64_t Kp32_in = Kp_bits / 32;
            cudaMemcpy2DAsync(
                A_u32.data_ptr(), Kp32_pad * sizeof(uint32_t),
                Ac.data_ptr(),    Kp32_in  * sizeof(uint32_t),
                Kp32_in * sizeof(uint32_t), M,
                cudaMemcpyDeviceToDevice, cur_stream());
            cudaMemcpy2DAsync(
                B_u32.data_ptr(), Kp32_pad * sizeof(uint32_t),
                Bc.data_ptr(),    Kp32_in  * sizeof(uint32_t),
                Kp32_in * sizeof(uint32_t), N,
                cudaMemcpyDeviceToDevice, cur_stream());
        }

        auto C_pad = at::empty({M_pad, N_pad}, opts_u32);

        using Gemm = kernels::CutlassB1XorGemm;
        Gemm gemm_op;
        typename Gemm::Arguments args{
            {(int)M_pad, (int)N_pad, (int)Kp_bits_pad},
            {reinterpret_cast<cutlass::uint1b_t const*>(A_u32.data_ptr()), (int)Kp_bits_pad},
            {reinterpret_cast<cutlass::uint1b_t const*>(B_u32.data_ptr()), (int)Kp_bits_pad},
            {C_pad.data_ptr<int32_t>(), (int)N_pad},
            {C_pad.data_ptr<int32_t>(), (int)N_pad},
            // Epilogue params — pass K_logical so K-2H is fused into the GEMM.
            {(int32_t)K},
        };

        cutlass::Status status = gemm_op(args, /*workspace=*/nullptr, cur_stream());
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "CUTLASS B1 GEMM failed: ", int(status));

        if (M_pad == M && N_pad == N) return C_pad;
        return C_pad.slice(0, 0, M).slice(1, 0, N).contiguous();
    }

    // Fallback: hand-tuned XOR-popc kernel for sm_70/sm_75 or tiny K.
    auto C = at::empty({M, N}, A.options().dtype(at::kInt));
    dim3 block(16, 16);
    dim3 grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16));
    kernels::k_xnor_popcount_matmul<<<grid, block, 0, cur_stream()>>>(
        as_u64(Ac), as_u64(Bc), C.data_ptr<int32_t>(), M, N, Kp, (int32_t)K);
    return C;
}

//  popcount — per-element int32, supports any integer dtype + bool.
at::Tensor popcount(const at::Tensor& x) {
    const auto p = x.contiguous();
    auto out = at::empty(p.sizes(), p.options().dtype(at::kInt));
    const int64_t n = p.numel();
    if (n == 0) return out;

    dim3 block(256);
    dim3 grid((unsigned)((n + 255) / 256));

    switch (p.scalar_type()) {
        case at::kByte:
        case at::kBool:
        case at::kChar:
            kernels::k_popcnt_u8<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint8_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        case at::kShort:
            kernels::k_popcnt_u16<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint16_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        case at::kInt:
            kernels::k_popcnt_u32<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint32_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        case at::kLong:
            kernels::k_popcnt_u64<<<grid, block, 0, cur_stream()>>>(
                reinterpret_cast<const uint64_t*>(p.data_ptr()),
                out.data_ptr<int32_t>(), n);
            break;
        default:
            TORCH_CHECK(false, "popcount: unsupported dtype ", p.scalar_type());
    }
    return out;
}

//  packed_popcount — total 1-bit count across the buffer (int64 scalar).
at::Tensor packed_popcount(const at::Tensor& x) {
    const auto p = x.contiguous();
    auto out = at::zeros({}, p.options().dtype(at::kLong));

    const int64_t n_bytes = (int64_t)p.nbytes();
    const int64_t n_words = n_bytes / (int64_t)sizeof(uint64_t);
    const int64_t tail    = n_bytes - n_words * (int64_t)sizeof(uint64_t);

    if (n_words > 0) {
        dim3 block(256);
        dim3 grid((unsigned)std::min<int64_t>((n_words + 255) / 256, 65535));
        kernels::k_packed_popcount_total<<<grid, block, 0, cur_stream()>>>(
            as_u64(p), n_words,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    if (tail > 0) {
        kernels::k_popcount_tail_byte<<<1, 1, 0, cur_stream()>>>(
            as_u8(p) + n_words * sizeof(uint64_t), (int)tail,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    return out;
}

//  hamming_distance — per-element popcount(A^B). Broadcasting via TensorIterator.
at::Tensor hamming_distance(const at::Tensor& A, const at::Tensor& B) {
    return popcount(at::bitwise_xor(A, B));
}

//  bit1_hamming_total — fused total popcount(A^B), int64 scalar.
at::Tensor bit1_hamming_total(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.sizes()       == B.sizes(),       "bit1_hamming_total: shape mismatch");
    TORCH_CHECK(A.scalar_type() == B.scalar_type(), "bit1_hamming_total: dtype mismatch");
    const auto Ac = A.contiguous();
    const auto Bc = B.contiguous();
    auto out = at::zeros({}, A.options().dtype(at::kLong));

    const int64_t n_bytes = (int64_t)Ac.nbytes();
    const int64_t n_words = n_bytes / (int64_t)sizeof(uint64_t);
    const int64_t tail    = n_bytes - n_words * (int64_t)sizeof(uint64_t);

    if (n_words > 0) {
        dim3 block(256);
        dim3 grid((unsigned)std::min<int64_t>((n_words + 255) / 256, 65535));
        kernels::k_hamming_total<<<grid, block, 0, cur_stream()>>>(
            as_u64(Ac), as_u64(Bc), n_words,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    if (tail > 0) {
        kernels::k_hamming_tail_byte<<<1, 1, 0, cur_stream()>>>(
            as_u8(Ac) + n_words * sizeof(uint64_t),
            as_u8(Bc) + n_words * sizeof(uint64_t),
            (int)tail,
            reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    }
    return out;
}

at::Tensor& randomize_bits(at::Tensor& out) { out.random_(); return out; }

}} // cbrute::cuda
#endif // HAVE_CUDA
