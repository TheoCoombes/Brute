#pragma once

#include <vector>
#include <cstdint>
#include <torch/torch.h>
#include "dtype.h"

namespace cbrute {

// Thin wrapper for a 1-bit packed tensor.
// data_ is the actual packed storage (uint8 / int32 / int64 depending on pack_dtype).
// logical_shape_ is the element shape before packing (last dim = number of bits).
struct Bit1Tensor {
    at::Tensor             data_;
    std::vector<int64_t>   logical_shape_;
    PackDType              pack_dtype_;

    Bit1Tensor(at::Tensor data, std::vector<int64_t> logical_shape, PackDType pd)
        : data_(std::move(data))
        , logical_shape_(std::move(logical_shape))
        , pack_dtype_(pd) {}

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : logical_shape_) n *= s;
        return n;
    }

    int64_t pack_width() const { return cbrute::pack_width(pack_dtype_); }

    at::Device device()   const { return data_.device(); }
    const at::Tensor&           data()          const { return data_; }
    const std::vector<int64_t>& logical_shape() const { return logical_shape_; }
    PackDType                   pack_dtype()    const { return pack_dtype_; }
};

} // namespace cbrute
