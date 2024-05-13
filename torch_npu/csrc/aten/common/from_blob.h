#pragma once
#include <ATen/core/Tensor.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace at_npu {

namespace native {

class TensorMaker {
    friend TensorMaker for_blob(void* data, at::IntArrayRef sizes) noexcept;

public:

    TensorMaker& strides(at::OptionalIntArrayRef value) noexcept
    {
        strides_ = value;
        return *this;
    }

    TensorMaker& storage_offset(c10::optional<int64_t> value) noexcept
    {
        storage_offset_ = value;
        return *this;
    }

    TensorMaker& target_device(c10::optional<c10::Device> value) noexcept
    {
        device_ = value;
        return *this;
    }

    TensorMaker& options(at::TensorOptions value) noexcept
    {
        opts_ = value;
        return *this;
    }

    TensorMaker& allocator(c10::Allocator* allocator) noexcept
    {
        allocator_ = allocator;
        return *this;
    }

    at::Tensor make_tensor();

private:
    explicit TensorMaker(void* data, at::IntArrayRef sizes) noexcept
        : data_{data}, sizes_{sizes} {}

    std::size_t computeStorageSize() const noexcept;

    at::IntArrayRef makeTempSizes() const noexcept;

    void* data_;
    at::IntArrayRef sizes_;
    at::OptionalIntArrayRef strides_{};
    c10::optional<int64_t> storage_offset_{};
    c10::optional<c10::Device> device_{};
    at::TensorOptions opts_{};
    c10::Allocator* allocator_{};
};

inline TensorMaker for_blob(void* data, at::IntArrayRef sizes) noexcept
{
    return TensorMaker{data, sizes};
}

TORCH_NPU_API at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    int64_t storage_offset,
    const at::TensorOptions& options = {},
    const c10::optional<c10::Device> target_device = c10::nullopt);

TORCH_NPU_API at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const c10::optional<c10::Device> target_device = c10::nullopt,
    const at::TensorOptions& options = {});

TORCH_NPU_API at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options = {});

TORCH_NPU_API at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = {});
} // namespace native

}  // namespace at_npu
