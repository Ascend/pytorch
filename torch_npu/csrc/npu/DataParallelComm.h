#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace torch_npu {
namespace data_parallel {
TORCH_NPU_API void ReleaseHcclCommList();

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

std::vector<at::Tensor>& broadcast_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors);

std::vector<at::Tensor> broadcast(
    const at::Tensor& tensor,
    at::IntArrayRef devices);

tensor_list2d broadcast_coalesced(
    at::TensorList tensors,
    at::IntArrayRef devices,
    size_t buffer_size);

std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim = 0,
    const c10::optional<std::vector<c10::optional<c10_npu::NPUStream>>>& streams = c10::nullopt);

std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const c10::optional<std::vector<int64_t>>& chunk_sizes = c10::nullopt,
    int64_t dim = 0,
    const c10::optional<std::vector<c10::optional<c10_npu::NPUStream>>>& streams = c10::nullopt);

at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim);

at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    c10::optional<int32_t> destination_index);
}
} // namespace torch_npu
