#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d::npushmem_extension {

void initialize_npushmem_with_store(c10::intrusive_ptr<c10d::Store> store, int rank, int world_size);

TORCH_API void npushmem_put(at::Tensor& tensor, int64_t peer);

TORCH_API void npushmem_get(at::Tensor& tensor, int64_t peer);

TORCH_API void npushmem_put_nbi(at::Tensor& tensor, int64_t peer);

TORCH_API void npushmem_get_nbi(at::Tensor& tensor, int64_t peer);

TORCH_API void npushmem_quiet(const at::Tensor& tensor);

// Deprecated nvshmem_* aliases: warn once per process, forward to npushmem_*.
TORCH_API void nvshmem_put(at::Tensor& tensor, int64_t peer);

TORCH_API void nvshmem_get(at::Tensor& tensor, int64_t peer);

} // namespace c10d::npushmem_extension
