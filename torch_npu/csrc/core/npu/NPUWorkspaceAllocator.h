#pragma once

#include <c10/core/Allocator.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace NPUWorkspaceAllocator {

c10::Allocator* get();
void init();
c10::DataPtr malloc_with_stream(size_t size, aclrtStream stream);
void emptyCache(int device, bool need_empty_queue, bool check_error = true);

} // namespace NPUWorkspaceAllocator
} // namespace c10_npu

namespace at_npu {
namespace native {

TORCH_NPU_API at::Tensor allocate_workspace(uint64_t workspace_size, aclrtStream stream);

} // namespace native
} // namespace at_npu
