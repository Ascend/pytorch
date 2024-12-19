#pragma once

#include <c10/core/Allocator.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace NPUWorkspaceAllocator {

c10::Allocator* get();
void init();
c10::DataPtr malloc_with_stream(size_t size, aclrtStream stream);
void emptyCache(int device, bool check_error = true);

} // namespace NPUWorkspaceAllocator
} // namespace c10_npu
