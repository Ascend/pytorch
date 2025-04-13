#pragma once

#include <c10/core/Allocator.h>

namespace c10_npu {
namespace NPUSwapMemoryAllocator {

c10::Allocator* get();

TORCH_NPU_API void emptyCache();

} // namespace NPUSwapMemoryAllocator
} // namespace c10_npu
