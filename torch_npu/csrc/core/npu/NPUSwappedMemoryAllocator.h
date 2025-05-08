#pragma once

#include <c10/core/Allocator.h>

namespace c10_npu {
namespace NPUSwappedMemoryAllocator {

c10::Allocator* get();

TORCH_NPU_API void emptyCache();

} // namespace NPUSwappedMemoryAllocator
} // namespace c10_npu
