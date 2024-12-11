#include <c10/core/Allocator.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include <third_party/acl/inc/acl/acl.h>

namespace at_npu {
namespace native {

TORCH_NPU_API c10::Allocator* getCachingHostAllocator();

TORCH_NPU_API aclError CachingHostAllocator_recordEvent(void* ptr, c10_npu::NPUStream stream);

TORCH_NPU_API bool CachingHostAllocator_isPinned(void* ptr);
// Releases cached pinned memory allocations via npuHostFree
TORCH_NPU_API void CachingHostAllocator_emptyCache();

c10::Allocator* getPinnedMemoryAllocator();

} // namespace native
} // namespace at_npu
