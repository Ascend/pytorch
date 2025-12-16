#include <c10/core/Allocator.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include <third_party/acl/inc/acl/acl.h>

namespace at_npu {
namespace native {

// the host memory is not allocated by malloc
aclError process_unregistered_mem_location_type(c10_npu::NPUStream stream, aclrtMemcpyKind kind);

// the host memory is allocated by aclrtMallocHost or malloc and register
void process_host_mem_location_type(c10_npu::NPUStream stream, aclrtMemcpyKind kind, void* ptr);

// process non_blocking copy between host and device
void process_non_blocking_copy(void* ptr, void* currentPtr, c10_npu::NPUStream stream, aclrtMemcpyKind kind);

TORCH_NPU_API c10::Allocator* getCachingHostAllocator();

TORCH_NPU_API aclError CachingHostAllocator_recordEvent(void* ptr, aclrtMemcpyKind kind, c10_npu::NPUStream stream);

TORCH_NPU_API bool CachingHostAllocator_isPinned(void* ptr);
// Releases cached pinned memory allocations via npuHostFree
TORCH_NPU_API void CachingHostAllocator_emptyCache();

c10::Allocator* getPinnedMemoryAllocator();

} // namespace native
} // namespace at_npu
