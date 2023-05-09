#include <c10/core/Allocator.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <c10/util/Exception.h>
#include <third_party/acl/inc/acl/acl.h>
#include <c10/util/SmallVector.h>

c10::Allocator* getTHNPUCachingHostAllocator(void);

aclError THNPUCachingHostAllocator_recordEvent(void* ptr, c10_npu::NPUStream stream);

void THNPUCachingHostAllocator_insertCompleteEvent(aclrtEvent event);

bool THNPUCachingHostAllocator_isPinndPtr(void* ptr);
// Releases cached pinned memory allocations via npuHostFree
void THNPUCachingHostAllocator_emptyCache(void);

c10::Allocator* getPinnedMemoryAllocator(void);
