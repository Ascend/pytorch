#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_rt.h>

#include <c10/core/DeviceGuard.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Utils.h>

#include <ATen/core/CachingHostAllocator.h>
#include <c10/util/Deprecated.h>

#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

namespace at_npu::native {

bool ptr_exist(void* ptr);

// the host memory is not allocated by malloc
aclError process_unregistered_mem_location_type(c10_npu::NPUStream stream, aclrtMemcpyKind kind);

// the host memory is allocated by aclrtMallocHost or malloc and register
void process_host_mem_location_type(const c10::Storage& storage, c10_npu::NPUStream stream);

// process non_blocking copy between host and device
void process_non_blocking_copy(const c10::Storage& storage, void *currentPtr, c10_npu::NPUStream stream, aclrtMemcpyKind kind);

inline TORCH_NPU_API c10::Allocator* getCachingHostAllocator() {
    return at::getHostAllocator(at::kPrivateUse1);
}

inline TORCH_NPU_API bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, c10_npu::NPUStream stream) {
    return at::getHostAllocator(at::kPrivateUse1)->record_event(ptr, ctx, stream.unwrap());
}

// Releases cached pinned memory allocations via npuHostFree
inline TORCH_NPU_API void CachingHostAllocator_emptyCache() {
    return at::getHostAllocator(at::kPrivateUse1)->empty_cache();
}

inline TORCH_NPU_API bool CachingHostAllocator_isPinned(void* ptr) {
    if (c10_npu::acl::AclrtPointerGetAttributesExist()) {
        if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
            return false;
        }
        if (c10_npu::GetLocalDevice() < 0) {
            c10_npu::SetCurrentDevice();
        }
        aclrtPtrAttributes attributes;
        NPU_CHECK_ERROR(c10_npu::acl::AclrtPointerGetAttributes(ptr, &attributes), "aclrtPointerGetAttributes");
        return ACL_MEM_LOCATION_TYPE_HOST == attributes.location.type;
    }
    return at_npu::native::ptr_exist(ptr);
}

inline at::DataPtr HostAlloc(size_t size)
{
    return at::getHostAllocator(at::kPrivateUse1)->allocate(size);
}

c10::Allocator* getPinnedMemoryAllocator();

} // namespace at_npu::native
