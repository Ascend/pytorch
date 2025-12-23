#include <unistd.h>
#include <c10/util/flat_hash_map.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/NPUSwappedMemoryAllocator.h"

size_t kAlignSize = 4096; // The first address must be aligned to page_size

struct HostPtr {
    void* ptr;
    void* alignedPtr;
};

ska::flat_hash_map<void*, HostPtr> memBlocks;
bool initialized = false;

// malloc host memopy
void* mallocHostMemory(size_t size, bool& is_support_consistency)
{
    void* ptr = nullptr;
    if (c10_npu::acl::AclrtMallocHostWithCfgExist()) {
        aclrtMallocAttrValue attrValue;
        attrValue.vaFlag = 1;

        aclrtMallocAttribute attributes[1];
        attributes[0].attr = ACL_RT_MEM_ATTR_VA_FLAG;
        attributes[0].value = attrValue;

        aclrtMallocConfig cfg;
        cfg.numAttrs = 1;
        cfg.attrs = attributes;

        aclError mallocError = c10_npu::acl::AclrtMallocHostWithCfg(static_cast<void**>(&ptr), size, &cfg);
        // if feature not support, then fall back to the old logic
        if (ACL_ERROR_RT_FEATURE_NOT_SUPPORT == mallocError) {
            is_support_consistency = false;
            ASCEND_LOGD("The current version of driver does not support the VA address normalization feature.");
            NPU_CHECK_ERROR(aclrtMallocHost(static_cast<void**>(&ptr), size));
        } else {
            is_support_consistency = true;
            NPU_CHECK_ERROR(mallocError);
        }
    } else {
        ASCEND_LOGD("The current version of driver and runtime does not support the VA address normalization feature.");
        NPU_CHECK_ERROR(aclrtMallocHost(static_cast<void**>(&ptr), size));
    }
    return ptr;
}

// register host memopy to device
void* registerSvmMem(void* ptr, size_t size, bool is_support_consistency)
{
    void *svmPtr = nullptr;
    aclrtHostRegisterType regType = ACL_HOST_REGISTER_MAPPED;
    void* alignedPtr = nullptr;
    if (c10_npu::acl::AclrtMallocHostWithCfgExist() && is_support_consistency) {
        alignedPtr = ptr;
    } else {
        uintptr_t aligned_ptr = (reinterpret_cast<uintptr_t>(ptr) + kAlignSize - 1) / kAlignSize * kAlignSize;
        alignedPtr = reinterpret_cast<void*>(aligned_ptr);
    }
    if (c10_npu::acl::AclrtHostRegister(alignedPtr, size, regType, &svmPtr) != ACL_ERROR_NONE) {
        NPU_CHECK_ERROR(aclrtFreeHost(ptr));
        TORCH_CHECK(false, "AclrtHostRegister failed.", PTA_ERROR(ErrCode::ACL));
    }
    if (alignedPtr != svmPtr) {
        ASCEND_LOGW("The svmPtr(0x%llx) is not equel to alignedPtr(0x%llx), then the memory pointed by svmPtr can not be printed directly on host ", svmPtr, alignedPtr)
    }

    HostPtr hostPtr;
    hostPtr.ptr = ptr;
    hostPtr.alignedPtr = alignedPtr;
    memBlocks.emplace(svmPtr, hostPtr);
    return svmPtr;
}

// malloc swap memopy
void* mallocHostSwapMemory(size_t size)
{
    if (!initialized) {
        kAlignSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        initialized = true;
    }
    size = (size + kAlignSize - 1) & ~(kAlignSize - 1);
    bool is_support_consistency = false;
    void *ptr = mallocHostMemory(size, is_support_consistency);
    void *svmPtr = registerSvmMem(ptr, size, is_support_consistency);
    return svmPtr;
}

static void svm_deleter(void* ptr)
{
}

namespace c10_npu {
namespace NPUSwappedMemoryAllocator {

class NpuSwappedMemoryAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t size) override
    {
        int device = 0;
        NPU_CHECK_ERROR(c10_npu::GetDevice(&device));

        void* dev_ptr = mallocHostSwapMemory(size);
        void (*delete_func)(void*) = &svm_deleter;
        return {dev_ptr, dev_ptr, delete_func, c10::Device(c10::DeviceType::PrivateUse1, device)};
    }

    c10::DeleterFnPtr raw_deleter() const override
    {
        return &svm_deleter;
    }

    // Note [COW/lazy_clone is not supported yet]
    void copy_data(void* dest, const void* src, std::size_t count) const final
    {
        default_copy_data(dest, src, count);
    }
}; // class NpuSwappedMemoryAllocator

NpuSwappedMemoryAllocator swapmemory_allocator;

c10::Allocator* get()
{
    return &swapmemory_allocator;
}

void emptyCache()
{
    for (auto it = memBlocks.begin(); it != memBlocks.end(); it++) {
        NPU_CHECK_ERROR(c10_npu::acl::AclrtHostUnregister(it->second.alignedPtr));
        NPU_CHECK_ERROR(aclrtFreeHost(it->second.ptr));
    }
    memBlocks.clear();
}

} // namespace NPUSwappedMemoryAllocator
} // namespace c10_npu
