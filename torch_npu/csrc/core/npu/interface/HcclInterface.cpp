#include "torch_npu/csrc/core/npu/interface/HcclInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace at_npu {
namespace hccl {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
    REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
    GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclGetCommName)
LOAD_FUNCTION(HcclCommResume)
LOAD_FUNCTION(HcclCommSetMemoryRange)
LOAD_FUNCTION(HcclCommUnsetMemoryRange)
LOAD_FUNCTION(HcclCommActivateCommMemory)
LOAD_FUNCTION(HcclCommDeactivateCommMemory)

extern HcclResult HcclGetCommNameFace(HcclComm commHandle, char* commName) {
    typedef HcclResult (*HcclGetCommNameFace)(HcclComm commHandle, char* commName);
    static HcclGetCommNameFace func = nullptr;
    if (func == nullptr)
    {
        func = (HcclGetCommNameFace)GET_FUNC(HcclGetCommName);
    }
    TORCH_CHECK(func, "Failed to find function HcclGetCommName,"
                " maybe you cann version is too low, please upgrade it",
                PTA_ERROR(ErrCode::NOT_FOUND));
    return func(commHandle, commName);
}

extern HcclResult HcclCommResumeFace(HcclComm comm)
{
    typedef HcclResult (*HcclCommResumeFace)(HcclComm comm);
    static HcclCommResumeFace func = nullptr;
    if (func == nullptr) {
        func = (HcclCommResumeFace)GET_FUNC(HcclCommResume);
    }
    TORCH_CHECK(func, "Failed to find function HcclCommResume,"
                " maybe you cann version is too low, please upgrade it", DIST_ERROR(ErrCode::NOT_FOUND));
    return func(comm);
}

extern bool isHcclFeatureSupported(HcclCommConfigCapability configParameter)
{
    typedef uint32_t(*HcclGetCommConfigCapabilityFunc)();
    static HcclGetCommConfigCapabilityFunc func = (HcclGetCommConfigCapabilityFunc) GET_FUNC(
            HcclGetCommConfigCapability);
    if (func == nullptr) {
        return false;
    }
    return configParameter < func();
}

HcclResult HcclCommSetMemoryRangeFace(HcclComm comm, void *virPtr, size_t size, size_t alignment, uint64_t flags)
{
    typedef HcclResult (*HcclCommSetMemoryRangeFace)(HcclComm comm, void *virPtr, size_t size, size_t alignment,
            uint64_t flags);
    static HcclCommSetMemoryRangeFace func = nullptr;
    if (func == nullptr) {
        func = (HcclCommSetMemoryRangeFace)GET_FUNC(HcclCommSetMemoryRange);
    }
    TORCH_CHECK(func, "Failed to find function HcclCommSetMemoryRange,"
                      " maybe you cann version is too low, please upgrade it", DIST_ERROR(ErrCode::NOT_FOUND));
    return func(comm, virPtr, size, alignment, flags);
}

HcclResult HcclCommUnsetMemoryRangeFace(HcclComm comm, void *virPtr)
{
    typedef HcclResult (*HcclCommUnsetMemoryRangeFace)(HcclComm comm, void *virPtr);
    static HcclCommUnsetMemoryRangeFace func = nullptr;
    if (func == nullptr) {
        func = (HcclCommUnsetMemoryRangeFace)GET_FUNC(HcclCommUnsetMemoryRange);
    }
    TORCH_CHECK(func, "Failed to find function HcclCommUnsetMemoryRange,"
                      " maybe you cann version is too low, please upgrade it", DIST_ERROR(ErrCode::NOT_FOUND));
    return func(comm, virPtr);
}

HcclResult HcclCommActivateCommMemoryFace(HcclComm comm, void *virPtr, size_t size, size_t offset,
                                          aclrtDrvMemHandle handle, uint64_t flags)
{
    typedef HcclResult (*HcclCommActivateCommMemoryFace)(HcclComm comm, void *virPtr, size_t size, size_t offset,
            aclrtDrvMemHandle handle, uint64_t flags);
    static HcclCommActivateCommMemoryFace func = nullptr;
    if (func == nullptr) {
        func = (HcclCommActivateCommMemoryFace)GET_FUNC(HcclCommActivateCommMemory);
    }
    TORCH_CHECK(func, "Failed to find function HcclCommActivateCommMemory,"
                      " maybe you cann version is too low, please upgrade it", DIST_ERROR(ErrCode::NOT_FOUND));
    return func(comm, virPtr, size, offset, handle, flags);
}

HcclResult HcclCommDeactivateCommMemoryFace(HcclComm comm, void *virPtr)
{
    typedef HcclResult (*HcclCommDeactivateCommMemoryFace)(HcclComm comm, void *virPtr);
    static HcclCommDeactivateCommMemoryFace func = nullptr;
    if (func == nullptr) {
        func = (HcclCommDeactivateCommMemoryFace)GET_FUNC(HcclCommDeactivateCommMemory);
    }
    TORCH_CHECK(func, "Failed to find function HcclCommDeactivateCommMemory,"
                      " maybe you cann version is too low, please upgrade it", DIST_ERROR(ErrCode::NOT_FOUND));
    return func(comm, virPtr);
}
} // namespace native
} // namespace at_npu
