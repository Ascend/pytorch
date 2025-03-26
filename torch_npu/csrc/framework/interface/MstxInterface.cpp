#include "torch_npu/csrc/framework/interface/MstxInterface.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"

namespace at_npu {
namespace native {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libms_tools_ext, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libms_tools_ext, funcName)


REGISTER_LIBRARY(libms_tools_ext)
LOAD_FUNCTION(mstxMarkA)
LOAD_FUNCTION(mstxRangeStartA)
LOAD_FUNCTION(mstxRangeEnd)
LOAD_FUNCTION(mstxDomainCreateA)
LOAD_FUNCTION(mstxDomainDestroy)
LOAD_FUNCTION(mstxDomainMarkA)
LOAD_FUNCTION(mstxDomainRangeStartA)
LOAD_FUNCTION(mstxDomainRangeEnd)
LOAD_FUNCTION(mstxMemHeapRegister)
LOAD_FUNCTION(mstxMemHeapUnregister)
LOAD_FUNCTION(mstxMemRegionsRegister)
LOAD_FUNCTION(mstxMemRegionsUnregister)

// save python range id with cann mstx range id.
// when mstx.range_end(id) is called, we can check if this id is invalid
static std::unordered_map<int, mstxRangeId> g_rangeIdMap;

static std::mutex g_mutex;

static bool IsSupportMstxFuncImpl()
{
    bool isSupport = false;
    char* path = std::getenv("ASCEND_HOME_PATH");
    if (path != nullptr) {
        std::string soPath = std::string(path) + "/lib64/libms_tools_ext.so";
        soPath = torch_npu::toolkit::profiler::Utils::RealPath(soPath);
        isSupport = !soPath.empty();
    }
    return isSupport;
}

static bool IsSupportMstxDomainFuncImpl()
{
    bool isSupport = (MstxDomainCreateA("test") == nullptr) ? false : true;
    return isSupport;
}

bool IsSupportMstxFunc()
{
    static bool isSupport = IsSupportMstxFuncImpl();
    return isSupport;
}

bool IsSupportMstxDomainFunc()
{
    static bool isSupport = IsSupportMstxDomainFuncImpl();
    return isSupport;
}

void MstxMarkA(const char* message, aclrtStream stream)
{
    using MstxMarkAFunc = void (*)(const char*, aclrtStream);
    static MstxMarkAFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxMarkAFunc)GET_FUNC(mstxMarkA);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxMarkA");
            noFuncFlag = true;
            return;
        }
    }
    func(message, stream);
}

int MstxRangeStartA(const char* message, aclrtStream stream, int ptRangeId)
{
    using MstxRangeStartAFunc = mstxRangeId (*)(const char*, aclrtStream);
    static MstxRangeStartAFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return 0;
    }
    if (func == nullptr) {
        func = (MstxRangeStartAFunc)GET_FUNC(mstxRangeStartA);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxRangeStartA");
            noFuncFlag = true;
            return 0;
        }
    }
    mstxRangeId taskId = func(message, stream);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_rangeIdMap.insert({ptRangeId, taskId});
    return 0;
}

void MstxRangeEnd(int ptRangeId)
{
    using MstxRangeEndFunc = void (*)(mstxRangeId);
    static MstxRangeEndFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxRangeEndFunc)GET_FUNC(mstxRangeEnd);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxRangeEnd");
            noFuncFlag = true;
            return;
        }
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto iter = g_rangeIdMap.find(ptRangeId);
    if (iter == g_rangeIdMap.end()) {
        ASCEND_LOGW("Failed to find mstx range id for python input range id %d", ptRangeId);
        return;
    }
    func(iter->second);
    g_rangeIdMap.erase(iter);
}

mstxDomainHandle_t MstxDomainCreateA(const char* name)
{
    using MstxDomainCreateAFunc = mstxDomainHandle_t (*)(const char*);
    static MstxDomainCreateAFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return nullptr;
    }
    if (func == nullptr) {
        func = (MstxDomainCreateAFunc)GET_FUNC(mstxDomainCreateA);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxDomainCreateA");
            noFuncFlag = true;
            return nullptr;
        }
    }
    return func(name);
}

void MstxDomainDestroy(mstxDomainHandle_t handle)
{
    using MstxDomainDestroyFunc = void (*)(mstxDomainHandle_t);
    static MstxDomainDestroyFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxDomainDestroyFunc)GET_FUNC(mstxDomainDestroy);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxDomainDestroy");
            noFuncFlag = true;
            return;
        }
    }
    func(handle);
}

void MstxDomainMarkA(mstxDomainHandle_t handle, const char* message, aclrtStream stream)
{
    using MstxDomainMarkAFunc = void (*)(mstxDomainHandle_t, const char*, aclrtStream);
    static MstxDomainMarkAFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxDomainMarkAFunc)GET_FUNC(mstxDomainMarkA);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxDomainMarkA");
            noFuncFlag = true;
            return;
        }
    }
    func(handle, message, stream);
}

int MstxDomainRangeStartA(mstxDomainHandle_t handle, const char* message, aclrtStream stream, int ptRangeId)
{
    using MstxDomainRangeStartAFunc = mstxRangeId (*)(mstxDomainHandle_t, const char*, aclrtStream);
    static MstxDomainRangeStartAFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return 0;
    }
    if (func == nullptr) {
        func = (MstxDomainRangeStartAFunc)GET_FUNC(mstxDomainRangeStartA);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxDomainRangeStartA");
            noFuncFlag = true;
            return 0;
        }
    }
    mstxRangeId taskId = func(handle, message, stream);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_rangeIdMap.insert({ptRangeId, taskId});
    return 0;
}

void MstxDomainRangeEnd(mstxDomainHandle_t handle, int ptRangeId)
{
    using MstxDomainRangeEndFunc = void (*)(mstxDomainHandle_t, mstxRangeId);
    static MstxDomainRangeEndFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxDomainRangeEndFunc)GET_FUNC(mstxDomainRangeEnd);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxDomainRangeEnd");
            noFuncFlag = true;
            return;
        }
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    auto iter = g_rangeIdMap.find(ptRangeId);
    if (iter == g_rangeIdMap.end()) {
        ASCEND_LOGW("Failed to find mstx range id for python input range id %d", ptRangeId);
        return;
    }
    func(handle, iter->second);
    g_rangeIdMap.erase(iter);
}

mstxMemHeapHandle_t MstxMemHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const* desc)
{
    using MstxMemHeapRegisterFunc = mstxMemHeapHandle_t (*)(mstxDomainHandle_t, mstxMemHeapDesc_t const*);
    static MstxMemHeapRegisterFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return nullptr;
    }
    if (func == nullptr) {
        func = (MstxMemHeapRegisterFunc)GET_FUNC(mstxMemHeapRegister);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxMemHeapRegister");
            noFuncFlag = true;
            return nullptr;
        }
    }
    return func(domain, desc);
}

void MstxMemHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    using MstxMemHeapUnregisterFunc = void (*)(mstxDomainHandle_t, mstxMemHeapHandle_t);
    static MstxMemHeapUnregisterFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxMemHeapUnregisterFunc)GET_FUNC(mstxMemHeapUnregister);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxMemHeapUnregister");
            noFuncFlag = true;
            return;
        }
    }
    func(domain, heap);
}

void MstxMemRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const* desc)
{
    using MstxMemRegionsRegisterFunc = void (*)(mstxDomainHandle_t, mstxMemRegionsRegisterBatch_t const*);
    static MstxMemRegionsRegisterFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxMemRegionsRegisterFunc)GET_FUNC(mstxMemRegionsRegister);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxMemRegionsRegister");
            noFuncFlag = true;
            return;
        }
    }
    func(domain, desc);
}

void MstxMemRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const* desc)
{
    using MstxMemRegionsUnregisterFunc = void (*)(mstxDomainHandle_t, mstxMemRegionsUnregisterBatch_t const*);
    static MstxMemRegionsUnregisterFunc func = nullptr;
    static bool noFuncFlag = false;
    if (noFuncFlag) {
        return;
    }
    if (func == nullptr) {
        func = (MstxMemRegionsUnregisterFunc)GET_FUNC(mstxMemRegionsUnregister);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxMemRegionsUnregister");
            noFuncFlag = true;
            return;
        }
    }
    func(domain, desc);
}

}
}