#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/interface/MstxInterface.h"
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

// save python range id with cann mstx range id.
// when mstx.range_end(id) is called, we can check if this id is invalid
static std::unordered_map<int, mstxRangeId> g_rangeIdMap;

static std::mutex g_mutex;

static std::mutex g_supportMstx;

bool IsSupportMstxFunc()
{
    static bool isSupport = false;
    static bool isChecked = false;
    std::lock_guard<std::mutex> lock(g_supportMstx);
    if (!isChecked) {
        char* path = std::getenv("ASCEND_HOME_PATH");
        if (path != nullptr) {
            std::string soPath = std::string(path) + "/lib64/libms_tools_ext.so";
            soPath = torch_npu::toolkit::profiler::Utils::RealPath(soPath);
            isSupport = !soPath.empty();
            isChecked = true;
        }
    }
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

void MstxRangeEnd(int ptRangdId)
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
    auto iter = g_rangeIdMap.find(ptRangdId);
    if (iter == g_rangeIdMap.end()) {
        ASCEND_LOGW("Failed to find mstx range id for python input range id %d", ptRangdId);
        return;
    }
    func(iter->second);
    g_rangeIdMap.erase(iter);
}

}
}