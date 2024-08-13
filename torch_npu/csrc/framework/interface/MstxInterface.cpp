#include <unordered_set>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/interface/MstxInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace at_npu {
namespace native {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libms_tools_ext, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libms_tools_ext, funcName)


REGISTER_LIBRARY(libms_tools_ext)
LOAD_FUNCTION(mstxRangeStartA)
LOAD_FUNCTION(mstxRangeEnd)

// save python range id with cann mstx range id.
// when mstx.range_end(id) is called, we can check if this id is invalid
static std::unordered_map<int, mstxRangeId> g_rangeIdMap;

// save python range id with stream.
// when mstx.range_end(id) is called, we cann know if add range end event on pure host or to device aswell.
static std::unordered_set<int> g_rangeIdsWithStream;

static std::mutex g_mutex;

int MstxRangeStartA(const char* message, aclrtStream stream, int ptRangeId)
{
    using MstxRangeStartAFunc = mstxRangeId (*)(const char*, aclrtStream);
    static MstxRangeStartAFunc func = nullptr;
    static bool flag = false;
    if (func == nullptr) {
        if (flag) {
            return 0;
        }
        func = (MstxRangeStartAFunc)GET_FUNC(mstxRangeStartA);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxRangeStartA");
            flag = true;
            return 0;
        }
    }
    mstxRangeId taskId = func(message, stream);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_rangeIdMap.insert({ptRangeId, taskId});
    if (stream) {
        g_rangeIdsWithStream.insert(ptRangeId);
    }
    return 0;
}

void MstxRangeEnd(int ptRangdId)
{
    using MstxRangeEndFunc = void (*)(mstxRangeId);
    static MstxRangeEndFunc func = nullptr;
    static bool flag = false;
    if (func == nullptr) {
        if (flag) {
            return;
        }
        func = (MstxRangeEndFunc)GET_FUNC(mstxRangeEnd);
        if (func == nullptr) {
            ASCEND_LOGW("Failed to get func mstxRangeEnd");
            flag = true;
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

bool IsRangeIdWithStream(int ptRangeId)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    auto iter = g_rangeIdsWithStream.find(ptRangeId);
    if (iter == g_rangeIdsWithStream.end()) {
        return false;
    }
    g_rangeIdsWithStream.erase(iter);
    return true;
}
}
}