#ifndef __TORCH_NPU_MSPROF_TX__
#define __TORCH_NPU_MSPROF_TX__

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/profiler/profiler_mgr.h"

namespace torch_npu {
namespace profiler {
typedef enum tagMarkResult {
    MARK_SUCCESS = 0          /**< success */
} markResult_t;

aclError profMark(const char *message, aclrtStream stream)
{
    aclError markRet = at_npu::native::AclProfilingMarkEx(message, strlen(message), stream);
    if (markRet == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
        ASCEND_LOGE("Failed to find function aclprofMarkEx");
        TORCH_WARN_ONCE("Profiling mark is unsupported, please upgrade CANN version");
        return MARK_SUCCESS;
    }
    return markRet;
}

void Mark(const char *message)
{
    if (!ProfilerMgr::GetInstance()->GetNpuTrace().load()) {
        return;
    }
    if (!ProfilerMgr::GetInstance()->GetMsprofTx().load()) {
        return;
    }
    RECORD_FUNCTION("mark_op", std::vector<c10::IValue>({}));
    c10::DeviceIndex device_id = -1;
    aclrtStream stream = c10_npu::getCurrentNPUStreamNoWait(device_id);
    auto mark_call = [message, stream]() -> int {
        return profMark(message, stream);
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("mstx_mark_op");
    cmd.SetCustomHandler(mark_call);
    cmd.Run();
}

} // profiler
} // torch_npu

#endif
