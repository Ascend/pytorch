#ifndef __TORCH_NPU_MSPROF_TX__
#define __TORCH_NPU_MSPROF_TX__

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/profiler/npu_profiler.h"

namespace torch_npu {
namespace profiler {

void profMark(const char *message, aclrtStream stream)
{
    if (at_npu::native::IsSupportMstxFunc()) {
        at_npu::native::MstxMarkA(message, stream);
    } else {
        (void)at_npu::native::AclProfilingMarkEx(message, strlen(message), stream);
    }
}

void Mark(const char *message)
{
    if (!mstxEnable()) {
        return;
    }
    RECORD_FUNCTION("mark_op", std::vector<c10::IValue>({}));
    c10::DeviceIndex device_id = -1;
    aclrtStream stream = c10_npu::getCurrentNPUStreamNoWait(device_id);
    auto mark_call = [msg_ptr = std::make_shared<std::string>(message), stream]() -> int {
        (void)profMark(msg_ptr->c_str(), stream);
        return 0;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("mstx_mark_op");
    cmd.SetCustomHandler(mark_call);
    cmd.Run();
}

} // profiler
} // torch_npu

#endif
