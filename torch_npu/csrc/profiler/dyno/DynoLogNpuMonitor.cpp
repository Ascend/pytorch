#include "DynoLogNpuMonitor.h"
#include "utils.h"

namespace torch_npu {
namespace profiler {

bool DynoLogNpuMonitor::Init()
{
    if (isInitialized_) {
        ASCEND_LOGW("DynoLog npu monitor is initialized !");
        return true;
    }
    bool res = ipcClient_.RegisterInstance(npuId_);
    if (res) {
        isInitialized_ = true;
        ASCEND_LOGI("DynoLog npu monitor initialized success !");
    }
    return res;
}

std::string DynoLogNpuMonitor::Poll()
{
    std::string res = ipcClient_.IpcClientNpuConfig();
    if (res.empty()) {
        ASCEND_LOGI("Request for dynolog server is empty !");
        return "";
    }
    ASCEND_LOGI("Received NPU configuration successfully");
    return res;
}

} // namespace profiler
} // namespace torch_npu