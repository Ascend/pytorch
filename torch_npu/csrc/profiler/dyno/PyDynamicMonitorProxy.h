#pragma once
#include "MonitorBase.h"
#include "DynoLogNpuMonitor.h"
namespace torch_npu {
namespace profiler {
class PyDynamicMonitorProxy {
public:
    PyDynamicMonitorProxy() = default;
    bool InitDyno(int npuId)
    {
        try {
            monitor_ = DynoLogNpuMonitor::GetInstance();
            monitor_->SetNpuId(npuId);
            bool res = monitor_->Init();
            return res;
        } catch (const std::exception &e) {
            ASCEND_LOGE("Error when init dyno %s !", e.what());
            return false;
        }
    }
    std::string PollDyno()
    {
        return monitor_->Poll();
    };

private:
    MonitorBase *monitor_ = nullptr;
};
} // namespace profiler
} // namespace torch_npu
