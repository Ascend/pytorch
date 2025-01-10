#pragma once
#include <torch_npu/csrc/toolkit/profiler/common/singleton.h>
#include "MonitorBase.h"
#include "NpuIpcClient.h"

namespace torch_npu {
namespace profiler {

class DynoLogNpuMonitor : public MonitorBase, public torch_npu::toolkit::profiler::Singleton<DynoLogNpuMonitor> {
    friend class torch_npu::toolkit::profiler::Singleton<DynoLogNpuMonitor>;

public:
    DynoLogNpuMonitor() = default;
    bool Init() override;
    std::string Poll() override;
    void SetNpuId(int id) override
    {
        npuId_ = id;
    }

private:
    bool isInitialized_ = false;
    int32_t npuId_ = 0;
    IpcClient ipcClient_;
};

} // namespace profiler
} // namespace torch_npu
