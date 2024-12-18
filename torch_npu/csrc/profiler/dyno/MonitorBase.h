#pragma once
#include <string>
namespace torch_npu {
namespace profiler {
class MonitorBase {
public:
    virtual bool Init() = 0;
    virtual std::string Poll() = 0;
    virtual void SetNpuId(int id) = 0;
};
} // namespace profiler
} // namespace torch_npu
