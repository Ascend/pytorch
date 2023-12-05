#pragma once

#include <third_party/acl/inc/acl/acl.h>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include <functional>
#include "c10/macros/Export.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#define NpuSysStatus c10_npu::NpuSysCtrl::SysStatus

namespace c10_npu {
using ReleaseFn = std::function<void()>;

enum class ReleasePriority : uint8_t {
    PriorityFirst = 0,
    PriorityMiddle = 5,
    PriorityLast = 10
};

class NpuSysCtrl {
public:
    ~NpuSysCtrl() = default;

    enum SysStatus {
        INIT_SUCC = 0,
        INIT_ALREADY,
        INIT_FAILED,
        CREATE_SESS_SUCC,
        CREATE_SESS_FAILED,
        ADD_GRAPH_SUCC,
        ADD_GRAPH_FAILED,
        RUN_GRAPH_SUCC,
        RUN_GRAPH_FAILED,
        FINALIZE_SUCC,
        FINALIZE_FAILED,
    };

    // Get NpuSysCtrl singleton instance
     C10_NPU_API static NpuSysCtrl& GetInstance();

    // GE Environment Initialize, return SysStatus
     SysStatus Initialize(int device_id = -1);

    // Change current device from pre_device to device
     SysStatus ExchangeDevice(int pre_device, int device);

    // Init backwards thread
     C10_NPU_API SysStatus BackwardsInit();

    // Set overflow switch
     SysStatus OverflowSwitchEnable();

    // GE Environment Finalize, return SysStatus
     C10_NPU_API SysStatus Finalize();

    // Get Init_flag
     C10_NPU_API bool GetInitFlag();

    aclrtContext InitializedContext(int device_index)
    {
        if (GetInitFlag()) {
            return ctx_[device_index];
        }
        TORCH_CHECK(false, "no npu device context has been initialized!");
        return nullptr;
    }

    // Register fn to be called during stage of exit and
    // the callability of fn is guaranteed by the caller.
     void RegisterReleaseFn(ReleaseFn release_fn,
         ReleasePriority priority = ReleasePriority::PriorityMiddle);
private:
    NpuSysCtrl();

private:
    bool init_flag_;
    bool is_soc_match;
    std::string soc_name_;
    int device_id_;
    uint32_t device_count_;
    aclrtContext ctx_[C10_COMPILE_TIME_MAX_NPUS] = {nullptr};
    std::map<ReleasePriority, std::vector<ReleaseFn>> release_fn_;
    std::unordered_set<int> used_devices;
};
} // namespace c10_npu

