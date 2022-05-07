#pragma once

#include <third_party/acl/inc/acl/acl.h>
#include <map>
#include <string>
#include <vector>
#include "c10/macros/Export.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#define NpuSysStatus c10_npu::NpuSysCtrl::SysStatus

namespace c10_npu {

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
     static NpuSysCtrl& GetInstance();

    // GE Environment Initialize, return SysStatus
     SysStatus Initialize(int device_id = -1);

    // Change current device from pre_device to device
     SysStatus ExchangeDevice(int pre_device, int device);

    // Init backwards thread
     SysStatus BackwardsInit();

    // GE Environment Finalize, return SysStatus
     SysStatus Finalize();

    // Get Init_flag
     bool GetInitFlag();
private:
    NpuSysCtrl();

private:
    bool init_flag_;
    int device_id_;
};
} // namespace c10_npu

