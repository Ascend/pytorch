// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __C10_NPU_SYS_CTRL___
#define __C10_NPU_SYS_CTRL___

#include <third_party/acl/inc/acl/acl.h>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include "c10/macros/Export.h"
#include <c10/npu/NPUEventManager.h>
#define NpuSysStatus c10::npu::NpuSysCtrl::SysStatus

namespace c10 {
namespace npu {
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
    C10_API static NpuSysCtrl& GetInstance();

    // GE Environment Initialize, return SysStatus
    C10_API SysStatus Initialize(int device_id = -1);
    
    // Change current device from pre_device to device 
    C10_API SysStatus ExchangeDevice(int pre_device, int device);
  
    // Init backwards thread
    C10_API SysStatus BackwardsInit();

    // GE Environment Finalize, return SysStatus
    C10_API SysStatus Finalize();
    
    // Get Init_flag
    C10_API bool GetInitFlag();

    // Register fn to be called during stage of exit and
    // the callability of fn is guaranteed by the caller.
    C10_API void RegisterReleaseFn(ReleaseFn release_fn,
        ReleasePriority priority = ReleasePriority::PriorityMiddle);

private:
    NpuSysCtrl();

private:
    bool init_flag_;
    int device_id_;
    std::map<ReleasePriority, std::vector<ReleaseFn>> release_fn_;
};

} // namespace npu
} // namespace c10

#endif
