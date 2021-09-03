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

#include "npu_sys_ctrl.h"
#include <c10/npu/npu_log.h>
#include <c10/npu/NPUStream.h>
#include <c10/npu/OptionsManager.h>
#include <third_party/acl/inc/acl/acl_op_compiler.h>

namespace c10 {
namespace npu {

NpuSysCtrl::NpuSysCtrl() : init_flag_(false), device_id_(0) {}

// Get NpuSysCtrl singleton instance
C10_API NpuSysCtrl& NpuSysCtrl::GetInstance() {
  static NpuSysCtrl instance;
  return instance;
}

// GE Environment Initialize, return Status: SUCCESS, FAILED
C10_API NpuSysCtrl::SysStatus NpuSysCtrl::Initialize(int device_id) {

    if (init_flag_) {
        return INIT_SUCC;
    }
    C10_NPU_CHECK(aclInit(nullptr));

    if (c10::npu::OptionsManager::CheckAclDumpDateEnable()){
        C10_NPU_CHECK(aclmdlInitDump());
        NPU_LOGD("dump init success");
    }

    auto ret = aclrtGetDevice(&device_id_);
    if (ret != ACL_ERROR_NONE) {
        device_id_ = (device_id == -1) ? 0 : device_id;
        C10_NPU_CHECK(aclrtSetDevice(device_id_));
    }else{
        NPU_LOGE("Npu device %d has been set before global init.", device_id_);
    }

    init_flag_ = true;
    NPU_LOGD("Npu sys ctrl initialize successfully.");

    if (c10::npu::OptionsManager::CheckAclDumpDateEnable()) {
      const char *aclConfigPath = "acl.json";
      C10_NPU_CHECK(aclmdlSetDump(aclConfigPath));
      NPU_LOGD("set dump config success");  
    }

    return INIT_SUCC;
}

C10_API NpuSysCtrl::SysStatus NpuSysCtrl::ExchangeDevice(int pre_device, int device) {
    C10_NPU_CHECK(aclrtResetDevice(pre_device));
    C10_NPU_CHECK(aclrtSetDevice(device));
    device_id_= device;
    return INIT_SUCC;
}

C10_API NpuSysCtrl::SysStatus NpuSysCtrl::BackwardsInit() {
    C10_NPU_CHECK(aclrtSetDevice(device_id_));
    return INIT_SUCC;
}

// GE Environment Finalize, return SysStatus
C10_API NpuSysCtrl::SysStatus NpuSysCtrl::Finalize() {
    if (!init_flag_) {
        return FINALIZE_SUCC;
    }
    c10::npu::NPUEventManager::GetInstance().ClearEvent();
    auto stream = c10::npu::getCurrentNPUStream();
    (void)aclrtDestroyStream(stream);
    C10_NPU_CHECK(aclrtResetDevice(device_id_));
    C10_NPU_CHECK(aclFinalize());
    init_flag_ = false;

    if (c10::npu::OptionsManager::CheckAclDumpDateEnable()) {
        C10_NPU_CHECK(aclmdlFinalizeDump());
    }

    NPU_LOGD("Npu sys ctrl finalize successfully.");
    return FINALIZE_SUCC;
}

C10_API bool NpuSysCtrl::GetInitFlag() {
    return init_flag_;
}

} // namespace npu
} // namespace c10
