// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/libs/init_npu.h"
#include "torch_npu/csrc/framework/graph/execute/GraphExecutor.h"
#include "torch_npu/csrc/framework/graph/util/TdtChannelForPrint.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"


namespace torch_npu {

bool is_npu_device(const at::Device& device) {
  return device.type() == at_npu::key::NativeDeviceType;
}


void init_npu(const c10::DeviceIndex device_index) {
  c10_npu::NpuSysCtrl::SysStatus status =
      c10_npu::NpuSysCtrl::GetInstance().Initialize((int)device_index);
  if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
    C10_NPU_SHOW_ERR_MSG();
    return;
  }
}


void init_npu(const std::string& device_str) {
  auto device = at::Device(device_str);
  TORCH_CHECK(is_npu_device(device), "NPU device init fail, expected to get NPU device, but got ", device_str);
  init_npu(device.index());
}


void init_npu(const at::Device& device) {
  TORCH_CHECK(is_npu_device(device), "NPU device init fail, expected to get NPU device, but got ", str(device));
  init_npu(device.index());
}


void finalize_npu() {
  if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    try {
      c10_npu::npuSynchronizeDevice();
    } catch (std::exception& e) {
      TORCH_CHECK(false, "NPU SynchronizeDevice failed err=:%s", e.what());
    }
    at_npu::native::GraphExecutor::GetInstance().Finalize();
    at_npu::native::TdtChannelForPrint::GetInstance().Finalize();

    THNPUCachingHostAllocator_emptyCache();
    try {
      c10_npu::NPUCachingAllocator::emptyCache();
    } catch (std::exception& e) {
      TORCH_CHECK(false, "NPU CachingAllocator::emptyCache failed err=:%s", e.what());
    }

    c10_npu::NpuSysCtrl::SysStatus status = c10_npu::NpuSysCtrl::GetInstance().Finalize();
    if (status != c10_npu::NpuSysCtrl::SysStatus::FINALIZE_SUCC) {
      TORCH_CHECK(false, "NPU sys finalize failed.\n");
    }
  } else {
    TORCH_WARN("Please init npu device first!");
  }
}


c10::DeviceIndex current_device() {
  if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    int device;
    aclrtGetDevice(&device);
    return (c10::DeviceIndex)device;
  } else {
    TORCH_WARN("Please init npu device first!");
    return (c10::DeviceIndex)-1;
  }
}

} // namespace torch_npu
