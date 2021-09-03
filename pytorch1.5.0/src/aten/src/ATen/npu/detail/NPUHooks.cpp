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

#include <ATen/npu/detail/NPUHooks.h>

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/npu/NPUGenerator.h>

#include <ATen/detail/NPUHooksInterface.h>
#include <c10/npu/NPUException.h>
#include <c10/npu/NPUFunctions.h>
#include <c10/npu/sys_ctrl/npu_sys_ctrl.h>
#include <c10/util/Exception.h>

namespace at {
namespace npu {
namespace detail {

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
void NPUHooks::initNPU() const {
  C10_LOG_API_USAGE_ONCE("aten.init.npu");
  c10::npu::NpuSysCtrl::SysStatus status =
      c10::npu::NpuSysCtrl::GetInstance().Initialize();
  if (status != c10::npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
    NPU_LOGE("Npu init fail.");
  }
}

Generator* NPUHooks::getDefaultNPUGenerator(DeviceIndex device_index) const {
  return at::npu::detail::getDefaultNPUGenerator(device_index);
}

bool NPUHooks::hasNPU() const {
  return c10::npu::device_count() > 0;
}

int64_t NPUHooks::current_device() const {
  int device = 0;
  aclError err = aclrtGetDevice(&device);
  if (err == ACL_ERROR_NONE) {
    return device;
  }
  return -1;
}

Allocator* NPUHooks::getPinnedMemoryAllocator() const {
  initNPU();
  return getTHNPUCachingHostAllocator();
}

int NPUHooks::getNumNPUs() const {
  return c10::npu::device_count();
}

using at::NPUHooksRegistry;
using at::RegistererNPUHooksRegistry;

REGISTER_NPU_HOOKS(NPUHooks);

} // namespace detail
} // namespace npu
} // namespace at