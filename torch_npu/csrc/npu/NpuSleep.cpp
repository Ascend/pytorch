// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/npu/NpuSleep.h"

#include <dlfcn.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"
#include "torch_npu/csrc/utils/LazyInit.h"

// Forward declarations for types not available in the build environment's CANN
// headers
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclIntArray aclIntArray;
typedef int aclnnStatus;

namespace c10_npu {

// Function pointer types for aclnnSleep operator
using AclCreateIntArrayFunc = aclIntArray* (*)(const int64_t*, uint64_t);
using AclDestroyIntArrayFunc = aclError (*)(const aclIntArray*);
using AclnnSleepGetWorkspaceSizeFunc =
    aclnnStatus (*)(const aclIntArray*, uint64_t*, aclOpExecutor**);
using AclnnSleepFunc =
    aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

void npu_sleep(int64_t cycles) {
  TORCH_CHECK(
      cycles >= 0,
      "torch.npu._sleep(): expected non-negative cycles, got ",
      cycles);

  torch_npu::utils::npu_lazy_init();

  auto device = c10_npu::current_device();
  aclrtContext ctx = c10_npu::GetDeviceContext(device);
  TORCH_CHECK(ctx != nullptr, "Failed to get NPU context for device ", device);
  NPU_CHECK_ERROR(aclrtSetCurrentContext(ctx));

  static auto opapi_handle = dlopen("libopapi_nn.so", RTLD_NOW | RTLD_GLOBAL);
  TORCH_CHECK(
      opapi_handle != nullptr, "Failed to load libopapi_nn.so: ", dlerror());

  static auto createIntArray = reinterpret_cast<AclCreateIntArrayFunc>(
      dlsym(opapi_handle, "aclCreateIntArray"));
  static auto destroyIntArray = reinterpret_cast<AclDestroyIntArrayFunc>(
      dlsym(opapi_handle, "aclDestroyIntArray"));
  static auto sleepGetWorkspaceSize =
      reinterpret_cast<AclnnSleepGetWorkspaceSizeFunc>(
          dlsym(opapi_handle, "aclnnSleepGetWorkspaceSize"));
  static auto sleepExec =
      reinterpret_cast<AclnnSleepFunc>(dlsym(opapi_handle, "aclnnSleep"));

  TORCH_CHECK(
      createIntArray != nullptr && destroyIntArray != nullptr &&
          sleepGetWorkspaceSize != nullptr && sleepExec != nullptr,
      "aclnnSleep is not available in the current CANN version.");

  int64_t cycles_arr[] = {cycles};
  auto intArray = createIntArray(cycles_arr, 1);
  TORCH_CHECK(intArray != nullptr, "Failed to create aclIntArray");

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = sleepGetWorkspaceSize(intArray, &workspaceSize, &executor);
  NPU_CHECK_ERROR(ret);

  auto npu_stream = c10_npu::getCurrentNPUStream();
  auto acl_stream = npu_stream.stream(false);

  // Allocate workspace on NPU side via allocate_workspace (returns Tensor),
  // then record_stream to keep the storage alive until the stream completes.
  void* workspace = nullptr;
  at::Tensor workspace_tensor;
  if (workspaceSize > 0) {
    workspace_tensor =
        at_npu::native::allocate_workspace(workspaceSize, acl_stream);
    workspace = workspace_tensor.data_ptr();
    TORCH_CHECK(workspace != nullptr, "Failed to allocate NPU workspace");
    workspace_tensor.record_stream(npu_stream);
  }

  NPU_CHECK_ERROR(sleepExec(workspace, workspaceSize, executor, acl_stream));

  destroyIntArray(intArray);
}

} // namespace c10_npu
