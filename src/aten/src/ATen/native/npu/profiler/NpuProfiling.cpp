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

#include "NpuProfiling.h"
#include <c10/npu/NPUStream.h>
#include <c10/npu/NPUException.h>

namespace at {
namespace native {
namespace npu {

NpuProfiling& NpuProfiling::Instance() {
  static NpuProfiling npuProfiling;
  return npuProfiling;
}

void NpuProfiling::Init(const std::string &path) {
  TORCH_CHECK(status == PROFILING_FINALIZE, "init current profile status is: ", status, " error!")
  auto ret = c10::npu::acl::AclProfilingInit(path.c_str(), path.length());
  if (ret) {
    NPU_LOGE("npu AclProfInit fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    return;
  }
  status = PROFILING_INIT;
}

void NpuProfiling::Start() {
  TORCH_CHECK(status == PROFILING_INIT || status == PROFILING_STOP, 
            "start current profile status is: ", status, " error!")
  int deviceIndex = 0;
  aclError ret = aclrtGetDevice(&deviceIndex);
  if(ret){
    NPU_LOGE("npu profiling aclrtGetDevice fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    (void)c10::npu::acl::AclProfilingFinalize();
    status = PROFILING_FINALIZE;
    return;
  }
  const uint32_t deviceNum = 1;
  uint32_t deviceIdList[deviceNum] = {deviceIndex};
  profCfg = c10::npu::acl::AclProfilingCreateConfig(
    deviceIdList,
    deviceNum,
    ACL_AICORE_ARITHMETIC_UTILIZATION,
    nullptr,
    ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU);
  if (profCfg == nullptr) {
    NPU_LOGE("npu profiling profiling_create_config fail, error  profCfg is null.");
    C10_NPU_SHOW_ERR_MSG();
    (void)c10::npu::acl::AclProfilingFinalize();
    status = PROFILING_FINALIZE;
    return;
  }
  ret = c10::npu::acl::AclProfilingStart(profCfg);
  if(ret){
    NPU_LOGE("npu profiling AclProfStart fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  status = PROFILING_START;
}

void NpuProfiling::Stop() {
  TORCH_CHECK(status == PROFILING_START, "stop current profile status is: ", status, " error!")
  auto ret = c10::npu::acl::AclProfilingStop(profCfg);
  if (ret) {
    NPU_LOGE("npu AclProfStop fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  status = PROFILING_STOP;
}

void NpuProfiling::Finalize() {
  if (profCfg != nullptr) {
    if (status != PROFILING_STOP) {
      NPU_LOGW("finalize current profile status ( %u ) is not stopped, and call stop now.", status);
      auto ret = c10::npu::acl::AclProfilingStop(profCfg);
      if (ret) {
        NPU_LOGE("npu AclProfStop fail, error code: %d", ret);
        C10_NPU_SHOW_ERR_MSG();
      }
    }
    auto ret = c10::npu::acl::AclProfilingDestroyConfig(profCfg);
    if (ret) {
      NPU_LOGE("npu AclProfDestoryConfig fail, error code: %d", ret);
      C10_NPU_SHOW_ERR_MSG();
    }
    profCfg = nullptr;
  }
  auto ret = c10::npu::acl::AclProfilingFinalize();
  if (ret) {
    NPU_LOGE("npu AclProfFinalize fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  status = PROFILING_FINALIZE;
}

}
}
}
