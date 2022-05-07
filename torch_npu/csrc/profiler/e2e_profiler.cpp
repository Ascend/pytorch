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


#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <third_party/acl/inc/acl/acl_prof.h>
#include <mutex>
#include "torch_npu/csrc/profiler/e2e_profiler.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"

namespace torch_npu {
namespace profiler {

aclprofConfig* local_profCfg = nullptr;

void checkProfilerRet(aclError ret, const char* message) {
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE(message);
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    return;
  }
}

void checkProfilerRet(aclError ret, const std::string message) {
  checkProfilerRet(ret, message.c_str());
}

void initMsPorf(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics) {
  // to init MsProf, there are 4 steps:
  // 1. create profile config, configure option,
  //    such as type of aicore metrics and
  //    which modules(ACL, TASK, AICORE, AICORE, L2CACHE) need profiling
  // 2. set msprof switch to be true and set profiling result path.
  // 3. create `stamp` used to record time info.
  // 4. configure the option of `stamp`.

  int deviceIndex = 0;
  aclError ret = aclrtGetDevice(&deviceIndex);
  if(ret){
    NPU_LOGE("In npu e2e profiling, aclrtGetDevice fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    return;
  }
  const uint32_t deviceNum = 1;
  uint32_t deviceIdList[deviceNum] = {deviceIndex};
    local_profCfg = at_npu::native::AclProfilingCreateConfig(
        deviceIdList,
        deviceNum,
        (aclprofAicoreMetrics)aicore_metrics,
        nullptr,
        npu_event);
  if (local_profCfg == nullptr) {
    NPU_LOGE("In npu e2e profiling, create_config fail, error profCfg is null.");
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    return;
  }
  c10_npu::npuSynchronizeDevice();
  ret  = at_npu::native::AclProfilingInit(dump_path.c_str(), dump_path.length());
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("In npu e2e profiling, AclProfilingInit failed.");
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    return;
  }
  ret = at_npu::native::AclProfilingStart(local_profCfg);
  if(ret){
    NPU_LOGE("In npu e2e profiling, AclProfStart fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    return;
  }
}

void pushStartTime(at::RecordFunction& fn) {
  auto local_stamp_ = at_npu::native::AclprofCreateStamp();
  if (local_stamp_  == nullptr) {
    NPU_LOGE("In npu e2e profiling, aclprofCreateStamp failed, created stamp is nullptr.");
    return;
  }
  auto ret = at_npu::native::AclprofSetStampTraceMessage(
      local_stamp_, fn.name().str(), strlen(fn.name().str()));
  checkProfilerRet(ret, "In npu e2e profiling, AclprofSetStampTraceMessage set failed.");
  uint32_t range_id_ = 0;
  ret = at_npu::native::AclprofRangeStart(local_stamp_, &range_id_);
  checkProfilerRet(ret, "In npu e2e profiling, AclprofRangeStart failed.");
  fn.setHandle((uint64_t)range_id_);
  fn.setForwardThreadId((uint64_t)local_stamp_);
}

void popEndTime(const at::RecordFunction& fn) {
  auto ret = at_npu::native::AclprofRangeStop((uint32_t)fn.handle());
  checkProfilerRet(ret, "In npu e2e profiling, AclprofRangeStop failed.");

  at_npu::native::AclprofDestroyStamp((void*)fn.forwardThreadId());
}

void init_e2e_profiler(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics) {

  initMsPorf(dump_path, npu_event, aicore_metrics);
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        torch_npu::profiler::pushStartTime(const_cast<at::RecordFunction&>(fn));
        return nullptr;
      },
      [](const at::RecordFunction& fn, at::ObserverContext*) {
        torch_npu::profiler::popEndTime(fn);
      }));
}

void finalize_e2e_profiler() {
  c10_npu::npuSynchronizeDevice();
  auto ret = at_npu::native::AclProfilingStop(local_profCfg);
  if (ret) {
    NPU_LOGE("In npu e2e profiling, AclProfStop fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
  at_npu::native::AclProfilingFinalize();
  at::clearThreadLocalCallbacks();
}

}
}
