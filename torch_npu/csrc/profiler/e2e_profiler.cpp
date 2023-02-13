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

std::atomic<bool> global_enable_profiling(false);

namespace torch_npu {
namespace profiler {

aclprofConfig* local_profCfg = nullptr;
bool global_call_stack = false;

std::vector<FileLineFunc> prepareCallstack(const std::vector<torch::jit::StackEntry> &cs) {
  std::vector<FileLineFunc> entries;
  entries.reserve(cs.size());
  for (const auto& entry : cs) {
    auto& range = entry.range;
    if (range.source()) {
      auto& src = range.source();
      if (src && src->filename()) {
        auto line = src->starting_line_no() + src->lineno_for_offset(range.start());
        entries.emplace_back(FileLineFunc{*(src->filename()), line, entry.filename});
      }
    }
  }
  return entries;
}

std::vector<std::string> callstack2Str(const std::vector<FileLineFunc> &cs) {
  std::vector<std::string> cs_str;
  cs_str.reserve(cs.size());
  for (const auto& entry : cs) {
    std::stringstream loc;
    loc << entry.filename << "(" << entry.line << "):" << entry.funcname;
    cs_str.push_back(loc.str());
  }
  return cs_str;
}

void CheckProfilerRet(aclError ret, const char* message) {
  static bool checkOnce = false;
  if (ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
    if (!checkOnce) {
      checkOnce = true;
      NPU_LOGW("%s", message);
    }
    return;
  }
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("%s", message);
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    return;
  }
}

void CheckProfilerRet(aclError ret, const std::string message) {
  CheckProfilerRet(ret, message.c_str());
}

void InitMsPorf(const std::string dump_path, uint64_t npu_event,
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

void PushStartTime(at::RecordFunction& fn) {
  auto local_stamp_ = at_npu::native::AclprofCreateStamp();
  if (local_stamp_  == nullptr) {
    NPU_LOGE("In npu e2e profiling, aclprofCreateStamp failed, created stamp is nullptr.");
    return;
  }
  static const std::string tag_name = "torch_op";
  auto ret = at_npu::native::AclprofSetStampTagName(local_stamp_, tag_name.c_str(), tag_name.size());
  CheckProfilerRet(ret, "In npu e2e profiling, AclprofSetStampTagName set failed.");

  ret = at_npu::native::AclprofSetStampTraceMessage(
      local_stamp_, fn.name().str(), strlen(fn.name().str()));
  CheckProfilerRet(ret, "In npu e2e profiling, AclprofSetStampTraceMessage set failed.");
  if (global_call_stack) {
    std::vector<std::string> py_stack;
    if (fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
      auto cs = prepareCallstack(torch::jit::currentCallstack());
      if (cs.empty()) {
        cs = prepareCallstack(torch::jit::tracer::pythonCallstack());
      }
      py_stack = callstack2Str(cs);
    }
    std::string call_stack_data;
    for (size_t i = 0; i < py_stack.size(); i++) {
      call_stack_data += py_stack[i];
      call_stack_data += ((i == py_stack.size() - 1) ? "" : ";");
    }
    if (!call_stack_data.empty()) {
      ret = at_npu::native::AclprofSetStampCallStack(local_stamp_, call_stack_data.c_str(), call_stack_data.size());
      CheckProfilerRet(ret, "In npu e2e profiling, AclprofSetStampCallStack set warning."
        " Try to install the matching Ascend Profiler.");
    }
  }
  uint32_t range_id_ = 0;
  ret = at_npu::native::AclprofRangeStart(local_stamp_, &range_id_);
  CheckProfilerRet(ret, "In npu e2e profiling, AclprofRangeStart failed.");
  fn.setHandle((uint64_t)range_id_);
  fn.setForwardThreadId((uint64_t)local_stamp_);
}

void PopEndTime(const at::RecordFunction& fn) {
  auto ret = at_npu::native::AclprofRangeStop((uint32_t)fn.handle());
  CheckProfilerRet(ret, "In npu e2e profiling, AclprofRangeStop failed.");

  at_npu::native::AclprofDestroyStamp((void*)fn.forwardThreadId());
}

void InitE2eProfiler(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics, bool call_stack) {
  global_enable_profiling.store(true);
  global_call_stack = call_stack;
  InitMsPorf(dump_path, npu_event, aicore_metrics);
  auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        torch_npu::profiler::PushStartTime(const_cast<at::RecordFunction&>(fn));
        return nullptr;
      },
      [](const at::RecordFunction& fn, at::ObserverContext*) {
        torch_npu::profiler::PopEndTime(fn);
      }));
}

void FinalizeE2eProfiler() {
  global_enable_profiling.store(false);
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
