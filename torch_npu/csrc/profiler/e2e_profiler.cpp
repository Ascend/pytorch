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


#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>
#include <third_party/acl/inc/acl/acl_prof.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/profiler/e2e_profiler.h"
#include "torch_npu/csrc/profiler/profiler_legacy.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
std::atomic<bool> global_enable_profiling(false);

std::atomic<bool>& get_global_enable_profiling() {
  return global_enable_profiling;
}

namespace torch_npu {
namespace profiler {

aclprofConfig* local_profCfg = nullptr;
bool global_call_stack = false;
std::mutex g_rangeStampMtx;
std::mutex g_pipelineStampMtx;
struct StampGroupMng g_rangeStamp;
struct StampRingQueue g_markStamp;
struct StampRingQueue g_pipelineStamp;
bool g_concatenateReport = false;

void InitRangeStamp() {
  g_rangeStamp.groups = reinterpret_cast<struct StampGroup *>(malloc(sizeof(struct StampGroup) * GROUP_NUM));
  if (g_rangeStamp.groups == nullptr) {
    NPU_LOGE("InitRangeStamp malloc fail.");
    return;
  }
  memset(g_rangeStamp.groups, 0, sizeof(struct StampGroup) * GROUP_NUM);
  g_rangeStamp.idleGrpInd.clear();
  for (int i = 0; i < GROUP_NUM; i++) {
    g_rangeStamp.idleGrpInd.push_back(i);
    g_rangeStamp.groups[i].idleNodeInd = 0;
    g_rangeStamp.groups[i].fillEndNodeCnt = 0;
    for (int j = 0; j < GROUP_CAPACITY; j++) {
      g_rangeStamp.groups[i].nodes[j].magicNum = MSPROF_DATA_HEAD_MAGIC_NUM;
      g_rangeStamp.groups[i].nodes[j].dataTag = MSPROF_MSPROFTX_DATA_TAG;
      g_rangeStamp.groups[i].nodes[j].groupId = i;
      g_rangeStamp.groups[i].nodes[j].nodeId = j;
      g_rangeStamp.groups[i].nodes[j].processId = static_cast<int>(getpid());
    }
  }
  g_rangeStamp.idleGrpInd.pop_front();
  g_rangeStamp.curUsedGrpInd = 0;
}

struct Stamp *GetRangeStamp() {
  if (g_rangeStamp.groups == nullptr) {
    NPU_LOGE("GetRangeStamp groups is null.");
    return nullptr;
  }
  std::lock_guard<std::mutex> lk(g_rangeStampMtx);
  if (g_rangeStamp.curUsedGrpInd < 0) {
    NPU_LOGE("GetRangeStamp fail, no idle node.");
    return nullptr;
  }
  int usedGrpInd = g_rangeStamp.curUsedGrpInd;
  int idleNodeInd = g_rangeStamp.groups[usedGrpInd].idleNodeInd;
  g_rangeStamp.groups[usedGrpInd].idleNodeInd++;
  if (g_rangeStamp.groups[usedGrpInd].idleNodeInd >= GROUP_CAPACITY) {
    if (g_rangeStamp.idleGrpInd.empty()) {
      g_rangeStamp.curUsedGrpInd = -1;
    } else {
      g_rangeStamp.curUsedGrpInd = g_rangeStamp.idleGrpInd.front();
      g_rangeStamp.idleGrpInd.pop_front();
    }
  }
  return &g_rangeStamp.groups[usedGrpInd].nodes[idleNodeInd];
}

void PutRangeStamp(struct Stamp *stamp) {
  if (g_rangeStamp.groups == nullptr || stamp == nullptr) {
    NPU_LOGE("PutRangeStamp groups/stamp is null.");
    return;
  }
  std::lock_guard<std::mutex> lk(g_rangeStampMtx);
  int grpIdx = stamp->groupId;
  g_rangeStamp.groups[grpIdx].fillEndNodeCnt++;
  if (g_rangeStamp.groups[grpIdx].fillEndNodeCnt >= GROUP_CAPACITY) {
    int ret = at_npu::native::AclprofReportStamp("torch_op", strlen("torch_op"),
      (unsigned char *)g_rangeStamp.groups[grpIdx].nodes,
      sizeof(struct Stamp) * GROUP_CAPACITY);
    if (ret != ACL_ERROR_NONE) {
      NPU_LOGE("PutRangeStamp report fail, ret=%d.", ret);
    }
    g_rangeStamp.groups[grpIdx].idleNodeInd = 0;
    g_rangeStamp.groups[grpIdx].fillEndNodeCnt = 0;
    g_rangeStamp.idleGrpInd.push_back(grpIdx);
    if (g_rangeStamp.curUsedGrpInd < -1) {
      g_rangeStamp.curUsedGrpInd = g_rangeStamp.idleGrpInd.front();
      g_rangeStamp.idleGrpInd.pop_front();
    }
  }
}

void FlushRangeStamp() {
  if (g_rangeStamp.groups == nullptr) {
    NPU_LOGE("FlushRangeStamp groups is null.");
    return;
  }
  if (g_rangeStamp.curUsedGrpInd < 0) {
    return;
  }
  int grpIdx = g_rangeStamp.curUsedGrpInd;
  int fillEndNodeCnt = g_rangeStamp.groups[grpIdx].fillEndNodeCnt;
  if (fillEndNodeCnt == 0) {
    return;
  }
  int ret = at_npu::native::AclprofReportStamp("torch_op", strlen("torch_op"),
    (unsigned char *)g_rangeStamp.groups[grpIdx].nodes,
    sizeof(struct Stamp) * fillEndNodeCnt);
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("FlushRangeStamp report fail, ret=%d.", ret);
  }
}

void UninitRangeStamp() {
  if (g_rangeStamp.groups != nullptr) {
    free(g_rangeStamp.groups);
    g_rangeStamp.groups = nullptr;
  }
}

void InitMarkStamp() {
  g_markStamp.idleNodeInd = 0;
  g_markStamp.nodes = reinterpret_cast<struct Stamp *>(malloc(sizeof(struct Stamp) * STAMP_QUEUE_LEN));
  if (g_markStamp.nodes == nullptr) {
    NPU_LOGE("InitMarkStamp malloc fail.");
    return;
  }
  memset(g_markStamp.nodes, 0, sizeof(struct Stamp) * STAMP_QUEUE_LEN);
  for (int i = 0; i < STAMP_QUEUE_LEN; i++) {
    g_markStamp.nodes[i].magicNum = MSPROF_DATA_HEAD_MAGIC_NUM;
    g_markStamp.nodes[i].dataTag = MSPROF_MSPROFTX_DATA_TAG;
    g_markStamp.nodes[i].processId = static_cast<int>(getpid());
  }
}

static int64_t getClockMonotonicRaw() {
  struct timespec ts{};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + static_cast<int64_t>(ts.tv_nsec);
}

void PutMarkStamp(const std::string &opName) {
  if (!g_concatenateReport) {
    using namespace at_npu::native;
    void *local_stamp = AclprofCreateStamp();
    if (local_stamp == nullptr) {
      return;
    }
    static const std::string tag_name = "torch_cann_op";
    do {
        if (AclprofSetStampTagName(local_stamp, tag_name.c_str(), tag_name.size()) != ACL_ERROR_NONE) {
            break;
        }
        if (AclprofSetStampTraceMessage(local_stamp, opName.c_str(), opName.size()) != ACL_ERROR_NONE) {
            break;
        }
        if (at_npu::native::AclprofMark(local_stamp) != ACL_ERROR_NONE) {
            break;
        }
    } while (0);
    at_npu::native::AclprofDestroyStamp(local_stamp);
  } else {
    if (g_markStamp.nodes == nullptr) {
      NPU_LOGE("PutMarkStamp nodes is null.");
      return;
    }
    // get idle node index
    static std::mutex markStampMtx;
    int index;
    do {
        std::lock_guard<std::mutex> lk(markStampMtx);
        index = g_markStamp.idleNodeInd;
        g_markStamp.idleNodeInd = (g_markStamp.idleNodeInd + 1) % STAMP_QUEUE_LEN;
    } while (0);

    // set tid/time/opname
    static thread_local int tid = syscall(SYS_gettid);
    g_markStamp.nodes[index].threadId = tid;
    g_markStamp.nodes[index].eventType = 0;
    g_markStamp.nodes[index].startTime = static_cast<unsigned long long>(torch_npu::toolkit::profiler::Utils::GetClockTime());
    g_markStamp.nodes[index].endTime = g_markStamp.nodes[index].startTime;
    std::strncpy(g_markStamp.nodes[index].message, opName.c_str(), OP_NAME_LEN);
    // report data
    if ((index % ONCE_REPORT_NUM) == (ONCE_REPORT_NUM - 1)) {
      int ret = at_npu::native::AclprofReportStamp("torch_cann_op", strlen("torch_cann_op"),
        (unsigned char *)&g_markStamp.nodes[index + 1 - ONCE_REPORT_NUM],
        sizeof(struct Stamp) * ONCE_REPORT_NUM);
      if (ret != ACL_ERROR_NONE) {
        NPU_LOGE("PutMarkStamp report fail, ret=%d.", ret);
      }
    }
  }
}

void FlushMarkStamp() {
  if (g_markStamp.nodes == nullptr) {
    NPU_LOGE("FlushMarkStamp nodes is null.");
    return;
  }
  int unReportNum = g_markStamp.idleNodeInd % ONCE_REPORT_NUM;
  if (unReportNum == 0) {
    return;
  }
  int ret = at_npu::native::AclprofReportStamp("torch_cann_op", strlen("torch_cann_op"),
    (unsigned char *)&g_markStamp.nodes[g_markStamp.idleNodeInd - unReportNum],
    sizeof(struct Stamp) * unReportNum);
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("FlushMarkStamp report fail, ret=%d.", ret);
  }
}

void UninitMarkStamp() {
  if (g_markStamp.nodes != nullptr) {
    free(g_markStamp.nodes);
    g_markStamp.nodes = nullptr;
  }
}

void InitPipelineStamp() {
  g_pipelineStamp.idleNodeInd = 0;
  g_pipelineStamp.nodes = reinterpret_cast<struct Stamp *>(malloc(sizeof(struct Stamp) * STAMP_QUEUE_LEN));
  if (g_pipelineStamp.nodes == nullptr) {
    NPU_LOGE("InitPipelineStamp malloc fail.");
    return;
  }
  memset(g_pipelineStamp.nodes, 0, sizeof(struct Stamp) * STAMP_QUEUE_LEN);
  for (int i = 0; i < STAMP_QUEUE_LEN; i++) {
    g_pipelineStamp.nodes[i].magicNum = MSPROF_DATA_HEAD_MAGIC_NUM;
    g_pipelineStamp.nodes[i].dataTag = MSPROF_MSPROFTX_DATA_TAG;
    g_pipelineStamp.nodes[i].processId = static_cast<int>(getpid());
  }
}

void PutPipelineStamp(uint32_t category, const std::string &op_name) {
  static const std::string tag_name = "torch_pipeline";
  if (!g_concatenateReport) {
    void *stamp = at_npu::native::AclprofCreateStamp();
    if (stamp == nullptr) {
      return;
    }
    if (at_npu::native::AclprofSetStampTagName(stamp, tag_name.c_str(), tag_name.size()) != ACL_ERROR_NONE ||
      at_npu::native::AclprofSetStampCategory(stamp, category) != ACL_ERROR_NONE ||
      at_npu::native::AclprofSetStampTraceMessage(stamp, op_name.c_str(), op_name.size()) != ACL_ERROR_NONE ||
      at_npu::native::AclprofMark(stamp) != ACL_ERROR_NONE) {
      NPU_LOGE("Report Pipeline data to MsProfiler failed.");
    }
    at_npu::native::AclprofDestroyStamp(stamp);
  } else {
    if (g_pipelineStamp.nodes == nullptr) {
      NPU_LOGE("PutPipelineStamp nodes is null.");
      return;
    }
    std::lock_guard<std::mutex> lk(g_pipelineStampMtx);
    int index = g_pipelineStamp.idleNodeInd;
    g_pipelineStamp.idleNodeInd = (g_pipelineStamp.idleNodeInd + 1) & (STAMP_QUEUE_LEN - 1);
    static thread_local int tid = syscall(SYS_gettid);
    g_pipelineStamp.nodes[index].threadId = tid;
    g_pipelineStamp.nodes[index].category = static_cast<int>(category);
    g_pipelineStamp.nodes[index].eventType = 0;
    g_pipelineStamp.nodes[index].startTime = static_cast<unsigned long long>(torch_npu::toolkit::profiler::Utils::GetClockTime());
    g_pipelineStamp.nodes[index].endTime = g_pipelineStamp.nodes[index].startTime;
    std::strncpy(g_pipelineStamp.nodes[index].message, op_name.c_str(), OP_NAME_LEN);
    if ((index & (ONCE_REPORT_NUM - 1)) == (ONCE_REPORT_NUM - 1)) {
      int ret = at_npu::native::AclprofReportStamp(tag_name.c_str(), tag_name.size(),
          (unsigned char *)&g_pipelineStamp.nodes[index + 1 - ONCE_REPORT_NUM],
          sizeof(struct Stamp) * ONCE_REPORT_NUM);
      if (ret != ACL_ERROR_NONE) {
        NPU_LOGE("PutPipelineStamp report fail, ret=%d.", ret);
      }
    }
  }
}

void FlushPipelineStamp() {
  static const std::string tag_name = "torch_pipeline";
  if (g_pipelineStamp.nodes == nullptr) {
    NPU_LOGE("FlushPipelineStamp nodes is null.");
    return;
  }
  int unReportNum = g_pipelineStamp.idleNodeInd % ONCE_REPORT_NUM;
  if (unReportNum == 0) {
    return;
  }
  int ret = at_npu::native::AclprofReportStamp(tag_name.c_str(), tag_name.size(),
      (unsigned char *)&g_pipelineStamp.nodes[g_pipelineStamp.idleNodeInd - unReportNum],
      sizeof(struct Stamp) * unReportNum);
  if (ret != ACL_ERROR_NONE) {
    NPU_LOGE("FlushPipelineStamp report fail, ret=%d.", ret);
  }
}

void UninitPipelineStamp() {
  if (g_pipelineStamp.nodes != nullptr) {
    free(g_pipelineStamp.nodes);
    g_pipelineStamp.nodes = nullptr;
  }
}

void MarkQueueStamp(uint32_t category, const std::string &op_name) {
  if (!global_enable_profiling.load()) {
    return;
  }
  PutPipelineStamp(category, op_name);
}

void MarkQueueStamp(uint32_t category, void *data, size_t offset) {
  if (!global_enable_profiling.load()) {
    return;
  }
  void *cur_addr = (uint8_t *)data + (sizeof(c10_npu::queue::QueueParas) + at_npu::native::MAX_PARAS_BYTE_SIZE) * offset;
  auto cur_param = static_cast<c10_npu::queue::QueueParas *>(cur_addr);
  if (cur_param->paramType != c10_npu::queue::COMPILE_AND_EXECUTE) {
    return;
  }
  auto param_val = static_cast<at_npu::native::ExecuteParas *>(cur_param->paramVal);
  PutPipelineStamp(category, std::string(param_val->opType));
}

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
  // 5. init local stamp info

  int deviceIndex = 0;
  aclError ret = aclrtGetDevice(&deviceIndex);
  if (ret) {
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
  if (ret) {
    NPU_LOGE("In npu e2e profiling, AclProfStart fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
    (void)at_npu::native::AclProfilingFinalize();
    return;
  }
  if (g_concatenateReport) {
    InitRangeStamp();
    InitMarkStamp();
    InitPipelineStamp();
  }
}

void PushStartTime(at::RecordFunction& fn) {
  if (!g_concatenateReport || global_call_stack) {
    auto local_stamp_ = at_npu::native::AclprofCreateStamp();
    if (local_stamp_  == nullptr) {
      NPU_LOGE("In npu e2e profiling, aclprofCreateStamp failed, created stamp is nullptr.");
      return;
    }
    static const std::string tag_name = "torch_op";
    auto ret = at_npu::native::AclprofSetStampTagName(local_stamp_, tag_name.c_str(), tag_name.size());
    CheckProfilerRet(ret, "In npu e2e profiling, AclprofSetStampTagName set failed.");
    ret = at_npu::native::AclprofSetStampTraceMessage(
        local_stamp_, fn.name(), strlen(fn.name()));
    CheckProfilerRet(ret, "In npu e2e profiling, AclprofSetStampTraceMessage set failed.");
    if (global_call_stack) {
      std::string seq_nr = "seq=" + std::to_string(fn.seqNr());
      std::vector<std::string> py_stack;
      std::string call_stack_data;
      if (fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
        auto cs = prepareCallstack(torch::jit::currentCallstack());
        if (cs.empty()) {
          cs = prepareCallstack(torch::jit::tracer::pythonCallstack());
        }
        py_stack = callstack2Str(cs);
        for (size_t i = 0; i < py_stack.size(); ++i) {
          call_stack_data += py_stack[i];
          call_stack_data += ("," + seq_nr);
          call_stack_data += ((i == py_stack.size() - 1) ? "" : ";");
        }
      } else {
        call_stack_data = seq_nr;
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
  } else {
    struct Stamp *node = GetRangeStamp();
    if (node == nullptr) {
      return;
    }
    static thread_local int tid = syscall(SYS_gettid);
    node->threadId = tid;
    node->startTime = static_cast<unsigned long long>(torch_npu::toolkit::profiler::Utils::GetClockTime());
    int nameLen = strlen(fn.name());
    std::strncpy(node->message, fn.name(), OP_NAME_LEN);
    fn.setForwardThreadId(reinterpret_cast<uint64_t>(node));
  }
}

void PopEndTime(const at::RecordFunction& fn) {
  if (!g_concatenateReport || global_call_stack) {
    auto ret = at_npu::native::AclprofRangeStop((uint32_t)fn.handle());
    CheckProfilerRet(ret, "In npu e2e profiling, AclprofRangeStop failed.");
    at_npu::native::AclprofDestroyStamp((void*)fn.forwardThreadId());
  } else {
    struct Stamp *node = reinterpret_cast<struct Stamp *>(fn.forwardThreadId());
    node->endTime = static_cast<unsigned long long>(torch_npu::toolkit::profiler::Utils::GetClockTime());
    node->eventType = 2;  // msproftx data envent type: START_OR_STOP
    PutRangeStamp(node);
  }
}

void InitE2eProfiler(const std::string dump_path, uint64_t npu_event,
    uint64_t aicore_metrics, bool call_stack) {
  global_call_stack = call_stack;
  g_concatenateReport = at_npu::native::CheckInterfaceReportStamp();
  InitMsPorf(dump_path, npu_event, aicore_metrics);
  global_enable_profiling.store(true);
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
  c10_npu::npuSynchronizeDevice();
  global_enable_profiling.store(false);
  auto ret = at_npu::native::AclProfilingStop(local_profCfg);
  if (ret) {
    NPU_LOGE("In npu e2e profiling, AclProfStop fail, error code: %d", ret);
    C10_NPU_SHOW_ERR_MSG();
  }
    ret = at_npu::native::AclProfilingDestroyConfig(local_profCfg);
    if (ret != ACL_SUCCESS) {
        NPU_LOGE("AclProfDestoryConfig fail, error code: %d", ret);
    }
    local_profCfg = nullptr;
  if (g_concatenateReport) {
    FlushRangeStamp();
    FlushMarkStamp();
    FlushPipelineStamp();
    UninitRangeStamp();
    UninitMarkStamp();
    UninitPipelineStamp();
  }
  at_npu::native::AclProfilingFinalize();
  at::clearThreadLocalCallbacks();
}

}
}
