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

#include "AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include <ATen/record_function.h>
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/NPUDefine.h"
#include "third_party/acl/inc/acl/acl_rt.h"
namespace c10_npu {
namespace queue {
std::atomic<uint64_t> QueueParas::g_correlation_id{0};
std::map<int64_t, std::string> CopyParas::COPY_PARAS_MAP{
  {ACL_MEMCPY_HOST_TO_HOST, "acl_memcpy_host_to_host"},
  {ACL_MEMCPY_HOST_TO_DEVICE, "acl_memcpy_host_to_device"},
  {ACL_MEMCPY_DEVICE_TO_HOST, "acl_memcpy_device_to_host"},
  {ACL_MEMCPY_DEVICE_TO_DEVICE, "acl_memcpy_device_to_device"},
};
std::map<int64_t, std::string> EventParas::EVENT_PARAS_MAP{
  {HOST_ALLOCATOR_EVENT, "host_allocator_event"},
  {NPU_ALLOCATOR_EVENT, "npu_alloctor_event"},
  {RESERVED, "reserved"},
};
void CopyParas::Copy(CopyParas& other) {
  this->dst = other.dst;
  this->dstLen = other.dstLen;
  this->src = other.src;
  this->srcLen = other.srcLen;
  this->kind = other.kind;
}

void EventParas::Copy(EventParas& other) {
  this->event = other.event;
  this->eventAllocatorType = other.eventAllocatorType;
}

class AsyncCopyTask {
public:
  AsyncCopyTask(
      void* dst,
      size_t dstLen,
      void* src,
      size_t srcLen,
      aclrtMemcpyKind kind);
  ~AsyncCopyTask() = default;
  void LaunchCopyTask();

private:
  CopyParas copyParam_;
};

class EventTask {
public:
  explicit EventTask(
      aclrtEvent event,
      EventAllocatorType allocatorType = RESERVED)
      : eventParam_(event, allocatorType){};
  ~EventTask() = default;
  void LaunchRecordTask(
      c10_npu::NPUStream npuStream);
  void LaunchWaitTask(c10_npu::NPUStream npuStream);
  void LaunchResetTask(c10_npu::NPUStream npuStream);
  void LaunchLazyDestroyTask();

private:
  EventParas eventParam_;
};

AsyncCopyTask::AsyncCopyTask(
    void* dst,
    size_t dstLen,
    void* src,
    size_t srcLen,
    aclrtMemcpyKind kind) {
  copyParam_.dst = dst;
  copyParam_.dstLen = dstLen;
  copyParam_.src = src;
  copyParam_.srcLen = srcLen;
  copyParam_.kind = kind;
}

void AsyncCopyTask::LaunchCopyTask() {
  RECORD_FUNCTION(CopyParas::COPY_PARAS_MAP[copyParam_.kind], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, CopyParas::COPY_PARAS_MAP[copyParam_.kind]);
    QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam_);
    c10_npu::enCurrentNPUStream(&params);
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, CopyParas::COPY_PARAS_MAP[copyParam_.kind], params.correlation_id);
  } else {
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(aclrtMemcpyAsync(
        copyParam_.dst,
        copyParam_.dstLen,
        copyParam_.src,
        copyParam_.srcLen,
        copyParam_.kind,
        stream));
  }
}

aclError LaunchAsyncCopyTask(
    void* dst,
    size_t dstLen,
    void* src,
    size_t srcLen,
    aclrtMemcpyKind kind) {
  AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind);
  copyTask.LaunchCopyTask();
  return ACL_ERROR_NONE;
}

void EventTask::LaunchRecordTask(
    c10_npu::NPUStream npuStream) {
  RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType]);
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(RECORD_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    c10_npu::setCurrentNPUStream(currentStream);
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], params.correlation_id);
  } else {
    NPU_CHECK_ERROR(aclrtRecordEvent(eventParam_.event, npuStream));
    ASCEND_LOGI("aclrtRecordEvent is successfully executed.");
  }
}

aclError HostAllocatorLaunchRecordEventTask(
    aclrtEvent event,
    c10_npu::NPUStream npuStream) {
  EventTask recordTask(event, HOST_ALLOCATOR_EVENT);
  recordTask.LaunchRecordTask(npuStream);
  return ACL_ERROR_NONE;
}

aclError NpuAllocatorLaunchRecordEventTask(
    aclrtEvent event,
    c10_npu::NPUStream npuStream) {
  EventTask recordTask(event, NPU_ALLOCATOR_EVENT);
  recordTask.LaunchRecordTask(npuStream);
  ASCEND_LOGI("NpuAllocatorLaunchRecordEventTask is successfully executed.");
  return ACL_ERROR_NONE;
}

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask recordTask(event);
  recordTask.LaunchRecordTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchWaitTask(c10_npu::NPUStream npuStream) {
  RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType]);
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(WAIT_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    c10_npu::setCurrentNPUStream(currentStream);
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], params.correlation_id);
  } else {
    NPU_CHECK_ERROR(aclrtStreamWaitEvent(npuStream, eventParam_.event));
    ASCEND_LOGI("aclrtStreamWaitEvent is successfully executed.");
  }
}

aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask waitTask(event);
  waitTask.LaunchWaitTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchResetTask(c10_npu::NPUStream npuStream) {
  RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType]);
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(RESET_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    c10_npu::setCurrentNPUStream(currentStream);
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], params.correlation_id);
  } else {
    NPU_CHECK_ERROR(aclrtResetEvent(eventParam_.event, npuStream));
    ASCEND_LOGI("aclrtResetEvent is successfully executed.");
  }
}

aclError LaunchResetEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask resetTask(event);
  resetTask.LaunchResetTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchLazyDestroyTask() {
  RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType]);
    QueueParas params(LAZY_DESTROY_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], params.correlation_id);
  } else {
    NPU_CHECK_ERROR(c10_npu::NPUEventManager::GetInstance().LazyDestroy(
        eventParam_.event));
  }
}

aclError LaunchLazyDestroyEventTask(aclrtEvent event) {
  EventTask lazyDestroyTask(event);
  lazyDestroyTask.LaunchLazyDestroyTask();
  return ACL_ERROR_NONE;
}
} // namespace queue
} // namespace c10