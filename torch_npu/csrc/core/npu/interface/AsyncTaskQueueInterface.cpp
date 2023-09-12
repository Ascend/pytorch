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
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, CopyParas::COPY_PARAS_MAP[copyParam_.kind]);
#endif
    QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam_);
    c10_npu::enCurrentNPUStream(&params);
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, CopyParas::COPY_PARAS_MAP[copyParam_.kind], params.correlation_id);
#endif
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

void EventTask::LaunchRecordTask(c10_npu::NPUStream npuStream) {
  RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType]);
#endif
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(RECORD_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::NPUEventManager::GetInstance().IncreaseUnrecordedCount(eventParam_.event);
    c10_npu::enCurrentNPUStream(&params);
    c10_npu::setCurrentNPUStream(currentStream);
    ASCEND_LOGI("Event: LaunchRecordTask is successfully executed, eventParam_.event=%p.", eventParam_.event);
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], params.correlation_id);
#endif
  } else {
    NPU_CHECK_ERROR(aclrtRecordEvent(eventParam_.event, npuStream));
    ASCEND_LOGI("Event: aclrtRecordEvent is successfully executed, eventParam_.event=%p.", eventParam_.event);
  }
}

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask recordTask(event);
  recordTask.LaunchRecordTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchWaitTask(c10_npu::NPUStream npuStream) {
  RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType]);
#endif
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(WAIT_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    c10_npu::setCurrentNPUStream(currentStream);
    ASCEND_LOGI("Event: LaunchWaitTask is successfully executed, eventParam_.event=%p.", eventParam_.event);
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], params.correlation_id);
#endif
  } else {
    NPU_CHECK_ERROR(aclrtStreamWaitEvent(npuStream, eventParam_.event));
    ASCEND_LOGI("Event: aclrtStreamWaitEvent is successfully executed, eventParam_.event=%p.", eventParam_.event);
  }
}

aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask waitTask(event);
  waitTask.LaunchWaitTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchLazyDestroyTask() {
  RECORD_FUNCTION(EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], std::vector<c10::IValue>({}));
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType]);
#endif
    QueueParas params(LAZY_DESTROY_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    ASCEND_LOGI("Event: LaunchLazyDestroyTask is successfully executed, eventParam_.event=%p.", eventParam_.event);
#ifndef BUILD_LIBTORCH
    at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, EventParas::EVENT_PARAS_MAP[eventParam_.eventAllocatorType], params.correlation_id);
#endif
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