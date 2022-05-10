#include "AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/NPUEventManager.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
namespace c10_npu {
namespace queue {
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
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam_);
    c10_npu::enCurrentNPUStream(&params);
  } else {
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    C10_NPU_CHECK(aclrtMemcpyAsync(
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
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(RECORD_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    c10_npu::setCurrentNPUStream(currentStream);
  } else {
    C10_NPU_CHECK(aclrtRecordEvent(eventParam_.event, npuStream));
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
  return ACL_ERROR_NONE;
}

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask recordTask(event);
  recordTask.LaunchRecordTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchWaitTask(c10_npu::NPUStream npuStream) {
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(WAIT_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
    c10_npu::setCurrentNPUStream(currentStream);
  } else {
    C10_NPU_CHECK(aclrtStreamWaitEvent(npuStream, eventParam_.event));
  }
}

aclError LaunchWaitEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask waitTask(event);
  waitTask.LaunchWaitTask(npuStream);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchLazyDestroyTask() {
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    QueueParas params(LAZY_DESTROY_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params);
  } else {
    C10_NPU_CHECK(c10_npu::NPUEventManager::GetInstance().LazyDestroy(
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