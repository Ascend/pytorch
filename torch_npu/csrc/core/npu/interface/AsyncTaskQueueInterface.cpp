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
  if (!other.pinMem.empty()) {
    this->pinMem = other.pinMem;
  }
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
  AsyncCopyTask(
      void* dst,
      size_t dstLen,
      void* src,
      size_t srcLen,
      aclrtMemcpyKind kind,
      c10::Storage& st);
  ~AsyncCopyTask() = default;
  void LaunchCopyTask();
  void LaunchCopyTask(bool isPinnedMem);

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
      c10_npu::NPUStream npuStream,
      c10::SmallVector<c10::Storage, c10_npu::N>& needClearVec);
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

AsyncCopyTask::AsyncCopyTask(
    void* dst,
    size_t dstLen,
    void* src,
    size_t srcLen,
    aclrtMemcpyKind kind,
    c10::Storage& st) {
  copyParam_.dst = dst;
  copyParam_.dstLen = dstLen;
  copyParam_.src = src;
  copyParam_.srcLen = srcLen;
  copyParam_.kind = kind;
  copyParam_.pinMem.emplace_back(st);
}

void AsyncCopyTask::LaunchCopyTask() {
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam_);
    c10::SmallVector<c10::Storage, N> needClearVec;
    c10_npu::enCurrentNPUStream(&params, needClearVec);
    // free pin memory
    needClearVec.clear();
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

void AsyncCopyTask::LaunchCopyTask(bool isPinnedMem) {
  if (c10_npu::option::OptionsManager::CheckQueueEnable() && isPinnedMem) {
    QueueParas params(ASYNC_MEMCPY_EX, sizeof(CopyParas), &copyParam_);
    c10::SmallVector<c10::Storage, N> needClearVec;
    c10_npu::enCurrentNPUStream(&params, needClearVec);
    // free pin memory
    needClearVec.clear();
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

aclError LaunchAsyncCopyTask(
    void* dst,
    size_t dstLen,
    void* src,
    size_t srcLen,
    aclrtMemcpyKind kind,
    c10::Storage& st,
    bool isPinMem) {
  AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind, st);
  copyTask.LaunchCopyTask(isPinMem);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchRecordTask(
    c10_npu::NPUStream npuStream,
    c10::SmallVector<c10::Storage, N>& needClearVec) {
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(RECORD_EVENT, sizeof(EventParas), &eventParam_);
    c10_npu::enCurrentNPUStream(&params, needClearVec);
    c10_npu::setCurrentNPUStream(currentStream);
  } else {
    C10_NPU_CHECK(aclrtRecordEvent(eventParam_.event, npuStream));
  }
}

aclError HostAllocatorLaunchRecordEventTask(
    aclrtEvent event,
    c10_npu::NPUStream npuStream,
    c10::SmallVector<c10::Storage, N>& needClearVec) {
  EventTask recordTask(event, HOST_ALLOCATOR_EVENT);
  recordTask.LaunchRecordTask(npuStream, needClearVec);
  return ACL_ERROR_NONE;
}

aclError NpuAllocatorLaunchRecordEventTask(
    aclrtEvent event,
    c10_npu::NPUStream npuStream) {
  EventTask recordTask(event, NPU_ALLOCATOR_EVENT);
  c10::SmallVector<c10::Storage, N> needClearVec;
  recordTask.LaunchRecordTask(npuStream, needClearVec);
  return ACL_ERROR_NONE;
}

aclError LaunchRecordEventTask(aclrtEvent event, c10_npu::NPUStream npuStream) {
  EventTask recordTask(event);
  c10::SmallVector<c10::Storage, N> needClearVec;
  recordTask.LaunchRecordTask(npuStream, needClearVec);
  return ACL_ERROR_NONE;
}

void EventTask::LaunchWaitTask(c10_npu::NPUStream npuStream) {
  if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
    c10_npu::NPUStream currentStream = c10_npu::getCurrentNPUStream();
    c10_npu::setCurrentNPUStream(npuStream);
    QueueParas params(WAIT_EVENT, sizeof(EventParas), &eventParam_);
    c10::SmallVector<c10::Storage, N> needClearVec;
    c10_npu::enCurrentNPUStream(&params, needClearVec);
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
    c10::SmallVector<c10::Storage, N> needClearVec;
    c10_npu::enCurrentNPUStream(&params, needClearVec);
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