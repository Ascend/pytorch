// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include "c10/npu/OptionsManager.h"

namespace c10 {
namespace npu {
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

void EventParas::Copy(EventParas& other)
{
  this->event = other.event;
}

class AsyncCopyTask {
public:
  AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind);
  AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind, Storage& st);
  ~AsyncCopyTask() = default;
  void LaunchCopyTask();
  void LaunchCopyTask(bool isPinnedMem);

private:
  CopyParas copyParam_;
};

class EventTask {
public:
  EventTask(aclrtEvent event);
  ~EventTask() = default;
  void LaunchRecordTask(at::npu::NPUStream npuStream, SmallVector<Storage, N>& needClearVec);

private:
  EventParas eventParam_;
};

AsyncCopyTask::AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind)
{
  copyParam_.dst = dst;
  copyParam_.dstLen = dstLen;
  copyParam_.src = src;
  copyParam_.srcLen = srcLen;
  copyParam_.kind = kind;
}

AsyncCopyTask::AsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind,
  Storage& st)
{
  copyParam_.dst = dst;
  copyParam_.dstLen = dstLen;
  copyParam_.src = src;
  copyParam_.srcLen = srcLen;
  copyParam_.kind = kind;
  copyParam_.pinMem.emplace_back(st);
}

void AsyncCopyTask::LaunchCopyTask()
{
  if (c10::npu::OptionsManager::CheckQueueEnable()) {
    QueueParas params(ASYNC_MEMCPY, sizeof(CopyParas), &copyParam_);
    SmallVector<Storage, N> needClearVec;
    c10::npu::enCurrentNPUStream(&params, needClearVec);
    // free pin memory
    needClearVec.clear();
  } else {
    c10::npu::NPUStream stream = c10::npu::getCurrentNPUStream();
    AT_NPU_CHECK(aclrtMemcpyAsync(
      copyParam_.dst,
      copyParam_.dstLen,
      copyParam_.src,
      copyParam_.srcLen,
      copyParam_.kind,
      stream));
  }
}

void AsyncCopyTask::LaunchCopyTask(bool isPinnedMem)
{
  if (c10::npu::OptionsManager::CheckQueueEnable() && isPinnedMem) {
    QueueParas params(ASYNC_MEMCPY_EX, sizeof(CopyParas), &copyParam_);
    SmallVector<Storage, N> needClearVec;
    c10::npu::enCurrentNPUStream(&params, needClearVec);
    // free pin memory
    needClearVec.clear();
  } else {
    c10::npu::NPUStream stream = c10::npu::getCurrentNPUStream();
    AT_NPU_CHECK(aclrtMemcpyAsync(
      copyParam_.dst,
      copyParam_.dstLen,
      copyParam_.src,
      copyParam_.srcLen,
      copyParam_.kind,
      stream));
  }
}

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind)
{
  AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind);
  copyTask.LaunchCopyTask();
  return ACL_ERROR_NONE;
}

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind,
  Storage& st, bool isPinMem)
{
  AsyncCopyTask copyTask(dst, dstLen, src, srcLen, kind, st);
  copyTask.LaunchCopyTask(isPinMem);
  return ACL_ERROR_NONE;
}

EventTask::EventTask(aclrtEvent event)
{
  eventParam_.event = event;
}

void EventTask::LaunchRecordTask(at::npu::NPUStream npuStream, SmallVector<Storage, N>& needClearVec)
{
  if (c10::npu::OptionsManager::CheckQueueEnable()) {
    at::npu::NPUStream currentStream = c10::npu::getCurrentNPUStream();
    c10::npu::setCurrentNPUStream(npuStream);
    QueueParas params(RECORD_EVENT, sizeof(EventParas), &eventParam_);
    c10::npu::enCurrentNPUStream(&params, needClearVec);
    c10::npu::setCurrentNPUStream(currentStream);
  } else {
    AT_NPU_CHECK(aclrtRecordEvent(
      eventParam_.event,
      npuStream));
  }
}

aclError LaunchRecordEventTask(aclrtEvent event, at::npu::NPUStream npuStream, SmallVector<Storage, N>& needClearVec)
{
  EventTask recordTask(event);
  recordTask.LaunchRecordTask(npuStream, needClearVec);
  return ACL_ERROR_NONE;
}
} // namespace queue
} // namespace npu
} // namespace c10