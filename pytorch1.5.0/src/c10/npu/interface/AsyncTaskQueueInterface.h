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

#ifndef __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__
#define __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__

#include "c10/core/Storage.h"
#include "c10/npu/NPUStream.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace c10 {
namespace npu {
namespace queue {
struct CopyParas {
  void *dst = nullptr;
  size_t dstLen = 0;
  void *src = nullptr;
  size_t srcLen = 0;
  aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_HOST;
  SmallVector<Storage, 1> pinMem;
  void Copy(CopyParas& other);
};

struct EventParas {
  aclrtEvent event = nullptr;
  void Copy(EventParas& other);
};

enum QueueParamType {
  COMPILE_AND_EXECUTE = 1,
  ASYNC_MEMCPY = 2,
  ASYNC_MEMCPY_EX = 3,
  RECORD_EVENT = 4,
};

struct QueueParas {
  QueueParas(QueueParamType type, size_t len, void *val) : paramType(type), paramLen(len), paramVal(val) {}
  QueueParamType paramType = COMPILE_AND_EXECUTE;
  size_t paramLen = 0;
  void* paramVal = nullptr;
};

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind);

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind,
  Storage& st, bool isPinMem);

aclError LaunchRecordEventTask(aclrtEvent event, at::npu::NPUStream npuStream, SmallVector<Storage, N>& needClearVec);
} // namespace queue
} // namespace npu
} // namespace c10

#endif // __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__