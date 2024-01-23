// Copyright (c) 2022 Huawei Technologies Co., Ltd
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

#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"

namespace c10d_npu {
#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libhccl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName) \
  GET_FUNCTION(libhccl, funcName)

REGISTER_LIBRARY(libhccl)
LOAD_FUNCTION(HcclAlltoAllV)
LOAD_FUNCTION(HcclReduce)
LOAD_FUNCTION(HcclGetCommAsyncError)
LOAD_FUNCTION(HcclAlltoAll)
LOAD_FUNCTION(HcclScatter)
LOAD_FUNCTION(HcclBatchSendRecv)

extern HcclResult hcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls,
    HcclDataType recvType, HcclComm comm, aclrtStream stream) {
  typedef HcclResult(*HcclAlltoAllVFunc)(
      const void *, const void *, const void *, HcclDataType,
      const void *, const void *, const void *, HcclDataType,
      HcclComm, aclrtStream);
  static HcclAlltoAllVFunc func = nullptr;
  if (func == nullptr) {
    func = (HcclAlltoAllVFunc)GET_FUNC(HcclAlltoAllV);
  }
  TORCH_CHECK(func, "Failed to find function ", "HcclAlltoAllV");
  auto ret = func(sendBuf, sendCounts, sdispls, sendType,
      recvBuf, recvCounts, rdispls, recvType, comm, stream);
  return ret;
}

extern HcclResult hcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType sendType,
    HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream) {
  typedef HcclResult(*HcclReduceVFunc)(
      void *, void *, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
  static HcclReduceVFunc func = nullptr;
  if (func == nullptr) {
    func = (HcclReduceVFunc)GET_FUNC(HcclReduce);
  }
  TORCH_CHECK(func, "Failed to find function ", "HcclReduce");
  auto ret = func(sendBuf, recvBuf, count, sendType, op, root, comm, stream);
  return ret;
}

HcclResult hcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError) {
    typedef HcclResult(*HcclGetCommAsyncErrorVFunc)(HcclComm, HcclResult*);
    static HcclGetCommAsyncErrorVFunc func = nullptr;
    if (func == nullptr) {
      func = (HcclGetCommAsyncErrorVFunc)GET_FUNC(HcclGetCommAsyncError);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclGetCommAsyncError");
    auto ret = func(comm, asyncError);
    return ret;
}

HcclResult hcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
    const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
    HcclComm comm, aclrtStream stream)
{
    typedef HcclResult(*HcclAlltoAllFunc)(
        const void *, uint64_t, HcclDataType,
        const void *, uint64_t, HcclDataType,
        HcclComm, aclrtStream);
    static HcclAlltoAllFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclAlltoAllFunc)GET_FUNC(HcclAlltoAll);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclAlltoAll");
    auto ret = func(sendBuf, sendCount, sendType,
                    recvBuf, recvCount, recvType, comm, stream);
    return ret;
}

HcclResult hcclScatter(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    typedef HcclResult(*HcclScatterVFunc)(void *, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    static HcclScatterVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclScatterVFunc)GET_FUNC(HcclScatter);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclScatter");
    auto ret = func(sendBuf, recvBuf, count, dataType, root, comm, stream);
    return ret;
}

HcclResult hcclBatchIsendIrecv(void* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    typedef HcclResult(*HcclBatchIsendIrecvVFunc)(
        void *, uint32_t, HcclComm, aclrtStream);
    static HcclBatchIsendIrecvVFunc func = nullptr;
    if (func == nullptr) {
        func = (HcclBatchIsendIrecvVFunc)GET_FUNC(HcclBatchSendRecv);
    }
    TORCH_CHECK(func, "Failed to find function ", "HcclBatchSendRecv");
    auto ret = func(sendRecvInfo, itemNum, comm, stream);
    return ret;
}

} // namespace c10d_npu

