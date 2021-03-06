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
}
