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

#ifndef __C10_NPU_NPUQUEUE_WITH_QUEUE__
#define __C10_NPU_NPUQUEUE_WITH_QUEUE__

#include "c10/npu/npu_log.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
namespace npu {

struct ACL_PARAMS {
  ACL_PARAMS() {
    input_num = 0;
    input_desc = nullptr;
    input_data_buf = nullptr;
    output_num = 0;
    output_desc = nullptr;
    output_data_buf = nullptr;
    inputDims = nullptr;
    outputDims = nullptr;
    inputFormats = nullptr;
    outputFormats = nullptr;
    hasAttr = false;
  }

  int input_num;
  const aclTensorDesc** input_desc;
  const aclDataBuffer** input_data_buf;
  int output_num;
  const aclTensorDesc** output_desc;
  aclDataBuffer** output_data_buf;
  int64_t* inputDims;
  int64_t* outputDims;
  aclFormat* inputFormats;
  aclFormat* outputFormats;
  bool hasAttr;
};

struct ExecuteParas {
  std::string opType;
  std::string attrInfo;
  bool isCompiling = false;
  bool isFuzzy = false;
  ACL_PARAMS paras;
  const aclopAttr* attr = nullptr;
  SmallVector<Storage, N> hostMemory;
  ExecuteParas(
      std::string opName,
      aclopAttr* acl_attr,
      const ACL_PARAMS& aclPars)
      : opType(opName),
        paras(aclPars),
        attr(acl_attr) {}
  ExecuteParas() = default;
  void Release();
  void Copy(ExecuteParas& other);
  void CopyEx(ExecuteParas& other);
};

NPUStatus DestroyAclParams(ACL_PARAMS& params);

} // namespace npu
} // namespace native
} // namespace at

#endif // __C10_NPU_NPUQUEUE_WITH_QUEUE__