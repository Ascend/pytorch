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

struct ACL_DYNAMIC_PARAMS {
  ACL_DYNAMIC_PARAMS() {
    int input_num = 0;
    input_desc = nullptr;
    input_data_buf = nullptr;
    int output_num = 0;
    output_desc = nullptr;
    output_data_buf = nullptr;
    inputDims = nullptr;
    outputDims = nullptr;
    inputFormats = nullptr;
    outputFormats = nullptr;
    compile_input_desc = nullptr;
    compile_output_desc = nullptr;
    dynamicKey = "";
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
  const aclTensorDesc** compile_input_desc;
  const aclTensorDesc** compile_output_desc;
  bool hasAttr;
  std::string dynamicKey;
};

struct CONST_PARAMS {
  int constNum = 0;
  const int64_t** constList = nullptr;
  const int64_t* constIdx = nullptr;
  CONST_PARAMS() = default;
};

struct ExecuteParas {
  std::string opType;
  std::string opDynamicType;
  std::string attrInfo;
  bool isCompiling = false;
  bool isFuzzy = false;
  ACL_PARAMS paras;
  ACL_DYNAMIC_PARAMS dynamicParam;
  CONST_PARAMS constParams;
  const aclopAttr* attr = nullptr;
  const aclopAttr* dynamicCompileAttr = nullptr;
  const aclopAttr* dynamicRunAttr = nullptr;
  int64_t constIdx = -1;
  SmallVector<Tensor, N> hostMemory;
  ExecuteParas(
      std::string opName,
      std::string opDynamicName,
      aclopAttr* acl_attr,
      aclopAttr* acl_compile_attr,
      aclopAttr* acl_run_attr,
      const ACL_PARAMS& aclPars)
      : opType(opName),
        paras(aclPars),
        attr(acl_attr),
        dynamicCompileAttr(acl_compile_attr),
        dynamicRunAttr(acl_run_attr) {}
  ExecuteParas() = default;
  void Release();
  void DynamicRelease();
  void Copy(ExecuteParas& other);
};

struct CopyParas {
    void *dst = nullptr;
    size_t dstLen = 0;
    void *src = nullptr;
    size_t srcLen = 0;
    aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_HOST;
    SmallVector<Tensor, 1> pinMem;
    void Copy(CopyParas& other);
};

struct EventParas {
  aclrtEvent event = nullptr;
};

enum QueueParamType {
    COMPILE_AND_EXECUTE,
    ASYNC_MEMCPY,
    ASYNC_MEMCPY_EX,
    RECORD_EVENT
};

struct QueueParas {
  QueueParas(QueueParamType type, size_t len, void *val) : paramType(type), paramLen(len), paramVal(val) {}
  QueueParamType paramType = COMPILE_AND_EXECUTE;
  size_t paramLen = 0;
  void* paramVal = nullptr;
};

NPUStatus DestroyAclParams(ACL_PARAMS& params);
NPUStatus DestroyDynamicAclParams(ACL_DYNAMIC_PARAMS& params);
void DestroyConstParams(CONST_PARAMS& params);

} // namespace npu
} // namespace native
} // namespace at

#endif // __C10_NPU_NPUQUEUE_WITH_QUEUE__