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

#ifndef __PULGIN_C10_NPUQUEUE_WITH_QUEUE__
#define __PULGIN_C10_NPUQUEUE_WITH_QUEUE__

#include "torch_npu/csrc/core/npu/npu_log.h"

#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace at_npu
{
  namespace native
  {

    struct ACL_PARAMS
    {
      ACL_PARAMS()
      {
        input_desc = nullptr;
        input_data_buf = nullptr;
        output_desc = nullptr;
        output_data_buf = nullptr;
      }

      int input_num;
      const aclTensorDesc **input_desc;
      const aclDataBuffer **input_data_buf;
      int output_num;
      const aclTensorDesc **output_desc;
      aclDataBuffer **output_data_buf;
    };

    struct ACL_DYNAMIC_PARAMS
    {
      ACL_DYNAMIC_PARAMS()
      {
        input_desc = nullptr;
        input_data_buf = nullptr;
        output_desc = nullptr;
        output_data_buf = nullptr;
        inputDims = nullptr;
        outputDims = nullptr;
        inputFormats = nullptr;
        outputFormats = nullptr;
        compile_input_desc = nullptr;
        compile_output_desc = nullptr;

        hasAttr = false;
      }

      int input_num;
      const aclTensorDesc **input_desc;
      const aclDataBuffer **input_data_buf;
      int output_num;
      const aclTensorDesc **output_desc;
      aclDataBuffer **output_data_buf;
      int64_t *inputDims;
      int64_t *outputDims;
      aclFormat *inputFormats;
      aclFormat *outputFormats;
      const aclTensorDesc **compile_input_desc;
      const aclTensorDesc **compile_output_desc;
      bool hasAttr;
      std::string dynamicKey;
    };

    struct CONST_PARAMS
    {
      int constNum = 0;
      const int64_t **constList = nullptr;
      const int64_t *constIdx = nullptr;
      CONST_PARAMS() = default;
    };

    struct ExecuteParas
    {
      std::string opType;
      bool isFuzzy = false;
      ACL_PARAMS paras;
      CONST_PARAMS constParams;
      const aclopAttr *attr;
      int64_t constIdx = -1;
      c10::SmallVector<at::Tensor, N> hostMemory;
      ExecuteParas(
          std::string opName,
          aclopAttr *acl_attr,
          aclopAttr *acl_compile_attr,
          aclopAttr *acl_run_attr,
          const ACL_PARAMS &aclPars)
          : opType(opName),
            paras(aclPars),
            attr(acl_attr) {}
      ExecuteParas() = default;
      void Release();
      void Copy(ExecuteParas &other);
      void CopyEx(ExecuteParas& other);
    };

    NPUStatus DestroyAclParams(ACL_PARAMS &params);
    void DestroyConstParams(CONST_PARAMS &params);
  } // namespace native
} // namespace at_npu

#endif // __C10_NPU_NPUQUEUE_WITH_QUEUE__