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

#include "torch_npu/csrc/core/npu/NPUException.h"

#include "torch_npu/csrc/framework/NPUDefine.h"

namespace at_npu
{
  namespace native
  {

    void ExecuteParas::Release()
    {
      // if useDynamicCompile, this attr will be freed in dynamic compile.
      if (attr != nullptr)
      {
        aclopDestroyAttr(attr);
      }
      DestroyConstParams(constParams);
      NPUStatus ret = DestroyAclParams(paras);
      if (ret != SUCCESS)
      {
        NPU_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
      }
      hostMemory.clear();
      return;
    }

    void ExecuteParas::Copy(ExecuteParas &other)
    {
      strncpy(this->opType, other.opType, sizeof(ExecuteParas::opType) - 1);
      this->paras = other.paras;
      this->attr = other.attr;
      this->constParams = other.constParams;
      this->hostMemory = other.hostMemory;
      this->isFuzzy = other.isFuzzy;
      this->isDataPreprocessOp = other.isDataPreprocessOp;
    }

    void ExecuteParas::CopyEx(ExecuteParas& other)
    {
      this->paras = other.paras;
      this->attr = other.attr;
      this->constParams = other.constParams;
    }

    NPUStatus DestroyAclParams(ACL_PARAMS& params)
    {
      if (params.input_num != 0) {
        if (params.input_desc != nullptr) {
          for (int i = 0; i < params.input_num; ++i) {
            aclDestroyTensorDesc(params.input_desc[i]);
          }
        }
        if (params.input_data_buf != nullptr) {
          for (int i = 0; i < params.input_num; ++i) {
            C10_NPU_CHECK(aclDestroyDataBuffer(params.input_data_buf[i]));
          }
        }
        params.input_num = 0;
      }
      if (params.output_num != 0)
      {
        if (params.output_desc != nullptr)
        {
          for (int i = 0; i < params.output_num; ++i)
          {
            aclDestroyTensorDesc(params.output_desc[i]);
          }
        }
        if (params.output_data_buf != nullptr)
        {
          for (int i = 0; i < params.output_num; ++i) {
            C10_NPU_CHECK(aclDestroyDataBuffer(params.output_data_buf[i]));
          }
        }
        params.output_num = 0;
      }
      free(params.input_desc);
      params.input_desc = nullptr;
      params.input_data_buf = nullptr;
      params.output_desc = nullptr;
      params.output_data_buf = nullptr;
      return SUCCESS;
    }

    void DestroyConstParams(CONST_PARAMS &params)
    {
      if (params.constList != nullptr)
      {
        for (int i = 0; i < params.constNum; ++i)
        {
          if (params.constList[i] != nullptr) {
            delete[] params.constList[i];
          }
        }
      }
      params.constList = nullptr;
      params.constIdx = nullptr;
    }
  } // namespace native
} // namespace at_npu