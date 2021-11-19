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

#include "ATen/native/npu/frame/NPUDefine.h"
#include <c10/npu/NPUException.h>

namespace at {
namespace native {
namespace npu {

void ExecuteParas::Release() {
  // if useDynamicCompile, this attr will be freed in dynamic compile. 
  if (!isCompiling) {
    aclopDestroyAttr(attr);
  }  
  DestroyConstParams(constParams);
  NPUStatus ret = DestroyAclParams(paras);
  if (ret != SUCCESS) {
    NPU_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
  }
  return;
}

void ExecuteParas::DynamicRelease() {
  // if useDynamicCompile, this attr will be freed in dynamic compile. 
  if (!isCompiling) {
    if (dynamicCompileAttr != nullptr) {
      aclopDestroyAttr(dynamicCompileAttr);
    }

    NPUStatus ret = DestroyDynamicAclParams(dynamicParam);
    if (ret != SUCCESS) {
      NPU_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
    }
  }
  return;
}

void ExecuteParas::Copy(ExecuteParas& other) {
  this->opType = other.opType;
  this->attrInfo = other.attrInfo;
  this->paras = other.paras;
  this->attr = other.attr;
  this->constParams = other.constParams;
  if (other.opDynamicType != "") {
    this->opDynamicType = other.opDynamicType;
    this->dynamicCompileAttr = other.dynamicCompileAttr;
    this->dynamicRunAttr = other.dynamicRunAttr;
    this->dynamicParam = other.dynamicParam;
  }
  this->hostMemory = other.hostMemory;
  this->isFuzzy = other.isFuzzy;
}

void ExecuteParas::CopyEx(ExecuteParas& other)
{
  this->paras = other.paras;
  this->attr = other.attr;
  this->constParams = other.constParams;
  if (other.opDynamicType != "") {
    this->dynamicCompileAttr = other.dynamicCompileAttr;
    this->dynamicRunAttr = other.dynamicRunAttr;
    this->dynamicParam = other.dynamicParam;
  }
}

NPUStatus DestroyAclParams(ACL_PARAMS& params) {
  if (params.input_num != 0) {
    if (params.input_desc != nullptr) {
      for (int i = 0; i < params.input_num; ++i) {
        aclDestroyTensorDesc(params.input_desc[i]);
      }
      delete[] params.input_desc;
      params.input_desc = nullptr;
    }
    if (params.inputDims != nullptr) {
      delete[] params.inputDims;
      params.inputDims = nullptr;
    }
    if (params.inputFormats != nullptr) {
      delete[] params.inputFormats;
      params.inputFormats = nullptr;
    }
    if (params.input_data_buf != nullptr) {
      for (int i = 0; i < params.input_num; ++i) {
        C10_NPU_CHECK(aclDestroyDataBuffer(params.input_data_buf[i]));
      }
      delete[] params.input_data_buf;
      params.input_data_buf = nullptr;
    }
    params.input_num = 0;
  }
  if (params.output_num != 0) {
    if (params.output_desc != nullptr) {
      for (int i = 0; i < params.output_num; ++i) {
        aclDestroyTensorDesc(params.output_desc[i]);
      }
      delete[] params.output_desc;
      params.output_desc = nullptr;
    }
    if (params.outputDims != nullptr) {
      delete[] params.outputDims;
      params.outputDims = nullptr;
    }
    if (params.outputFormats != nullptr) {
      delete[] params.outputFormats;
      params.outputFormats = nullptr;
    }

    if (params.output_data_buf != nullptr) {
      for (int i = 0; i < params.output_num; ++i) {
        C10_NPU_CHECK(aclDestroyDataBuffer(params.output_data_buf[i]));
      }
      delete[] params.output_data_buf;
      params.output_data_buf = nullptr;
    }
    params.output_num = 0;
  }
  return SUCCESS;
}

NPUStatus DestroyDynamicAclParams(ACL_DYNAMIC_PARAMS& params) {
  if (params.input_num != 0) {
    if (params.input_desc != nullptr) {
      for (int i = 0; i < params.input_num; ++i) {
        aclDestroyTensorDesc(params.input_desc[i]);
      }
      delete[] params.input_desc;
      params.input_desc = nullptr;
    }

    if (params.compile_input_desc != nullptr) {
      for (int i = 0; i < params.input_num; ++i) {
        aclDestroyTensorDesc(params.compile_input_desc[i]);
      }
      delete[] params.compile_input_desc;
      params.compile_input_desc = nullptr;
    }

    if (params.inputDims != nullptr) {
      delete[] params.inputDims;
      params.inputDims = nullptr;
    }
    
    if (params.inputFormats != nullptr) {
      delete[] params.inputFormats;
      params.inputFormats = nullptr;
    }

    params.input_num = 0;
  }
  if (params.output_num != 0) {
    if (params.output_desc != nullptr) {
      for (int i = 0; i < params.output_num; ++i) {
        aclDestroyTensorDesc(params.output_desc[i]);
      }
      delete[] params.output_desc;
      params.output_desc = nullptr;
    }

    if (params.compile_output_desc != nullptr) {
      for (int i = 0; i < params.output_num; ++i) {
        aclDestroyTensorDesc(params.compile_output_desc[i]);
      }
      delete[] params.compile_output_desc;
      params.compile_output_desc = nullptr;
    }

    if (params.outputDims != nullptr) {
      delete[] params.outputDims;
      params.outputDims = nullptr;
    }
    if (params.outputFormats != nullptr) {
      delete[] params.outputFormats;
      params.outputFormats = nullptr;
    }

    params.output_num = 0;
  }
  return SUCCESS;
}

void DestroyConstParams(CONST_PARAMS& params) { 
  if (params.constList != nullptr) {
    for (int i = 0; i < params.constNum; ++i) {
      if (params.constList[i] != nullptr) {
        delete[] params.constList[i];
      }    
    }
    delete[] params.constList;
    params.constList = nullptr;
  }

  if (params.constIdx != nullptr) {
    delete[] params.constIdx;
    params.constIdx = nullptr;
  }
}

} // namespace npu
} // namespace native
} // namespace at