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
#include <c10/util/SmallVector.h>
#include <ATen/native/npu/dynamicstrategy/Strategy.h>
#include "ATen/native/npu/utils/NpuUtils.h"
#include <third_party/acl/inc/acl/acl_base.h>
#include <ATen/native/npu/frame/InputInfoLib.h>

namespace at {
namespace native {
namespace npu {

class TransDataStrategy : public DescStrategyBase
{
public:
  virtual void CreateInputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;

  virtual void CreateOutputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;
};

void TransDataStrategy::CreateInputDescInfo(ACL_PARAMS& params,
  DynamicCompileShape& compileShape) {
  // create input shape
  for (int64_t i = 0; i < params.input_num; ++i) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(params.input_desc[i]);
    int64_t dim = (int64_t)aclGetTensorDescNumDims(desc);
    int64_t* storagedims = params.inputDims;

    dim = (dim == 0) ? 1 : dim;
    int64_t storageDim = (storagedims[i] == 0) ? 1 : storagedims[i];
     
    FormatShape shape(dim, -1);
    
    aclFormat storageFormat = params.inputFormats[i];
    if (storageFormat == ACL_FORMAT_NC1HWC0 && dim == 5) {
      shape[4] = 16;
    } else if (storageFormat == ACL_FORMAT_FRACTAL_NZ) {
      shape[storageDim - 2] = 16;
      shape[storageDim - 1] = 16;
    }

    compileShape.inputShape.emplace_back(shape);
  }
}

void TransDataStrategy::CreateOutputDescInfo(ACL_PARAMS& params, 
  DynamicCompileShape& compileShape) {
  // create output shape
  for (int64_t i = 0; i < params.output_num; ++i) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(params.output_desc[i]);
    int64_t dim = (int64_t)aclGetTensorDescNumDims(desc);
    int64_t* storageDims = params.outputDims;

    dim = (dim == 0) ? 1 : dim;
    int64_t storageDim = (storageDims[i] == 0) ? 1 : storageDims[i];

    FormatShape storageShape(storageDim, -1);
    aclFormat storageFormat = params.outputFormats[i];
    if (storageFormat == ACL_FORMAT_NC1HWC0) {
      storageShape[4] = 16;
    } else if (storageFormat == ACL_FORMAT_FRACTAL_NZ) {
      storageShape[storageDim - 2] = 16;
      storageShape[storageDim - 1] = 16;
    }
    
    // transdata need ouput storageshape set shape
    compileShape.outputShape.emplace_back(storageShape);
  }
}

REGISTER_DYNAMIC_SHAPE_OPT(TransData, TransDataStrategy)

} // namespace npu
} // namespace native
} // namespace at