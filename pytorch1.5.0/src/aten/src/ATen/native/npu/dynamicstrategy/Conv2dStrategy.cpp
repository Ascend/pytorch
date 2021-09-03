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

class Conv2dStrategy : public DescStrategyBase
{
public:
  virtual void CreateInputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;

  virtual void CreateOutputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;
};

// create input shape
void Conv2dStrategy::CreateInputDescInfo(ACL_PARAMS& params, 
    DynamicCompileShape& compileShape) {
  for (int64_t i = 0; i < params.input_num; ++i) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(params.input_desc[i]);
    int64_t dim = (int64_t)aclGetTensorDescNumDims(desc);
    int64_t storageDim = params.inputDims[i];
    aclFormat format = aclGetTensorDescFormat(desc);
    aclFormat storageFormat = params.inputFormats[i];

    FormatShape shape(dim, -1);
    FormatShape storageShape(storageDim, -1);
    // the dynamicshape function of conv2D requires x must fix C dim, weight and bias must fix all dims.
    if (i == 0) {
      if (format == ACL_FORMAT_NCHW){
        aclGetTensorDescDimV2(desc, 1, &shape[1]);
      }
      storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);  
    } else {
      for (int64_t i = 0; i < dim; ++i) {
        aclGetTensorDescDimV2(desc, i, &shape[i]);
      }
      storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);
    } 

    compileShape.inputShape.emplace_back(shape);
    compileShape.inputStorageShape.emplace_back(storageShape);
  }
}

void Conv2dStrategy::CreateOutputDescInfo(ACL_PARAMS& params, 
    DynamicCompileShape& compileShape) { 
  // create output shape
  aclTensorDesc* desc = const_cast<aclTensorDesc*>(params.output_desc[0]);
  int64_t dim = (int64_t)aclGetTensorDescNumDims(desc);
  int64_t storageDim = params.outputDims[0];
  aclFormat format = aclGetTensorDescFormat(desc);
  aclFormat storageFormat = params.outputFormats[0];

  FormatShape shape(dim, -1);
  FormatShape storageShape(storageDim, -1);
  // the dynamicshape function of conv2D requires y must fix C dim.
  if (format == ACL_FORMAT_NCHW) {
    aclGetTensorDescDimV2(desc, 1, &shape[1]);
    storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);
  }

  compileShape.outputShape.emplace_back(shape);
  compileShape.outputStorageShape.emplace_back(storageShape);
}

REGISTER_DYNAMIC_SHAPE_OPT(Conv2D, Conv2dStrategy)

} // namespace npu
} // namespace native
} // namespace at