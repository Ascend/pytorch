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

class ROIAlignBackwardStrategy : public DescStrategyBase
{
public:
  virtual void CreateInputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;

  virtual void CreateOutputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;
};

void ROIAlignBackwardStrategy::CreateInputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) {
  CreateDefaultDescInfo(params.input_desc,
    params.input_num,
    params.inputDims,
    params.inputFormats,
    compileShape.inputShape,
    compileShape.inputStorageShape);
}

void ROIAlignBackwardStrategy::CreateOutputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) {
  for (int64_t i = 0; i < params.output_num; ++i) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(params.output_desc[i]);

    int64_t dim = (int64_t)aclGetTensorDescNumDims(desc);
    dim = (dim == 0) ? 1 : dim;

    int64_t storageDim = (params.outputDims[i] == 0) ? 1 : params.outputDims[i];
    aclFormat storageFormat = params.outputFormats[i];

    FormatShape shape(dim, -1);
    FormatShape storageShape(storageDim, -1);  

    // fix all dims
    aclGetTensorDescDimV2(desc, 0, &shape[0]);
    aclGetTensorDescDimV2(desc, 1, &shape[1]);
    aclGetTensorDescDimV2(desc, 2, &shape[2]);
    aclGetTensorDescDimV2(desc, 3, &shape[3]);

    storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);
    
    compileShape.outputShape.emplace_back(shape);
    compileShape.outputStorageShape.emplace_back(storageShape);
  }
}

REGISTER_DYNAMIC_SHAPE_OPT(ROIAlignGrad, ROIAlignBackwardStrategy)

} // namespace npu
} // namespace native
} // namespace at