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

class LayerNormGardStrategy : public DescStrategyBase
{
public:
  virtual void CreateInputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;

  virtual void CreateOutputDescInfo(ACL_PARAMS& params,
    DynamicCompileShape& compileShape) override;
};

// create input shape
void LayerNormGardStrategy::CreateInputDescInfo(ACL_PARAMS& params, 
    DynamicCompileShape& compileShape) {
  for (int64_t i = 0; i < params.input_num; ++i) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(params.input_desc[i]);
    aclFormat storageFormat = params.inputFormats[i];
    if (i < 2) {
      FormatShape shape = {-1, -1, -1};
      aclGetTensorDescDimV2(desc, 2, &shape[2]);
      FormatShape storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);
      compileShape.inputShape.emplace_back(shape);
      compileShape.inputStorageShape.emplace_back(storageShape);
    } else if (i == 2 || i == 3) {
      FormatShape shape = {-1, -1, 1};
      FormatShape storageShape = {-1, -1, 1};
      compileShape.inputShape.emplace_back(shape);
      compileShape.inputStorageShape.emplace_back(storageShape);
    } else {
      FormatShape shape = {-1};
      aclGetTensorDescDimV2(desc, 0, &shape[0]);
      FormatShape storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);
      compileShape.inputShape.emplace_back(shape);
      compileShape.inputStorageShape.emplace_back(storageShape);
    }
  }
}

void LayerNormGardStrategy::CreateOutputDescInfo(ACL_PARAMS& params, 
    DynamicCompileShape& compileShape) { 
  // create output shape
  for (int64_t i = 0; i < params.output_num; ++i) {
    aclTensorDesc* desc = const_cast<aclTensorDesc*>(params.output_desc[i]);
    aclFormat storageFormat = params.inputFormats[i];
    if (i == 0) {
      FormatShape shape = {-1, -1, -1};
      aclGetTensorDescDimV2(desc, 2, &shape[2]);
      FormatShape storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);
      compileShape.outputShape.emplace_back(shape);
      compileShape.outputStorageShape.emplace_back(storageShape);
    } else {
      FormatShape shape = {-1};
      aclGetTensorDescDimV2(desc, 0, &shape[0]);
      FormatShape storageShape = FormatHelper::GetStorageSizes(storageFormat, shape);
      compileShape.outputShape.emplace_back(shape);
      compileShape.outputStorageShape.emplace_back(storageShape);
    }
  }
}
REGISTER_DYNAMIC_SHAPE_OPT(LayerNormGrad, LayerNormGardStrategy)

} // namespace npu
} // namespace native
} // namespace at