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

class DefaultStrategy : public DescStrategyBase
{
public:
  virtual void CreateInputDescInfo(ACL_PARAMS& params, 
    DynamicCompileShape& compileShape) override;

  virtual void CreateOutputDescInfo(ACL_PARAMS& params, 
    DynamicCompileShape& compileShape) override;
};

void DefaultStrategy::CreateInputDescInfo(ACL_PARAMS& params, 
  DynamicCompileShape& compileShape) {

  CreateDefaultDescInfo(params.input_desc,
    params.input_num,
    params.inputDims,
    params.inputFormats,
    compileShape.inputShape,
    compileShape.inputStorageShape);
}

void DefaultStrategy::CreateOutputDescInfo(ACL_PARAMS& params,
  DynamicCompileShape& compileShape) {

  CreateDefaultDescInfo(params.output_desc,
    params.output_num,
    params.outputDims,
    params.outputFormats,
    compileShape.outputShape,
    compileShape.outputStorageShape);
}

REGISTER_DYNAMIC_SHAPE_OPT(Default, DefaultStrategy)

} // namespace npu
} // namespace native
} // namespace at