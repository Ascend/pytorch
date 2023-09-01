// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {
  
at::Tensor NPUNativeOpApiFunctions::trace(const at::Tensor &self)
{
  DO_COMPATIBILITY(aclnnTrace, NPUNativeFunctions::trace(self));
  c10::SmallVector<int64_t, N> outputSize = {};
  auto out_dtype=(isIntegralType(self.scalar_type(), true)) ?
                 at::kLong : self.scalar_type();
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(out_dtype));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnTrace, self, result);
  return result;
}
} // namespace native
} // namespace at_npu

