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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::ger_out(const at::Tensor& self , const at::Tensor& vec2, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnGer, NPUNativeFunctions::ger_out(self, vec2, result));

  // calculate the output size
  auto output_size = ger_output_size(self, vec2);
  auto result_type = at::result_type(self, vec2);
  OpPreparation::CheckOut({self, vec2}, result, result_type, output_size);

  EXEC_NPU_CMD(aclnnGer, self, vec2, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::ger(const at::Tensor& self, const at::Tensor& vec2) {
  DO_COMPATIBILITY(aclnnGer, NPUNativeFunctions::ger(self, vec2));

  // calculate the output size
  auto output_size = ger_output_size(self, vec2);
  auto result_type = at::result_type(self, vec2);

  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(result_type));

  EXEC_NPU_CMD(aclnnGer, self, vec2, result);
  return result;
}
} // namespace native
} // namespace at_npu

