// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::bmm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& result) {
  // input's dtype is int32, and bin kernel do not support int32, so use path 3
  if (self.dtype() == at::ScalarType::Int || mat2.dtype() == at::ScalarType::Int ||
      result.dtype() == at::ScalarType::Int) {
    NPUNativeFunctions::bmm_out(self, mat2, result);
    return result;
  }

  // input's dtype is fp16 or fp32, and use path5(host api)
  int cube_math_type = 1;
  EXEC_NPU_CMD(aclnnBatchMatMul, self, mat2, result, cube_math_type);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::bmm(const at::Tensor& self, const at::Tensor& mat2) {
  // calculate the output size
  auto output_size = {self.size(0), self.size(1), mat2.size(2)};
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      output_size, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

   // input's dtype is int32, and bin kernel do not support int32, so use path 3
  if (self.dtype() == at::ScalarType::Int || mat2.dtype() == at::ScalarType::Int ||
      result.dtype() == at::ScalarType::Int) {
    NPUNativeFunctions::bmm_out(self, mat2, result);
    return result;
  }

  // calculate the output result of the NPU
  int cube_math_type = 1;
  EXEC_NPU_CMD(aclnnBatchMatMul, self, mat2, result, cube_math_type);
  return result;
}

} // namespace native
} // namespace at_npu