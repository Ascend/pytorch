// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::reciprocal_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnReciprocal, NPUNativeFunctions::reciprocal_out(self, result));

  auto output_size = input_same_output_size(self);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      result.scalar_type(),
      output_size);

  EXEC_NPU_CMD(aclnnReciprocal, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::reciprocal(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnReciprocal, NPUNativeFunctions::reciprocal(self));
  // calculate the output size
  auto output_size = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result =
      OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());

  result = (result.dtype() == at::ScalarType::Half) ?
            result : NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Float);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnReciprocal, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::reciprocal_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceReciprocal, NPUNativeFunctions::reciprocal_(self));
  EXEC_NPU_CMD(aclnnInplaceReciprocal, self);
  return self;
}

}  // namespace native
}  // namespace at_npu

