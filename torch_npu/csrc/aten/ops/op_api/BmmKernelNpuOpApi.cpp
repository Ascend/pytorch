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
// limitations under the License.、
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::bmm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnBatchMatMul,
      NPUNativeFunctions::bmm_out(self, mat2, result));

  auto output_size = {self.size(0), self.size(1), mat2.size(2)};
  OpPreparation::CheckOut({self, mat2}, result, CalcuOpUtil::GetTensorNpuFormat(result),
      self.scalar_type(), output_size);

  // cube_math_type, an enumeration value of type int8 that determines which calculation logic the CUBE unit should use
  // and functions such as hfloat32 can be enabled through this switch
  int cube_math_type = 1;
  EXEC_NPU_CMD(aclnnBatchMatMul, self, mat2, result, cube_math_type);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::bmm(const at::Tensor& self, const at::Tensor& mat2) {
  DO_COMPATIBILITY(aclnnBatchMatMul,
      NPUNativeFunctions::bmm(self, mat2));

  // calculate the output size
  auto output_size = {self.size(0), self.size(1), mat2.size(2)};
  
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(
      output_size, self.options());

  // cube_math_type, an enumeration value of type int8 that determines which calculation logic the CUBE unit should use
  // and functions such as hfloat32 can be enabled through this switch
  int cube_math_type = 1;
  EXEC_NPU_CMD(aclnnBatchMatMul, self, mat2, result, cube_math_type);
  return result;
}

} // namespace native
} // namespace at_npu

