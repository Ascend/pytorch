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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"


namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::mm(const at::Tensor &self,
                                       const at::Tensor &mat2) {
  DO_COMPATIBILITY(aclnnMatmul, NPUNativeFunctions::mm(self, mat2));
  auto output_size = {self.size(0), mat2.size(1)};
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnMm, self, mat2, result, cube_math_type);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::mm_out(const at::Tensor &self,
                                            const at::Tensor &mat2,
                                            at::Tensor &result){
  DO_COMPATIBILITY(aclnnMatmul, NPUNativeFunctions::mm_out(self, mat2, result));
  auto output_size = {self.size(0), mat2.size(1)};
  OpPreparation::CheckOut({self, mat2}, result, CalcuOpUtil::GetTensorNpuFormat(result), self.scalar_type(),
                          output_size);
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnMm, self, mat2, result, cube_math_type);
  return result;
}


} // namespace native
} // namespace at_npu
