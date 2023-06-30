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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor &NPUNativeOpApiFunctions::addmm_out(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnAddmm,
      NPUNativeFunctions::addmm_out(self, mat1, mat2, beta, alpha, result));
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnAddmm, self, mat1, mat2, beta, alpha, result, cube_math_type);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::addmm(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
  DO_COMPATIBILITY(aclnnAddmm,
      NPUNativeFunctions::addmm(self, mat1, mat2, beta, alpha));
  auto output_size = addmm_npu_output_size(self, mat1, mat2, beta, alpha);
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(output_size, self.options());
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnAddmm, self, mat1, mat2, beta, alpha, result, cube_math_type);

  return result;
}

at::Tensor &NPUNativeOpApiFunctions::addmm_(
    at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
  DO_COMPATIBILITY(aclnnInplaceAddmm,
      NPUNativeFunctions::addmm_(self, mat1, mat2, beta, alpha));
  c10::SmallVector<at::Tensor, N> inputs = {self, mat1, mat2};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnInplaceAddmm, self, mat1, mat2, beta, alpha, cube_math_type);

  return self;
}

} // namespace native
} // namespace at_npu
