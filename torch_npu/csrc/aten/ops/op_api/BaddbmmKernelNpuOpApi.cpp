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

at::Tensor &NPUNativeOpApiFunctions::baddbmm_out(
    const at::Tensor &self,
    const at::Tensor &batch1,
    const at::Tensor &batch2,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &result)
{
  DO_COMPATIBILITY(aclnnBaddbmm,
      NPUNativeFunctions::baddbmm_out(self, batch1, batch2, beta, alpha, result));

  auto output_size = baddbmm_npu_output_size(batch1, batch2);
  OpPreparation::CheckOut(
      {self, batch1, batch2},
      result,
      self.scalar_type(),
      output_size);
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnBaddbmm, self, batch1, batch2, beta, alpha, result, cube_math_type);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::baddbmm(
    const at::Tensor &self,
    const at::Tensor &batch1,
    const at::Tensor &batch2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
  DO_COMPATIBILITY(aclnnBaddbmm,
      NPUNativeFunctions::baddbmm(self, batch1, batch2, beta, alpha));
  
  auto output_size = baddbmm_npu_output_size(batch1, batch2);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());

  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnBaddbmm, self, batch1, batch2, beta, alpha, result, cube_math_type);

  return result;
}

at::Tensor &NPUNativeOpApiFunctions::baddbmm_(
    at::Tensor &self,
    const at::Tensor &batch1,
    const at::Tensor &batch2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
  DO_COMPATIBILITY(aclnnBaddbmm,
      NPUNativeFunctions::baddbmm_(self, batch1, batch2, beta, alpha));

  NPUNativeOpApiFunctions::baddbmm_out(self, batch1, batch2, beta, alpha, self);
  return self;
}

} // namespace native
} // namespace at_npu

