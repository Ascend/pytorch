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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::renorm_out(const at::Tensor& self, const at::Scalar& p,
                                                int64_t dim, const at::Scalar& maxnorm, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnRenorm, NPUNativeFunctions::renorm_out(self, p, dim, maxnorm, result));

  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  auto output_size = input_same_output_size(self);
  OpPreparation::CheckOut(
      {self},
      result,
      result.scalar_type(),
      output_size);

  EXEC_NPU_CMD(aclnnRenorm, self, p, dim, maxnorm, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::renorm(const at::Tensor& self, const at::Scalar& p,
                                           int64_t dim, const at::Scalar& maxnorm) {
  DO_COMPATIBILITY(aclnnRenorm, NPUNativeFunctions::renorm(self, p, dim, maxnorm));

  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  // calculate the output size
  auto output_size = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result =
      OpPreparation::ApplyTensorWithoutFormat(output_size, self.options());

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnRenorm, self, p, dim, maxnorm, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::renorm_(at::Tensor& self, const at::Scalar& p,
                                             int64_t dim, const at::Scalar& maxnorm) {
  DO_COMPATIBILITY(aclnnInplaceRenorm, NPUNativeFunctions::renorm_(self, p, dim, maxnorm));

  dim = CalcuOpUtil::MakeWrapDim(dim, self.dim());
  EXEC_NPU_CMD(aclnnInplaceRenorm, self, p, dim, maxnorm);
  return self;
}

}  // namespace native
}  // namespace at_npu

