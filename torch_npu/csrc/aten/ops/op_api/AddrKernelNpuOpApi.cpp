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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <ATen/native/TypeProperties.h>

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::addr_out(const at::Tensor &self, const at::Tensor &vec1, const at::Tensor &vec2,
                                              const at::Scalar &beta, const at::Scalar &alpha, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnAddr, NPUNativeFunctions::addr_out(self, vec1, vec2, beta, alpha, result));
  OpPreparation::CheckOut({self, vec1, vec2}, result, result);

  EXEC_NPU_CMD(aclnnAddr, self, vec1, vec2, beta, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::addr(const at::Tensor &self, const at::Tensor &vec1, const at::Tensor &vec2,
                                         const at::Scalar &beta, const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnAddr, NPUNativeFunctions::addr(self, vec1, vec2, beta, alpha));

  // calculate the output size
  at::ScalarType high_dtype = at::native::result_type({self, vec1, vec2});
  auto output_size = addr_npu_output_size(self, vec1, vec2, beta, alpha);

  // construct the output tensor of the NPU
  at::Tensor result =
      OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(high_dtype));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnAddr, self, vec1, vec2, beta, alpha, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::addr_(at::Tensor &self, const at::Tensor &vec1, const at::Tensor &vec2,
                                           const at::Scalar &beta, const at::Scalar &alpha) {
  DO_COMPATIBILITY(aclnnAddr, NPUNativeFunctions::addr_(self, vec1, vec2, beta, alpha));
  NPUNativeOpApiFunctions::addr_out(self, vec1, vec2, beta, alpha, self);
  return self;
}

}  // namespace native
}  // namespace at_npu