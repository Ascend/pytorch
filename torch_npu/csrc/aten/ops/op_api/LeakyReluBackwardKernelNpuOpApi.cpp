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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::leaky_relu_backward_out(const at::Tensor& grad_output, const at::Tensor& self,
                                                             const at::Scalar& negval, bool is_result,
                                                             at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnLeakyReluBackward,
                   NPUNativeFunctions::leaky_relu_backward_out(grad_output, self, negval, is_result, grad_input));

  OpPreparation::CheckOut({self, grad_output}, grad_input, grad_input.scalar_type(), self.sizes());
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negval, is_result, grad_input);

  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::leaky_relu_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                                        const at::Scalar& negval, bool is_result) {
  DO_COMPATIBILITY(aclnnLeakyReluBackward,
                   NPUNativeFunctions::leaky_relu_backward(grad_output, self, negval, is_result));

  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negval, is_result, result);

  return result;
}
} // namespace native
} // namespace at_npu
