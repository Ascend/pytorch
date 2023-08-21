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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {
at::Tensor &NPUNativeOpApiFunctions::softshrink_backward_out(const at::Tensor &grad_output, const at::Tensor &self,
                                                             const at::Scalar &lambd, at::Tensor &grad_input) {
  DO_COMPATIBILITY(aclnnSoftShrinkBackward,
                   NPUNativeFunctions::softshrink_backward_out(grad_output, self, lambd, grad_input));
  auto output_size = broadcast_ops_npu_output_size(grad_output, self);
  OpPreparation::CheckOut({grad_output, self}, grad_input, grad_input.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnSoftShrinkBackward, grad_output, self, lambd, grad_input);
  return grad_input;
}

at::Tensor NPUNativeOpApiFunctions::softshrink_backward(const at::Tensor &grad_output, const at::Tensor &self,
                                                        const at::Scalar &lambd) {
  DO_COMPATIBILITY(aclnnSoftShrinkBackward, NPUNativeFunctions::softshrink_backward(grad_output, self, lambd));
  at::ScalarType result_dtype = at::native::result_type(grad_output, self);
  auto output_size = broadcast_ops_npu_output_size(grad_output, self);
  at::Tensor grad_input = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(result_dtype));
  EXEC_NPU_CMD(aclnnSoftShrinkBackward, grad_output, self, lambd, grad_input);
  return grad_input;
}

}  // namespace native
}  // namespace at_npu
