// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::gelu_backward(const at::Tensor& grad, const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnGeluBackward, NPUNativeFunctions::gelu_backward(grad, self));
  // calculate the output size
  auto output_size = broadcast_ops_npu_output_size(grad, self);
  // dtype promotion
  auto output_dtype = at::native::result_type(grad, self);
  // construct the output tensor of the NPU
  at::Tensor grad_input = OpPreparation::ApplyTensor(output_size, self.options().dtype(output_dtype), self);
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnGeluBackward, grad, self, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu
