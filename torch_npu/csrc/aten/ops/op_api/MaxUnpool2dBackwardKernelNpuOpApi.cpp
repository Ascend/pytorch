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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::max_unpool2d_backward_out(const at::Tensor& gradOutput, const at::Tensor& self,
                                                               const at::Tensor& indices, at::IntArrayRef outputSize,
                                                               at::Tensor& gradInput) {
  DO_COMPATIBILITY(aclnnMaxUnpool2dBackward,
                   NPUNativeFunctions::max_unpool2d_backward_out(gradOutput, self, indices, outputSize, gradInput));
  OpPreparation::CheckOut({self, gradOutput}, gradInput, self);
  EXEC_NPU_CMD(aclnnMaxUnpool2dBackward, gradOutput, self, indices, outputSize, gradInput);
  return gradInput;
}

at::Tensor NPUNativeOpApiFunctions::max_unpool2d_backward(const at::Tensor& gradOutput, const at::Tensor& self,
                                                          const at::Tensor& indices, at::IntArrayRef outputSize) {
  DO_COMPATIBILITY(aclnnMaxUnpool2dBackward,
                   NPUNativeFunctions::max_unpool2d_backward(gradOutput, self, indices, outputSize));
  at::Tensor gradInput = OpPreparation::ApplyTensor(self);
  max_unpool2d_backward_out(gradOutput, self, indices, outputSize, gradInput);
  return gradInput;
}
}  // namespace native
}  // namespace at_npu
