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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::sigmoid_backward_out(const at::Tensor& grad_output, const at::Tensor& output,
                                                          at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSigmoidBackward, NPUNativeFunctions::sigmoid_backward_out(grad_output, output, result));
  OpPreparation::CheckOut({grad_output, output}, result, grad_output);

  EXEC_NPU_CMD(aclnnSigmoidBackward, grad_output, output, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& output) {
  DO_COMPATIBILITY(aclnnSigmoidBackward, NPUNativeFunctions::sigmoid_backward(grad_output, output));
  at::Tensor grad_input = OpPreparation::ApplyTensorWithoutFormat(grad_output);

  EXEC_NPU_CMD(aclnnSigmoidBackward, grad_output, output, grad_input);
  return grad_input;
}

}  // namespace native
}  // namespace at_npu
