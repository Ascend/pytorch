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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& sigmoid_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& output) {
  auto unified_result = OpPreparation::binary_op_check(result, output, grad_output, true);
  OpCommand cmd;
  cmd.Name("SigmoidGrad")
    .Expect(unified_result)
    .Input(output)
    .Input(grad_output)
    .Output(result)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::sigmoid_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    at::Tensor& result) {
  OpPreparation::CheckOut({grad_output, output}, result, grad_output);
  sigmoid_backward_out_npu_nocheck(result, grad_output, output);
  return result;
}

at::Tensor NPUNativeFunctions::sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& output) {
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  sigmoid_backward_out_npu_nocheck(grad_input, grad_output, output);

  return grad_input;
}

} // namespace native
} // namespace at_npu