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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& sigmoid_backward_out_npu_nocheck(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& output) {
  // output'format must be same with grad_output
  if (CalcuOpUtil::get_tensor_npu_format(output) != CalcuOpUtil::get_tensor_npu_format(grad_output)) {
    output.npu_format_cast_(CalcuOpUtil::get_tensor_npu_format(grad_output));
  }
  
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

Tensor& sigmoid_backward_out_npu(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& output) {
  OpPreparation::CheckOut({grad_output, output}, result, grad_output);  
  sigmoid_backward_out_npu_nocheck(result, grad_output, output);
  return result;
}

Tensor sigmoid_backward_npu(const Tensor& grad_output, const Tensor& output) {
  // construct the output tensor of the NPU
  Tensor grad_input = at::empty_with_format(
      grad_output.sizes(),
      grad_output.options(),
      CalcuOpUtil::get_tensor_npu_format(grad_output));

  // calculate the output result of the NPU
  sigmoid_backward_out_npu_nocheck(grad_input, grad_output, output);

  return grad_input;
}

} // namespace native
} // namespace at