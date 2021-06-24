// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

IntArrayRef log_sigmoid_backward_npu_output_size(const Tensor& grad_output) {
  return input_same_output_size(grad_output);
}

SmallVector<NPUTensorDesc, N> log_sigmoid_backward_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> log_sigmoid_backward_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> log_sigmoid_backward_npu_attr(const Tensor& self) {
  SmallVector<NPUAttrDesc, N> attrs = {};
  return attrs;
}

Tensor& log_sigmoid_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  // constructs the input and output NPUTensorDesc
  auto inputs = log_sigmoid_backward_npu_input({grad_output, self});
  auto outputs = log_sigmoid_backward_npu_output({grad_input});

  // constructs the attr of the NPUAttrDesc
  auto attrs = log_sigmoid_backward_npu_attr(self);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("LogSigmoidGrad", inputs, outputs, attrs);

  return grad_input;
}

Tensor log_sigmoid_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  // calculate the output size
  auto outputSize = log_sigmoid_backward_npu_output_size(grad_output);

  // construct the output tensor of the NPU
  Tensor grad_input = at::empty_with_format(
      outputSize,
      grad_output.options(),
      CalcuOpUtil::get_tensor_npu_format(grad_output));

  // calculate the output result of the NPU
  log_sigmoid_backward_out_npu(grad_input, grad_output, self, buffer);

  return grad_input;
}

} // namespace native
} // namespace at