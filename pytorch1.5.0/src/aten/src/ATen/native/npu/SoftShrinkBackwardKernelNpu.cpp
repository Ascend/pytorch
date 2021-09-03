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

SmallVector<NPUTensorDesc, N> softshrink_backward_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> softshrink_backward_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> softshrink_backward_npu_attr(Scalar lambd) {
  float lambd_value = CalcuOpUtil::get_scalar_float_value(lambd);
  NPUAttrDesc npuAttrScalarLambd = NPUAttrDesc("lambd", lambd_value);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrScalarLambd};
  return attrs;
}

Tensor& softshrink_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd) {
  // constructs the input and output NPUTensorDesc
  auto inputs = softshrink_backward_npu_input({grad_output, self});
  auto outputs = softshrink_backward_npu_output({grad_input});

  // constructs the attr of the NPUAttrDesc
  auto attrs = softshrink_backward_npu_attr(lambd);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate(
      "SoftShrinkGrad", inputs, outputs, attrs);

  return grad_input;
}

Tensor softshrink_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar lambd) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor grad_input = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  softshrink_backward_out_npu(
      grad_input, grad_output, self, lambd);

  return grad_input;
}

}
}