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

#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& log_sigmoid_backward_out_nocheck(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  OpCommand cmd;
  cmd.Name("LogSigmoidGrad")
     .Input(grad_output)
     .Input(self)
     .Output(grad_input)
     .Run();

  return grad_input;
}

Tensor& log_sigmoid_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  OpPreparation::CheckOut(
      {grad_output, self, buffer},
      grad_input,
      grad_output);
  
  if (!NpuUtils::check_match(&grad_input)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(grad_input);
    log_sigmoid_backward_out_nocheck(contiguousResult, grad_output, self, buffer);
    NpuUtils::format_fresh_view(grad_input, contiguousResult);
  } else {
    log_sigmoid_backward_out_nocheck(grad_input, grad_output, self, buffer);
  }

  return grad_input;
}

Tensor log_sigmoid_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& buffer) {
  // calculate the output size
  auto outputSize = input_same_output_size(grad_output);    
  
  // construct the output tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output, outputSize);

  // calculate the output result of the NPU
  log_sigmoid_backward_out_npu(grad_input, grad_output, self, buffer);

  return grad_input;
}

} // namespace native
} // namespace at