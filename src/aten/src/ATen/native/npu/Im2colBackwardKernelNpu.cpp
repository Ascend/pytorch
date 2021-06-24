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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& im2col_backward_out_npu_nocheck(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor gradOutput = grad_output;
  gradOutput = gradOutput.view({
    grad_output.size(0),
    grad_output.size(1) / (kernel_size[0] * kernel_size[1]),
    kernel_size[0] * kernel_size[1],
    grad_output.size(2)});

  SmallVector<int64_t, N> inputSize = {input_size[0], input_size[1]};  

  SmallVector<int64_t, N> kernelSize = {kernel_size[0], kernel_size[1]};
  SmallVector<int64_t, N> dilations = {dilation[0], dilation[1]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[1]};
  SmallVector<int64_t, N> stridesSize = {stride[0], stride[1]};
  
  OpCommand cmd;
  cmd.Name("Col2im")
      .Input(gradOutput)
      .Input(inputSize, at::kInt)
      .Output(grad_input)
      .Attr("kernel_size", kernelSize)
      .Attr("dilation", dilations)
      .Attr("padding", paddings)
      .Attr("stride", stridesSize)
      .Run();

  return grad_input;
}

Tensor& im2col_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  SmallVector<int64_t, SIZE> outputSize = {
    grad_output.size(0),
    grad_output.size(1) / (kernel_size[0] * kernel_size[1]),
    input_size[0],
    input_size[1]};

  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      CalcuOpUtil::get_tensor_npu_format(grad_output),
      grad_output.scalar_type(),
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({grad_output}, {grad_input})
    .Func([&grad_output, &input_size, &kernel_size, &dilation, &padding, &stride]
    (Tensor& grad_input)
    {im2col_backward_out_npu_nocheck(
      grad_input,
      grad_output,
      input_size,
      kernel_size,
      dilation,
      padding,
      stride);})
    .Call(grad_input);
}

Tensor im2col_backward_npu(
    const Tensor& grad_output,
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = {
    grad_output.size(0),
    grad_output.size(1) / (kernel_size[0] * kernel_size[1]),
    input_size[0],
    input_size[1]};

  // construct the input tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output, outputSize);

  im2col_backward_out_npu_nocheck(
      grad_input,
      grad_output,
      input_size,
      kernel_size,
      dilation,
      padding,
      stride);

  return grad_input;
}
} // namespace native
} // namespace at
