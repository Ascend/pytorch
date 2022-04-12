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
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor col2im_backward_out_npu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  SmallVector<int64_t, N> kernelSize = {1, kernel_size[0], kernel_size[1], 1};
  SmallVector<int64_t, N> stridesSize = {1, stride[0], stride[1], 1};
  SmallVector<int64_t, N> dilations = {1, dilation[0], dilation[1], 1};

  OpCommand cmd;
  cmd.Name("ExtractImagePatches")
      .Input(grad_output)
      .Output(grad_input)
      .Attr("ksizes", kernelSize)
      .Attr("strides", stridesSize)
      .Attr("padding", (string)"SAME")
      .Attr("dilations", dilations)
      .Run();
  return grad_input;
}

Tensor& col2im_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      grad_output);
  col2im_backward_out_npu_template(grad_input, grad_output, kernel_size, dilation, padding, stride);
  return grad_input;
}

Tensor col2im_backward_npu(
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor grad_input = OpPreparation::ApplyTensor(grad_output);
  col2im_backward_out_npu_template(grad_input, grad_output, kernel_size, dilation,padding,stride);
  return grad_input;
}

}
}
