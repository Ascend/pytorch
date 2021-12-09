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

Tensor slow_conv_transpose2d_backward_grad_output_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns,
    const Tensor& ones) {
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Conv2D")
      .Input(grad_output)
      .Input(weight)
      .Output(grad_input)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return grad_input;
}

Tensor slow_conv_transpose2d_backward_weight_out_npu(
    Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns,
    const Tensor& ones) {
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  SmallVector<int64_t, N> dimList = array_to_small_vector(weight.sizes());
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Conv2DBackpropFilter")
      .Input(grad_output)
      .Input(dimList, at::kInt)
      .Input(self)
      .Output(grad_weight)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return grad_weight;
}

Tensor slow_conv_transpose2d_backward_bias_out_npu(
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns,
    const Tensor& ones) {
  Tensor gradView = grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), -1});
  at::sum_out(grad_bias, gradView, SmallVector<int64_t, N>{0, 2});

  return grad_bias;
}

tuple<Tensor&, Tensor&, Tensor&> slow_conv_transpose2d_backward_npu_nocheck(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns,
    const Tensor& ones,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  slow_conv_transpose2d_backward_grad_output_out_npu(
      grad_input,
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation,
      columns,
      ones);

  slow_conv_transpose2d_backward_weight_out_npu(
      grad_weight,
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation,
      columns,
      ones);
  slow_conv_transpose2d_backward_bias_out_npu(
      grad_bias,
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation,
      columns,
      ones);
  return tuple<Tensor&, Tensor&, Tensor&>(grad_input, grad_weight, grad_bias);
}

tuple<Tensor&, Tensor&, Tensor&> slow_conv_transpose2d_backward_out_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns,
    const Tensor& ones,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  auto outputSizes = slow_conv_transpose2d_backward_npu_output_size(
      grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);

  OpPreparation::CheckOut(
      {grad_output, self, weight},
      grad_input,
      ACL_FORMAT_NC1HWC0,
      self.scalar_type(),
      std::get<0>(outputSizes));
  OpPreparation::CheckOut(
      {grad_output, self, weight},
      grad_weight,
      ACL_FORMAT_FRACTAL_Z,
      at::kFloat,
      std::get<1>(outputSizes));
  OpPreparation::CheckOut(
      {grad_output, self, weight},
      grad_bias,
      ACL_FORMAT_NCHW,
      grad_output.scalar_type(),
      {grad_output.size(1)});

  return slow_conv_transpose2d_backward_npu_nocheck(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation,
      columns,
      ones,
      grad_input,
      grad_weight,
      grad_bias);
}

tuple<Tensor,Tensor,Tensor> slow_conv_transpose2d_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    const Tensor& columns,
    const Tensor& ones,
    std::array<bool, 3> output_mask) {
  auto outputSizes = slow_conv_transpose2d_backward_npu_output_size(
      grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = OpPreparation::ApplyTensorWithFormat(self, std::get<0>(outputSizes), ACL_FORMAT_NC1HWC0);
  }

  if (output_mask[1]) {
    grad_weight = OpPreparation::ApplyTensorWithFormat(std::get<1>(outputSizes), weight.options().dtype(kFloat), ACL_FORMAT_FRACTAL_Z);
  }

  if (output_mask[2]) {
    grad_bias = OpPreparation::ApplyTensorWithFormat(grad_output, {grad_output.size(1)}, ACL_FORMAT_NCHW);
  }

  return slow_conv_transpose2d_backward_npu_nocheck(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      output_padding,
      dilation,
      columns,
      ones,
      grad_input,
      grad_weight,
      grad_bias);
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("slow_conv_transpose2d_backward.grad_output", TORCH_FN(slow_conv_transpose2d_backward_out_npu));
  m.impl("slow_conv_transpose2d_backward.output_mask", TORCH_FN(slow_conv_transpose2d_backward_npu));
}

} // namespace native
} // namespace at
