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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor slow_conv_transpose2d_backward_grad_output_out_npu(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    const at::Tensor& columns,
    const at::Tensor& ones) {
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Conv2D")
      .Input(grad_output, "x", ACL_FORMAT_NCHW)
      .Input(weight, "filter", ACL_FORMAT_NCHW)
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return grad_input;
}

at::Tensor slow_conv_transpose2d_backward_weight_out_npu(
    at::Tensor& grad_weight,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    const at::Tensor& columns,
    const at::Tensor& ones) {
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  c10::SmallVector<int64_t, N> dimList = array_to_small_vector(weight.sizes());
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Conv2DBackpropFilter")
      .Input(grad_output, "x", ACL_FORMAT_NCHW)
      .Input(dimList, at::kInt)
      .Input(self, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_weight, "y", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return grad_weight;
}

at::Tensor slow_conv_transpose2d_backward_bias_out_npu(
    at::Tensor& grad_bias,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    const at::Tensor& columns,
    const at::Tensor& ones) {
  at::Tensor gradView = grad_output.contiguous().view({grad_output.size(0), grad_output.size(1), -1});
  NPUNativeFunctions::sum_out(gradView, c10::SmallVector<int64_t, N>{0, 2}, false, gradView.scalar_type(), grad_bias);

  return grad_bias;
}

tuple<at::Tensor&, at::Tensor&, at::Tensor&> slow_conv_transpose2d_backward_npu_nocheck(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    const at::Tensor& columns,
    const at::Tensor& ones,
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias) {
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
      
  return tuple<at::Tensor&, at::Tensor&, at::Tensor&>(grad_input, grad_weight, grad_bias);
}

tuple<at::Tensor&, at::Tensor&, at::Tensor&> NPUNativeFunctions::slow_conv_transpose2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    const at::Tensor& columns,
    const at::Tensor& ones,
    at::Tensor& grad_input,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias) {
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
      CalcuOpUtil::GetTensorNpuFormat(weight),
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

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::slow_conv_transpose2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    const at::Tensor& columns,
    const at::Tensor& ones,
    std::array<bool, 3> output_mask) {
  auto flag = 2;
  auto outputSizes = slow_conv_transpose2d_backward_npu_output_size(
      grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
  at::Tensor grad_input;
  at::Tensor grad_weight;
  at::Tensor grad_bias;
  
  if (output_mask[0]) {
    grad_input = OpPreparation::ApplyTensorWithFormat(self, std::get<0>(outputSizes), ACL_FORMAT_NC1HWC0);
  }

  if (output_mask[1]) {
    grad_weight = OpPreparation::ApplyTensorWithFormat(
        std::get<1>(outputSizes),
        weight.options().dtype(at::kFloat),
        CalcuOpUtil::GetTensorNpuFormat(weight));
  }

  if (output_mask[flag]) {
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
} // namespace native
} // namespace at_npu
