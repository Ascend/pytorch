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

at::Tensor slow_conv_dilated2d_backward_input_out_npu(
    at::Tensor grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  c10::SmallVector<int64_t, N> dimList = array_to_small_vector(self.sizes());
  OpCommand cmd;
  cmd.Name("Conv2DBackpropInput")
      .Input(dimList, at::kInt)
      .Input(weight, "filter", ACL_FORMAT_NCHW)
      .Input(grad_output, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Attr("_allow_hf32", true, at_npu::native::env::allowHF32Conv())
      .Run();
      
  return grad_input;
}

at::Tensor slow_conv_dilated2d_backward_weight_out_npu(
    at::Tensor grad_weight,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  c10::SmallVector<int64_t, N> dimList = array_to_small_vector(weight.sizes());
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Conv2DBackpropFilter")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(dimList, at::kInt)
      .Input(grad_output, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_weight)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Attr("_allow_hf32", true, at_npu::native::env::allowHF32Conv())
      .Run();

  return grad_weight;
}

at::Tensor slow_conv_dilated2d_backward_bias_out_npu(
    at::Tensor grad_bias,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  string dataFormat = "NCHW";
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("BiasAddGrad")
      .Input(self)
      .Output(grad_bias)
      .Attr("data_format", dataFormat)
      .Run();

  return grad_bias;
}

tuple<at::Tensor&, at::Tensor&, at::Tensor&> slow_conv_dilated2d_backward_out_npu(
    at::Tensor grad_input,
    at::Tensor grad_weight,
    at::Tensor grad_bias,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    std::array<bool, 3> output_mask) {
   // calculate the output result of the NPU

  if (output_mask[0]) {
    slow_conv_dilated2d_backward_input_out_npu(
        grad_input, grad_output, self, weight, kernel_size, stride, padding, dilation);
  }

  if (output_mask[1]) {
    slow_conv_dilated2d_backward_weight_out_npu(
        grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
  }

  if (output_mask[2]) {
    slow_conv_dilated2d_backward_bias_out_npu(
        grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation);
  }

  return tuple<at::Tensor&, at::Tensor&, at::Tensor&>(grad_input, grad_weight, grad_bias);
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::slow_conv_dilated2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    std::array<bool, 3> output_mask) {

  auto outputSizes =  slow_conv_dilated2d_backward_npu_output_size(
      grad_output,self,weight,kernel_size,stride,padding, dilation);

  at::Tensor undefined;

  at::Tensor grad_input =
      (output_mask[0] ? OpPreparation::ApplyTensor(grad_output, self.sizes()) : undefined);

  at::Tensor grad_weight =
      (output_mask[1] ? OpPreparation::ApplyTensor(grad_output, weight.sizes()) : undefined);

  at::Tensor grad_bias =
      (output_mask[2] ? OpPreparation::ApplyTensor(grad_output, weight.size(0)) : undefined);

  slow_conv_dilated2d_backward_out_npu(
      grad_input,
      grad_weight,
      grad_bias,
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      output_mask);

   return std::tie(grad_input, grad_weight, grad_bias);
}
} // namespace native
} // namespace at_npu
