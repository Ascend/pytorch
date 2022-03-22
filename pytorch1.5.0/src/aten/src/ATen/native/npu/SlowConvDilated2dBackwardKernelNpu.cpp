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

Tensor slow_conv_dilated2d_backward_input_out_npu(
    Tensor grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  SmallVector<int64_t, N> dimList = array_to_small_vector(self.sizes());
  OpCommand cmd;
  cmd.Name("Conv2DBackpropInput")
    .Input(dimList, at::kInt)
    .Input(weight)
    .Input(grad_output)
    .Output(grad_input)
    .Attr("strides", stridesSize)
    .Attr("pads", paddings)
    .Attr("dilations", dilations)
    .Attr("groups", groups)
    .Attr("data_format", dataFormat)
    .Run();

  return grad_input;
}

Tensor slow_conv_dilated2d_backward_weight_out_npu(
    Tensor grad_weight,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  SmallVector<int64_t, N> dimList = array_to_small_vector(weight.sizes());
  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Conv2DBackpropFilter")
      .Input(self)
      .Input(dimList, at::kInt)
      .Input(grad_output)
      .Output(grad_weight)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();
  return grad_weight;
}

Tensor slow_conv_dilated2d_backward_bias_out_npu(
    Tensor grad_bias,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
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


tuple<Tensor&, Tensor&, Tensor&> slow_conv_dilated2d_backward_out_npu(
    Tensor grad_input, 
    Tensor grad_weight, 
    Tensor grad_bias,
    const Tensor& grad_output, 
    const Tensor& self, 
    const Tensor& weight, 
    IntArrayRef kernel_size, 
    IntArrayRef stride, 
    IntArrayRef padding, 
    IntArrayRef dilation, 
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

  return tuple<Tensor&, Tensor&, Tensor&>(grad_input, grad_weight, grad_bias);
}


tuple<Tensor, Tensor, Tensor> slow_conv_dilated2d_backward_npu(
    const Tensor& grad_output, 
    const Tensor& self, 
    const Tensor& weight, 
    IntArrayRef kernel_size, 
    IntArrayRef stride, 
    IntArrayRef padding, 
    IntArrayRef dilation, 
    std::array<bool, 3> output_mask) {

  // calculate the output size
  auto outputSizes =  slow_conv_dilated2d_backward_npu_output_size(
      grad_output,self,weight,kernel_size,stride,padding, dilation);
  
  Tensor undefined;
  
  // construct the output tensor of the NPU
  Tensor grad_input =
      (output_mask[0] ? at::empty(self.sizes(), grad_output.options()) : undefined);
  
  Tensor grad_weight =
      (output_mask[1] ? at::empty(weight.sizes(), grad_output.options()) : undefined);
  
  Tensor grad_bias =
      (output_mask[2] ? at::empty(weight.size(0), grad_output.options()) : undefined);
  
  // calculate the output result of the NPU
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
} // namespace at
