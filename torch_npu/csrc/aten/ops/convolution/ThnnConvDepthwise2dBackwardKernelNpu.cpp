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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


void thnn_conv_depthwise2d_backward_input_out_npu(
    at::Tensor& grad_input,
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
  auto inputSize = self.sizes();
  OpCommand cmd;
  cmd.Name("DepthwiseConv2DBackpropInput")
    .Input(inputSize, at::kInt)
    .Input(weight, "filter", ACL_FORMAT_NCHW)
    .Input(grad_output, "out_backprop", ACL_FORMAT_NCHW)
    .Output(grad_input, "input_grad", ACL_FORMAT_NCHW)
    .Attr("strides", stridesSize)
    .Attr("pads", paddings)
    .Attr("dilations", dilations)
    .Attr("data_format", (string)"NCHW")
    .Attr("_allow_hf32", true, at_npu::native::env::allowHF32Conv())
    .Run();
}

void thnn_conv_depthwise2d_backward_weight_out_npu(
    at::Tensor& grad_weight,
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
  auto inputSize = weight.sizes();
  OpCommand cmd;
  cmd.Name("DepthwiseConv2DBackpropFilter")
      .Input(self, "input", ACL_FORMAT_NCHW)
      .Input(inputSize, at::kInt)
      .Input(grad_output, "out_backprop", ACL_FORMAT_NCHW)
      .Output(grad_weight, "filter_grad", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string)"NCHW")
      .Attr("_allow_hf32", true, at_npu::native::env::allowHF32Conv())
      .Run();
}

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::thnn_conv_depthwise2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::Tensor& grad_input,
    at::Tensor& grad_weight) {
  at::Tensor weight_ex = weight.permute({1, 0, 2, 3});
  if (grad_input.defined()) {
      thnn_conv_depthwise2d_backward_input_out_npu(
          grad_input,
          grad_output,
          self,
          weight_ex,
          kernel_size,
          stride,
          padding,
          dilation);
  }
  if (grad_weight.defined()) {
    thnn_conv_depthwise2d_backward_weight_out_npu(
        grad_weight,
        grad_output,
        self,
        weight_ex,
        kernel_size,
        stride,
        padding,
        dilation);
  }

  return tuple<at::Tensor&, at::Tensor&>(grad_input, grad_weight);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::thnn_conv_depthwise2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    std::array<bool, 2> output_mask) {
  // calculate the output size
  at::Tensor grad_input;
  at::Tensor grad_weight;
  // construct the output tensor of
  if (output_mask[0]) {
    grad_input = OpPreparation::ApplyTensorWithFormat(self, ACL_FORMAT_NC1HWC0);
  }

  if (output_mask[1]) {
    grad_weight = OpPreparation::ApplyTensor(weight);
  }

  return NPUNativeFunctions::thnn_conv_depthwise2d_backward_out(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      grad_input,
      grad_weight);
}

} // namespace native
} // namespace at_npu