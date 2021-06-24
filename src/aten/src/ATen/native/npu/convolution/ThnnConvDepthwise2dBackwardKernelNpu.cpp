// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION. 
// All rights reserved.
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

void thnn_conv_depthwise2d_backward_input_out_npu(
    Tensor& grad_input,
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
  auto inputSize = self.sizes();
  OpCommand cmd;
  if (!c10::npu::OptionsManager::CheckDynamicEnable()){
    cmd.Name("DepthwiseConv2DBackpropInput")
      .Input(inputSize, at::kInt)
      .Input(weight)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string)"NCHW")
      .Run();
  } else {
    cmd.Name("DepthwiseConv2DBackpropInputD")
      .Input(weight)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("strides", stridesSize)
      .Attr("input_size", inputSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string)"NCHW")
      .Run();
  }
}

void thnn_conv_depthwise2d_backward_weight_out_npu(
    Tensor& grad_weight,
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
  auto inputSize = weight.sizes();
  OpCommand cmd;
  if (!c10::npu::OptionsManager::CheckDynamicEnable()){
    cmd.Name("DepthwiseConv2DBackpropFilter")
        .Input(self)
        .Input(inputSize, at::kInt)
        .Input(grad_output)
        .Output(grad_weight)
        .Attr("strides", stridesSize)
        .Attr("pads", paddings)
        .Attr("dilations", dilations)
        .Attr("data_format", (string)"NCHW")
        .Run();
  } else {
    cmd.Name("DepthwiseConv2DBackpropFilterD")
      .Input(self)
      .Input(grad_output)
      .Output(grad_weight)
      .Attr("strides", stridesSize)
      .Attr("filter_size", inputSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string)"NCHW")
      .Run();
  }
}

tuple<Tensor&, Tensor&> thnn_conv_depthwise2d_backward_out_npu(
    Tensor& grad_input,
    Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  Tensor weight_ex = weight.permute({1, 0, 2, 3});
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

  return tuple<Tensor&, Tensor&>(grad_input, grad_weight);
}

tuple<Tensor, Tensor> thnn_conv_depthwise2d_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    std::array<bool, 2> output_mask) {
  // calculate the output size
  Tensor grad_input;
  Tensor grad_weight;
  // construct the output tensor of
  if (output_mask[0]) {
    grad_input = OpPreparation::ApplyTensorWithFormat(self, ACL_FORMAT_NC1HWC0);
  }

  if (output_mask[1]) {
    grad_weight = OpPreparation::ApplyTensorWithFormat(weight, ACL_FORMAT_NCHW);
  }

  // calculate the output result of the NPU
  return thnn_conv_depthwise2d_backward_out_npu(
      grad_input,
      grad_weight,
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      dilation);
}

} // namespace native
} // namespace at