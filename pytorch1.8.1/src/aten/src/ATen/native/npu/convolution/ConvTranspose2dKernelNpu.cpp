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
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& convolution_transpose_out_npu(
    Tensor& result,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> outputpadding = {0, 0, 0, 0};
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  string dataFormat = "NCHW";
  string sizeName = "input_size";

  SmallVector<int64_t, N> sizeVec = array_to_small_vector(result.sizes());
  OpCommand cmd;
  cmd.Name("Conv2DTranspose")
      .Input(sizeVec, at::kInt)
      .Input(input)
      .Input(weight);
  if (bias.defined()){
    cmd.Input(bias);
  }
  cmd.Output(result)
      .Attr("pads", paddings)
      .Attr("output_padding", outputpadding)
      .Attr("strides", stridesSize)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();
    
  return result;
}

Tensor convolution_transpose_npu(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

  // calculate the output size
  auto outputSize = convolution_transpose_npu_output_size(
      input, weight, bias, padding, output_padding, stride, dilation, groups);

  // construct the output tensor of the NPU
  Tensor result =
      at::empty_with_format(outputSize, input.options(), ACL_FORMAT_NC1HWC0);

  // calculate the output result of the NPU
  convolution_transpose_out_npu(
      result, input, weight, bias, padding, output_padding, stride, dilation, groups);

  return result;
}

Tensor npu_conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  
    return convolution_transpose_npu(input, weight, bias_opt, padding, output_padding, stride, dilation, groups);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_conv_transpose2d", TORCH_FN(convolution_transpose_npu));
}

} // namespace native
} // namespace at
