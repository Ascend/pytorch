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

Tensor conv_transpose3d_backward_input_out_npu(
    Tensor& gradInput,
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  // constructs the input and output NPUTensorDesc
  SmallVector<int64_t, N> stridesSize = {
      1, 1, stride[0], stride[1], stride[2]};
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
  SmallVector<int64_t, N> dilations = {
      1, 1, dilation[0], dilation[1], dilation[2]};
  string dataFormat = "NCDHW";

  OpCommand cmd;
  cmd.Name("Conv3D")
      .Input(grad_output, "x", ACL_FORMAT_NCDHW)
      .Input(weight, "filter", ACL_FORMAT_NCDHW)
      .Output(gradInput, "y", ACL_FORMAT_NCDHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();
  return gradInput;
}

Tensor conv_transpose3d_backward_weight_out_npu(
    Tensor& gradWeight,
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  SmallVector<int64_t, N> dimList = array_to_small_vector(weight.sizes());
  SmallVector<int64_t, N> stridesSize = {
      1, 1, stride[0], stride[1], stride[2]};
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
  SmallVector<int64_t, N> dilations = {
      1, 1, dilation[0], dilation[1], dilation[2]};
  string sizeName = "filter_size";
  string dataFormat = "NCDHW";

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("Conv3DBackpropFilterD")
      .Input(grad_output, "x", ACL_FORMAT_NCDHW)
      .Input(input, "out_backprop", ACL_FORMAT_NCDHW)
      .Output(gradWeight, "y", ACL_FORMAT_NCDHW)
      .Attr(sizeName, dimList)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();

  return gradWeight;
}

Tensor conv_transpose3d_backward_bias_out_npu(
    Tensor& gradBias,
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  Tensor gradView = grad_output.contiguous().view({
      grad_output.size(0),
      grad_output.size(1),
      grad_output.size(2),
      -1});
  at::sum_out(gradBias, gradView, SmallVector<int64_t, N>{0, 2, 3});

  return gradBias;
}
tuple<Tensor&, Tensor&, Tensor&> conv_transpose3d_backward_out_npu(
    Tensor& gradInput,
    Tensor& gradWeight,
    Tensor& gradBias,
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  // calculate the output result of the NPU
  if (output_mask[0]) {
    conv_transpose3d_backward_input_out_npu(
        gradInput,
        input,
        grad_output,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups);
  }

  if (output_mask[1]) {
    conv_transpose3d_backward_weight_out_npu(
        gradWeight,
        input,
        grad_output,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups);
  }

  if (output_mask[2]) {
    conv_transpose3d_backward_bias_out_npu(
        gradBias,
        input,
        grad_output,
        weight,
        padding,
        output_padding,
        stride,
        dilation,
        groups);
  }

  return std::tie(gradInput, gradWeight, gradBias);
}

tuple<Tensor, Tensor, Tensor> conv_transpose3d_backward_npu(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  Tensor gradInput;
  Tensor gradWeight;
  Tensor gradBias;

  // construct the output tensor of the NPU
  if (output_mask[0]) {
    gradInput = OpPreparation::ApplyTensorWithFormat(
        input, ACL_FORMAT_NDC1HWC0);
  }

  if (output_mask[1]) {
    gradWeight = OpPreparation::ApplyTensorWithFormat(
        weight.sizes(), weight.options().dtype(kFloat), ACL_FRACTAL_Z_3D);
  }

  if (output_mask[2]) {
    gradBias = OpPreparation::ApplyTensorWithFormat(
        {grad_output.size(1)}, grad_output.options(), ACL_FORMAT_NCDHW);
  }

  // calculate the output result of the NPU
  conv_transpose3d_backward_out_npu(
      gradInput,
      gradWeight,
      gradBias,
      input,
      grad_output,
      weight,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      output_mask);

  return std::tie(gradInput, gradWeight, gradBias);
}

} // namespace native
} // namespace at
