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

tuple<Tensor, Tensor, Tensor, Tensor> deformable_conv2d_backward_npu(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& offset_out,
    const Tensor& weight,
    const Tensor& offset,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated) {
  // construct the output tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensorWithFormat(input, ACL_FORMAT_NCHW);
  Tensor grad_offset = OpPreparation::ApplyTensorWithFormat(offset, ACL_FORMAT_NCHW);

  // deformable_conv2d_backward includes conv2d_backward and DeformableOffsetsGrad
  SmallVector<int64_t, SIZE> conv2dStride = array_to_small_vector(kernel_size);
  SmallVector<int64_t, SIZE> conv2dPadding = {0, 0, 0, 0};
  SmallVector<int64_t, SIZE> conv2dDilation = {1, 1};
  auto conv2dBackwardOutput = at::npu_conv2d_backward(
      offset_out, grad_output, weight, conv2dStride, conv2dPadding, conv2dDilation, groups, {true, true, true});

  // DeformableOffsetsGrad's input 'grad' is the output[0] of conv2d_backward
  Tensor deformableOffsetsBackwardInput = std::get<0>(conv2dBackwardOutput);
  Tensor grad_weight = std::get<1>(conv2dBackwardOutput);
  Tensor grad_bias = std::get<2>(conv2dBackwardOutput);

  string dataFormat = "NCHW";
  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("DeformableOffsetsGrad")
      .Input(deformableOffsetsBackwardInput)
      .Input(input)
      .Input(offset)
      .Output(grad_input)
      .Output(grad_offset)
      .Attr("strides", stride)
      .Attr("pads", padding)
      .Attr("ksize", kernel_size)
      .Attr("dilations", dilation)
      .Attr("data_format",dataFormat)
      .Attr("deformable_groups", deformable_groups)
      .Attr("modulated",modulated)
      .Run();
      
  return std::tie(grad_input, grad_weight, grad_offset, grad_bias);
}

} // namespace native
} // namespace at