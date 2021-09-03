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

tuple<Tensor, Tensor> deformable_conv2d_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t deformable_groups,
    bool modulated) {
  // calculate the output size
  auto outputSize = deformable_conv2d_npu_output_size(
      input, weight, offset, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);

  // construct the output tensor of the NPU
  Tensor deformableOffsetsOutput = OpPreparation::ApplyTensorWithFormat(outputSize, input.options(), ACL_FORMAT_NCHW);

  string dataFormat = "NCHW";
  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("DeformableOffsets")
      .Input(input)
      .Input(offset)
      .Output(deformableOffsetsOutput)
      .Attr("ksize", kernel_size)
      .Attr("strides", stride)
      .Attr("pads", padding)
      .Attr("dilations", dilation)
      .Attr("deformable_groups", deformable_groups)
      .Attr("data_format",dataFormat)
      .Attr("modulated",modulated)
      .Run();
  
  SmallVector<int64_t, SIZE> conv2dStride = array_to_small_vector(kernel_size);
  SmallVector<int64_t, SIZE> conv2dPadding = {0, 0, 0, 0};
  SmallVector<int64_t, SIZE> conv2dDilation = {1, 1};
  Tensor conv2dOutput = at::npu_conv2d(
      deformableOffsetsOutput, weight, bias, conv2dStride, conv2dPadding, conv2dDilation, groups);

  return std::tie(conv2dOutput, deformableOffsetsOutput);
}

} // namespace native
} // namespace at