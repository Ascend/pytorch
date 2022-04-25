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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor _nnpack_spatial_convolution_nocheck_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& result) {
  SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[0], padding[0]};
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[0]};
  SmallVector<int64_t, N> dilations = {1, 1, 1, 1};

  if (padding.size() != 1) {
    paddings[2] = padding[1];
    paddings[3] = padding[1];
  }
  if (stride.size() != 1) {
    stridesSize[3] = stride[1];
  }

  OpCommand cmd;
  cmd.Name("Conv2D")
      .Input(input, "x", ACL_FORMAT_NCHW)
      .Input(weight, "filter", ACL_FORMAT_NCHW)
      .Input(bias)
      .Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", (int64_t)1)
      .Attr("data_format", (string) "NCHW")
      .Run();
  return result;
}

Tensor _nnpack_spatial_convolution_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride) {
  auto outputSize = nnpack_spatial_convolution_npu_output_size(
      input, weight, padding, stride);
  Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, input.options(), ACL_FORMAT_NC1HWC0);
  _nnpack_spatial_convolution_nocheck_npu(
      input, weight, bias, padding, stride, result);
  return result;
}

} // namespace native
} // namespace at
