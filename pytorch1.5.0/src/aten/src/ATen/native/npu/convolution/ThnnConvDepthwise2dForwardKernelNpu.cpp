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

Tensor& thnn_conv_depthwise2d_forward_out_npu(
    Tensor& out,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  const Tensor& weightModify = weight.permute({1, 0, 2, 3});

  // constructs the input and output NPUTensorDesc
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1], padding[1]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};

  OpCommand cmd;
  cmd.Name("DepthwiseConv2D")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(weightModify, "filter", ACL_FORMAT_NCHW);
  if (bias.defined()) {
      cmd.Input(bias);
  }
  cmd.Output(out, "y", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("data_format", (string) "NCHW")
      .Run();
  return out;
}

Tensor thnn_conv_depthwise2d_forward_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  // calculate the output size
  int64_t N = self.size(0);
  int64_t Co = weight.size(0);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
          stride[0] + 1;
  int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
          stride[1] + 1;
  SmallVector<int64_t, SIZE> outputSize = {N, Co, Ho, Wo};

  // construct the output tensor of NPU
  Tensor result = OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_NC1HWC0);

  // calculate the output result of the NPU
  thnn_conv_depthwise2d_forward_out_npu(
      result, self, weight, kernel_size, bias, stride, padding, dilation);
  return result;
}

} // namespace native
} // namespace at
