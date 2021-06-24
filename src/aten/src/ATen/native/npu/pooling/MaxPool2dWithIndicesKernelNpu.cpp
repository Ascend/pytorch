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

tuple<Tensor&, Tensor&> max_pool2d_with_indices_out_npu(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (stride.empty()) {
    strideH = kernel_size[0];
    strideW = kernel_size[1];
  } else {
    strideH = stride[0];
    strideW = stride[1];
  }

  SmallVector<int64_t, N> kernelSize = {1, kernel_size[0], kernel_size[1], 1};
  SmallVector<int64_t, N> stridesSize = {1, strideH, strideW, 1};
  SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
  SmallVector<int64_t, N> dilations = {1, dilation[0], dilation[1], 1};

  OpCommand cmd;
  cmd.Name("MaxPoolWithArgmaxV1")
      .Input(self)
      .Output(output)
      .Output(indices, "uint16")
      .Attr("ksize", kernelSize)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilation", dilations)
      .Attr("ceil_mode", ceil_mode)
      .Run();
  return tuple<Tensor&, Tensor&>(output, indices);
}

tuple<Tensor, Tensor> max_pool2d_with_indices_npu(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // calculate the output size
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (stride.empty()) {
    strideH = kernel_size[0];
    strideW = kernel_size[1];
  } else {
    strideH = stride[0];
    strideW = stride[1];
  }

  int64_t N = self.size(0);
  int64_t C = self.size(1);
  int64_t H = self.size(2);
  int64_t W = self.size(3);

  int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1 +
                (ceil_mode ? strideH - 1 : 0)) / strideH + 1;
  int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1 +
                (ceil_mode ? strideW - 1 : 0)) / strideW + 1;
  SmallVector<int64_t, SIZE> outputSize = {N, C, Ho, Wo};

  const int64_t BLOCKSIZE = 16;
  int64_t maskH = kernel_size[0] * kernel_size[1];
  int64_t maskW = (CeilDiv(Ho * Wo, BLOCKSIZE) + 1);
  SmallVector<int64_t, SIZE> indicesSize = {N, C, maskH, maskW};

  // construct the output tensor of the NPU
  Tensor output = OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_NC1HWC0);
  Tensor indices = OpPreparation::ApplyTensorWithFormat(self, indicesSize, ACL_FORMAT_NC1HWC0);

  // calculate the output result of the NPU
  max_pool2d_with_indices_out_npu(
      output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
  return tuple<Tensor, Tensor>(output, indices);
}

} // namespace native
} // namespace at
