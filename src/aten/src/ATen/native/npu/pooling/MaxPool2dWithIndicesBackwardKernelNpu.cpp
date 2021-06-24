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

Tensor& max_pool2d_with_indices_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {

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
  cmd.Name("MaxPoolGradWithArgmaxV1")
      .Input(self)
      .Input(grad_output)
      .Input(indices, "", "uint16")
      .Output(grad_input)
      .Attr("ksize", kernelSize)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("ceil_mode", ceil_mode)
      .Run();
  return grad_input;
}

Tensor max_pool2d_with_indices_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  // construct the output tensor of the NPU
  Tensor grad_input =  OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  max_pool2d_with_indices_backward_out_npu(
      grad_input,
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices);

  return grad_input;
}

} // namespace native
} // namespace at
