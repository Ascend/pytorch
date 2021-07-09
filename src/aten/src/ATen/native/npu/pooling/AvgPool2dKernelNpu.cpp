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

Tensor& avg_pool2d_out_npu_nocheck(
  Tensor& result,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  if (padding.size() == 1) {
    SmallVector<int64_t, SIZE> paddings = {padding[0], padding[0]};
    padding = IntArrayRef(paddings);
  }

  // required attr
  int64_t strideH = 1;
  int64_t strideW = 1;
  if (!stride.empty()) {
    strideH = stride[0];
    strideW = stride[1];
  }
  SmallVector<int64_t, N> kernelSize = {1, 1, kernel_size[0], kernel_size[1]};
  SmallVector<int64_t, N> stridesSize = {1, 1, strideH, strideW};
  SmallVector<int64_t, N> pads = {padding[0], padding[0], padding[1], padding[1]};

  OpCommand cmd;
  cmd.Name("AvgPoolV2")
     .Input(self)
     .Output(result)
     .Attr("ksize", kernelSize)
     .Attr("strides", stridesSize)
     .Attr("padding_mode", (string)"CALCULATED")
     .Attr("pads", pads)
     .Attr("data_format", (string)"NCHW")
     .Attr("global_pooling", false)
     .Attr("ceil_mode", ceil_mode)
     .Attr("exclusive", true)
     .Run();

  return result;
}

Tensor& avg_pool2d_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  auto outputSize = avg_pool2d_npu_output_size(
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  
  avg_pool2d_out_npu_nocheck(
      result,
      self,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  return result;
}

Tensor avg_pool2d_npu(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  SmallVector<int64_t, SIZE> kernel_sizes = {kH, kW};
  IntArrayRef kernel_sizess = IntArrayRef(kernel_sizes);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int64_t dH = stride.empty() ? kH : stride[0];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  SmallVector<int64_t, SIZE> stride_sizes = {dH, dW};
  IntArrayRef stridess = IntArrayRef(stride_sizes);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int64_t padH = padding[0];
  const int64_t padW = padding.size() == 1 ? padH : padding[1];

  SmallVector<int64_t, SIZE> padding_sizes = {padH, padW};
  IntArrayRef paddingss = IntArrayRef(padding_sizes);

  TORCH_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "non-empty 2D or 3D (batch mode) tensor expected for input");

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  // calculate the output size
  auto outputSizes = avg_pool2d_npu_output_size(
      self,
      kernel_sizess,
      stridess,
      paddingss,
      ceil_mode,
      count_include_pad,
      divisor_override);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSizes);
  // calculate the output result of the NPU
  avg_pool2d_out_npu(
      result,
      self,
      kernel_sizess,
      stridess,
      paddingss,
      ceil_mode,
      count_include_pad,
      divisor_override);
  return result;
}

} // namespace native
} // namespace at
