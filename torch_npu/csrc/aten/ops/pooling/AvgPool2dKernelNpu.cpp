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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


at::Tensor& avg_pool2d_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  if (padding.size() == 1) {
    c10::SmallVector<int64_t, SIZE> paddings = {padding[0], padding[0]};
    padding = at::IntArrayRef(paddings);
  }

  int64_t strideH = 1;
  int64_t strideW = 1;
  if (!stride.empty()) {
    strideH = stride[0];
    strideW = stride[1];
  }
  c10::SmallVector<int64_t, N> kernelSize = {1, 1, kernel_size[0], kernel_size[1]};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, strideH, strideW};
  c10::SmallVector<int64_t, N> pads = {padding[0], padding[0], padding[1], padding[1]};
  bool exclusive = count_include_pad ? false : true;

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
      .Attr("ceil_mode", ceil_mode);
  if (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::Char) {
    cmd.Attr("exclusive", true);
  } else {
    cmd.Attr("exclusive", exclusive);
  }
  cmd.Run();
  return result;
}

at::Tensor& NPUNativeFunctions::avg_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& result) {
  at::Tensor selfCp = self;
  if (self.dim() == 3) {
    selfCp = selfCp.unsqueeze(0);
  }
  auto outputSize = avg_pool2d_npu_output_size(
      selfCp,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  OpPreparation::CheckOut(
      {self},
      result,
      selfCp,
      outputSize);

  avg_pool2d_out_npu_nocheck(
      result,
      selfCp,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  if (self.dim() == 3) {
    result = result.squeeze(0);
  }
  return result;
}

at::Tensor NPUNativeFunctions::avg_pool2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  at::Tensor selfCp = self;
  if (self.dim() == 3) {
    selfCp = selfCp.unsqueeze(0);
  }
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  c10::SmallVector<int64_t, SIZE> kernel_sizes = {kH, kW};
  at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int64_t dH = stride.empty() ? kH : stride[0];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  c10::SmallVector<int64_t, SIZE> stride_sizes = {dH, dW};
  at::IntArrayRef stridess = at::IntArrayRef(stride_sizes);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int64_t padH = padding[0];
  const int64_t padW = padding.size() == 1 ? padH : padding[1];

  c10::SmallVector<int64_t, SIZE> padding_sizes = {padH, padW};
  at::IntArrayRef paddingss = at::IntArrayRef(padding_sizes);

  TORCH_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "non-empty 2D or 3D (batch mode) tensor expected for input");

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  auto outputSizes = avg_pool2d_npu_output_size(
      selfCp,
      kernel_sizess,
      stridess,
      paddingss,
      ceil_mode,
      count_include_pad,
      divisor_override);
  at::Tensor result = OpPreparation::ApplyTensor(selfCp, outputSizes);

  avg_pool2d_out_npu_nocheck(
      result,
      selfCp,
      kernel_sizess,
      stridess,
      paddingss,
      ceil_mode,
      count_include_pad,
      divisor_override);
  if (self.dim() == 3) {
      result = result.squeeze(0);
  }
  return result;
}


} // namespace native
} // namespace at_npu
