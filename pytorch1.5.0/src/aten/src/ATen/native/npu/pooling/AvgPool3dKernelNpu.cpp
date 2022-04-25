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
#include <ATen/native/Pool.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& avg_pool3d_out_npu(
    Tensor& out,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  SmallVector<int64_t, N> pads = {0, 0, 0, padding[0], padding[1], padding[2]};

  Tensor input = self;
  if (self.ndimension() == 4) {
    input = input.unsqueeze(0);
    out = out.unsqueeze(0);
  }

  // calculate the output size
  int D = self.size(-3);
  int H = self.size(-2);
  int W = self.size(-1);

  int64_t D_size = ceil_mode
      ? (CeilDiv(D + 2 * padding[0] - kernel_size[0], stride[0]) + 1)
      : ((D + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
  int64_t H_size = ceil_mode
      ? (CeilDiv(H + 2 * padding[1] - kernel_size[1], stride[1]) + 1)
      : ((H + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);
  int64_t W_size = ceil_mode
      ? (CeilDiv(W + 2 * padding[2] - kernel_size[2], stride[2]) + 1)
      : ((W + 2 * padding[2] - kernel_size[2]) / stride[2] + 1);

  SmallVector<int64_t, SIZE> outputSize = {
      input.size(0), input.size(1), D_size, H_size, W_size};
  OpPreparation::CheckOut(
      {self},
      out,
      ACL_FORMAT_NCDHW,
      out.scalar_type(),
      outputSize);

  OpCommand cmd;
  cmd.Name("AvgPool3D")
      .Input(input, "x", ACL_FORMAT_NCDHW)
      .Output(out, "y", ACL_FORMAT_NCDHW)
      .Attr("ksize", kernel_size)
      .Attr("strides", stride)
      .Attr("pads", pads)
      .Attr("ceil_mode", ceil_mode)
      .Attr("count_include_pad", count_include_pad);

  if (divisor_override.has_value()) {
    cmd.Attr("divisor_override", divisor_override.value());
  }

  cmd.Attr("data_format", (string)"NCDHW")
      .Run();

  if (self.ndimension() == 4) {
    out = out.squeeze(0);
  }
  return out;
}

Tensor avg_pool3d_npu(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);
  SmallVector<int64_t, SIZE> kernel_sizes = {kT, kH, kW};
  IntArrayRef kernel_sizess = IntArrayRef(kernel_sizes);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  SmallVector<int64_t, SIZE> strides = {dT, dH, dW};
  IntArrayRef stridess = IntArrayRef(strides);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);
  SmallVector<int64_t, SIZE> paddings = {padT, padH, padW};
  IntArrayRef paddingss = IntArrayRef(paddings);

  TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  const int64_t nslices = self.size(-4);
  const int64_t itime = self.size(-3);
  const int64_t iheight = self.size(-2);
  const int64_t iwidth = self.size(-1);

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(
      self,
      nslices,
      kT, kH, kW,
      dT, dH, dW,
      padT, padH, padW,
      1, 1, 1,
      itime, iheight, iwidth,
      otime, oheight, owidth,
      true);

  Tensor input = self;
  if (self.ndimension() == 4) {
    input = self.unsqueeze(0);
  }

  SmallVector<int64_t, SIZE> outputSize = {input.size(0), input.size(1), otime, oheight, owidth};

  Tensor result = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_NCDHW);

  // calculate the output result of the NPU
  avg_pool3d_out_npu(
      result,
      input,
      kernel_sizess,
      stridess,
      paddingss,
      ceil_mode,
      count_include_pad,
      divisor_override);

  if (self.ndimension() == 4) {
    result = result.squeeze(0);
  }
  return result;
}

} // namespace native
} // namespace at
