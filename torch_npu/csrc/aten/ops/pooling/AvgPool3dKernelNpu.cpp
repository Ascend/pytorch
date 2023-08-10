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

#include <ATen/native/Pool.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& avg_pool3d_out_nocheck(
    at::Tensor& result,
    at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  c10::SmallVector<int64_t, N> pads = {padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
  OpCommand cmd;
  cmd.Name("AvgPool3D")
      .Input(input, "x")
      .Output(result, "y")
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
  return result;
}

void avg_pool3d_parameter_check(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");
}

c10::SmallVector<int64_t, SIZE> cal_kernel_size(at::IntArrayRef kernel_size) {
  const int k_T = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
  const int k_H = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[1]);
  const int k_W = kernel_size.size() == 1 ? k_T : at::native::safe_downcast<int, int64_t>(kernel_size[2]);
  c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_T, k_H, k_W};
  return kernel_sizes;
}

c10::SmallVector<int64_t, SIZE> cal_stride_size(at::IntArrayRef stride, c10::SmallVector<int64_t, SIZE> kernel_size) {
  const int d_T = stride.empty() ? kernel_size[0] : at::native::safe_downcast<int, int64_t>(stride[0]);
  const int d_H =
      stride.empty() ? kernel_size[1] : stride.size() == 1 ? d_T : at::native::safe_downcast<int, int64_t>(stride[1]);
  const int d_W =
      stride.empty() ? kernel_size[2] : stride.size() == 1 ? d_T : at::native::safe_downcast<int, int64_t>(stride[2]);
  c10::SmallVector<int64_t, SIZE> strides = {d_T, d_H, d_W};
  return strides;
}

c10::SmallVector<int64_t, SIZE> cal_pad_size(at::IntArrayRef padding) {
  const int pad_T = at::native::safe_downcast<int, int64_t>(padding[0]);
  const int pad_H = padding.size() == 1 ? pad_T : at::native::safe_downcast<int, int64_t>(padding[1]);
  const int pad_W = padding.size() == 1 ? pad_T : at::native::safe_downcast<int, int64_t>(padding[2]);
  c10::SmallVector<int64_t, SIZE> paddings = {pad_T, pad_H, pad_W};
  return paddings;
}

c10::SmallVector<int64_t, SIZE> cal_output_size(
    const at::Tensor& self,
    c10::SmallVector<int64_t, SIZE> kernel_size,
    c10::SmallVector<int64_t, SIZE> stride,
    c10::SmallVector<int64_t, SIZE> padding,
    bool ceil_mode) {
  const int64_t nslices = self.size(-4);
  const int64_t itime = self.size(-3);
  const int64_t iheight = self.size(-2);
  const int64_t iwidth = self.size(-1);
  const int64_t otime =
      at::native::pooling_output_shape<int64_t>(itime, kernel_size[0], padding[0], stride[0], 1, ceil_mode);
  const int64_t oheight =
      at::native::pooling_output_shape<int64_t>(iheight, kernel_size[1], padding[1], stride[1], 1, ceil_mode);
  const int64_t owidth =
      at::native::pooling_output_shape<int64_t>(iwidth, kernel_size[2], padding[2], stride[2], 1, ceil_mode);

  at::native::pool3d_shape_check(
      self,
      nslices,
      kernel_size[0], kernel_size[1], kernel_size[2],
      stride[0], stride[1], stride[2],
      padding[0], padding[1], padding[2],
      1, 1, 1,
      itime, iheight, iwidth,
      otime, oheight, owidth,
      true);

  at::Tensor input = self;
  if (self.ndimension() == 4) {
    input = self.unsqueeze(0);
  }
  c10::SmallVector<int64_t, SIZE> output_size = {input.size(0), input.size(1), otime, oheight, owidth};
  return output_size;
}

at::Tensor& NPUNativeFunctions::avg_pool3d_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& result) {
  avg_pool3d_parameter_check(self, kernel_size, stride, padding, divisor_override);

  c10::SmallVector<int64_t, SIZE> kernel_sizes = cal_kernel_size(kernel_size);
  c10::SmallVector<int64_t, SIZE> strides = cal_stride_size(stride, kernel_sizes);
  c10::SmallVector<int64_t, SIZE> paddings = cal_pad_size(padding);
  c10::SmallVector<int64_t, SIZE> output_size =
      cal_output_size(self, kernel_sizes, strides, paddings, ceil_mode);

  at::Tensor input = self;
  if (self.ndimension() == 4) {
    input = input.unsqueeze(0);
    result = result.unsqueeze(0);
  }

  OpPreparation::CheckOut(
      {self},
      result,
      result,
      output_size);

  at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);
  at::IntArrayRef stridess = at::IntArrayRef(strides);
  at::IntArrayRef paddingss = at::IntArrayRef(paddings);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    avg_pool3d_out_nocheck(contiguous_result, input, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad,
        divisor_override);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    avg_pool3d_out_nocheck(result, input, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad,
        divisor_override);
  }

  if (self.ndimension() == 4) {
    result = result.squeeze(0);
  }
  return result;
}

at::Tensor NPUNativeFunctions::avg_pool3d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  avg_pool3d_parameter_check(self, kernel_size, stride, padding, divisor_override);

  c10::SmallVector<int64_t, SIZE> kernel_sizes = cal_kernel_size(kernel_size);
  c10::SmallVector<int64_t, SIZE> strides = cal_stride_size(stride, kernel_sizes);
  c10::SmallVector<int64_t, SIZE> paddings = cal_pad_size(padding);
  c10::SmallVector<int64_t, SIZE> output_size =
      cal_output_size(self, kernel_sizes, strides, paddings, ceil_mode);

  at::Tensor input = self;
  if (self.ndimension() == 4) {
    input = self.unsqueeze(0);
  }
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(input, output_size, ACL_FORMAT_NCDHW);
  at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);
  at::IntArrayRef stridess = at::IntArrayRef(strides);
  at::IntArrayRef paddingss = at::IntArrayRef(paddings);

  avg_pool3d_out_nocheck(result, input, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad,
      divisor_override);

  if (self.ndimension() == 4) {
    result = result.squeeze(0);
  }
  return result;
}
} // namespace native
} // namespace at_npu
