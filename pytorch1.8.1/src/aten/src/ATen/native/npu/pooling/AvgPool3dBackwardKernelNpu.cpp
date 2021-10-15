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

void avg_pool3d_backward_out_npu_nocheck(
    Tensor& grad_output,
    const Tensor& grad_input,
    const Tensor& self,
    IntArrayRef kernel_sizess,
    IntArrayRef stridess,
    IntArrayRef paddingss,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor input = self;
  Tensor grads = grad_input.contiguous();

  grad_output.resize_as_(input);
  grad_output.zero_();
  if (self.ndimension() == 4) {
    input = input.unsqueeze(0);
    grads = grads.unsqueeze(0);
    grad_output = grad_output.unsqueeze(0);
  }

  SmallVector<int64_t, N> dimList(input.sizes());
  SmallVector<int64_t, N> pads = {paddingss[0], paddingss[0], paddingss[1], paddingss[1], paddingss[2], paddingss[2]};

  OpCommand cmd;
  cmd.Name("AvgPool3DGrad")
      .Input(dimList)
      .Input(grads)
      .Output(grad_output)
      .Attr("ksize", kernel_sizess)
      .Attr("strides", stridess)
      .Attr("pads", pads)
      .Attr("ceil_mode", ceil_mode)
      .Attr("count_include_pad", count_include_pad);

  if (divisor_override.has_value()) {
    cmd.Attr("divisor_override", divisor_override.value());
  }

  cmd.Attr("data_format", (string)"NCDHW")
      .Run();

  if (self.ndimension() == 4) {
    grad_output = grad_output.squeeze(0);
  }
}

Tensor& avg_pool3d_backward_out_npu(
    const Tensor& grad_input,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_output) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d_backward: kernel_size must be a single int, or a tuple of three ints");
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d_backward: stride must be omitted, a single int, or a tuple of three ints");
  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
      "avg_pool3d_backward: padding must be a single int, or a tuple of three ints");
  TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
      "avg_pool3d_backward divisor must be not zero");

  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);
  SmallVector<int64_t, SIZE> kernel_sizes = {1, 1, kT, kH, kW};
  IntArrayRef kernel_sizess = IntArrayRef(kernel_sizes);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);
  SmallVector<int64_t, SIZE> strides = {1, 1, dT, dH, dW};
  IntArrayRef stridess = IntArrayRef(strides);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);
  SmallVector<int64_t, SIZE> paddings = {padH, padW, padT};
  IntArrayRef paddingss = IntArrayRef(paddings);

  const int64_t nslices = self.size(-4);
  const int64_t itime = self.size(-3);
  const int64_t iheight = self.size(-2);
  const int64_t iwidth = self.size(-1);
  const int64_t otime = grad_input.size(-3);
  const int64_t oheight = grad_input.size(-2);
  const int64_t owidth = grad_input.size(-1);

  /* XXX shape check behavior from TH */
  const int64_t otime_for_shape_check = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight_for_shape_check = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_shape_check = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
      self,
      grad_input,
      nslices,
      kT, kH, kW,
      dT, dH, dW,
      padT, padH, padW,
      itime, iheight, iwidth,
      otime_for_shape_check, oheight_for_shape_check, owidth_for_shape_check);

  avg_pool3d_backward_out_npu_nocheck(
      grad_output,
      grad_input,
      self,
      kernel_sizess,
      stridess,
      paddingss,
      ceil_mode,
      count_include_pad,
      divisor_override);

  return grad_output;
}

Tensor avg_pool3d_backward_npu(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor input = self;
  Tensor grad_input = grad_output;
  if (self.ndimension() == 4) {
    input = self.unsqueeze(0);
    grad_input = grad_input.unsqueeze(0);
  }

  Tensor output = OpPreparation::ApplyTensorWithFormat(input, ACL_FORMAT_NCDHW);
  avg_pool3d_backward_out_npu(
      grad_input,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      output);

  if (self.ndimension() == 4) {
    output = output.squeeze(0);
  }

  return output;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("avg_pool3d_backward", TORCH_FN(avg_pool3d_backward_npu));
  m.impl("avg_pool3d_backward.grad_input", TORCH_FN(avg_pool3d_backward_out_npu));
}
} // namespace native
} // namespace at
