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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
c10::SmallVector<int64_t, SIZE> slow_conv_transpose2d_npu_output_size(
    const at::Tensor & self,
    const at::Tensor & weight,
    at::IntArrayRef kernel_size,
    const at::Tensor & bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation) {
  int ndim = self.dim();
  int dimh = 1;
  int dimw = 2;
  auto flag = 4;
  auto flag_a = 3;

  if (ndim == flag) {
    dimh++;
    dimw++;
  }

  TORCH_CHECK(
      self.numel() != 0 && (ndim == flag_a || ndim == flag),
      "non-empty 3D or 4D input tensor expected but got a tensor with size ",
      self.sizes());
  int64_t N = self.size(0);
  int64_t Co = weight.size(1);
  int64_t H = self.size(dimh);
  int64_t W = self.size(dimw);

  int64_t Ho = (H - 1) * stride[0] - 2 * padding[0] +
      dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
  int64_t Wo = (W - 1) * stride[1] - 2 * padding[1] +
      dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

  c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Ho, Wo};

  return outputSize;
}

static inline void slow_conv_transpose2d_shape_check_npu(
    const at::Tensor & self,
    const at::Tensor & weight,
    at::IntArrayRef kernel_size,
    const at::Tensor & bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size[0] > 0 && kernel_size[1] > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_size[0],
      " kernel_width: ",
      kernel_size[1]);
  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride[0],
      " stride_width: ",
      stride[1]);
  TORCH_CHECK(
      dilation[0] > 0 && dilation[1] > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation[0],
      ", dilation_width: ",
      dilation[1]);
  TORCH_CHECK(
      (output_padding[1] < stride[1] ||
       output_padding[1] < dilation[1]) &&
      (output_padding[0] < stride[0] ||
       output_padding[0] < dilation[0]),
      "output padding must be smaller than either stride or dilation, but got output_padding_height: ",
      output_padding[0],
      " output_padding_width: ",
      output_padding[1],
      " stride_height: ",
      stride[0],
      " stride_width: ",
      stride[1],
      " dilation_height: ",
      dilation[0],
      " dilation_width: ",
      dilation[1]);

  auto flag_a = 2;
  auto flag_b = 4;

  TORCH_CHECK(
      weight.numel() != 0 && (weight.dim() == flag_a || weight.dim() == flag_b),
      "non-empty 2D or 4D weight tensor expected, but got: ",
      weight.sizes());
  if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(1));
  }

  TORCH_CHECK(
      kernel_size.size() == flag_a,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == flag_a,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == flag_a,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == flag_a,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  TORCH_CHECK(
      output_padding.size() == flag_a,
      "It is expected stride equals to 2, but got size ",
      output_padding.size());
}

at::Tensor& slow_conv_transpose2d_npu_nocheck(
    const at::Tensor & self,
    const at::Tensor & weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor> & bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    at::Tensor& out) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  slow_conv_transpose2d_shape_check_npu(
      self, weight, kernel_size, bias, stride, padding, output_padding, dilation);

  auto outputSize = slow_conv_transpose2d_npu_output_size(
      self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (!out.sizes().equals(outputSize)) {
    out.resize_(outputSize);
  }

  c10::SmallVector<int64_t, N> paddings = {
      padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  c10::SmallVector<int64_t, N> outputpadding = {
      output_padding[0], output_padding[0], output_padding[1], output_padding[1]};
  string dataFormat = "NCHW";
  int64_t groups = 1;
  c10::SmallVector<int64_t, N> sizeVec = array_to_small_vector(out.sizes());
  OpCommand cmd;
  cmd.Name("Conv2DTranspose")
      .Input(sizeVec, at::kInt)
      .Input(self)
      .Input(weight);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(out)
      .Attr("pads", paddings)
      .Attr("output_padding", outputpadding)
      .Attr("strides", stridesSize)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", dataFormat)
      .Run();
  return out;
}

at::Tensor& NPUNativeFunctions::slow_conv_transpose2d_out(
    const at::Tensor & self,
    const at::Tensor & weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor> & bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation,
    at::Tensor& out) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = slow_conv_transpose2d_npu_output_size(
      self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  
  if (bias.defined()) {
    OpPreparation::CheckOut(
        {self, weight, bias},
        {out},
        ACL_FORMAT_NC1HWC0,
        self.scalar_type(),
        outputSize);
  } else {
    OpPreparation::CheckOut(
        {self, weight},
        {out},
        ACL_FORMAT_NC1HWC0,
        self.scalar_type(),
        outputSize);
  }

  return slow_conv_transpose2d_npu_nocheck(
      self, weight, kernel_size, bias_opt, stride, padding, output_padding, dilation, out);
}

at::Tensor NPUNativeFunctions::slow_conv_transpose2d(
    const at::Tensor & self,
    const at::Tensor & weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor> & bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef dilation) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = slow_conv_transpose2d_npu_output_size(
      self, weight, kernel_size, bias, stride, padding, output_padding, dilation);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_NC1HWC0);
  slow_conv_transpose2d_npu_nocheck(
      self, weight, kernel_size, bias_opt, stride, padding, output_padding, dilation, result);

  return result;
}
} // namespace native
} // namespace at_npu
