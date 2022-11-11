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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


c10::SmallVector<int64_t, SIZE> conv3d_npu_output_size(
    const at::Tensor &input, const at::Tensor &weight,
    const at::Tensor &bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  int64_t N = input.size(0);
  int64_t D = input.size(2);
  int64_t H = input.size(3);
  int64_t W = input.size(4);
  int64_t Co = weight.size(0);
  auto kernel_size = weight.sizes().slice(2);
  int64_t Do =
      (D + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
  int64_t Ho =
      (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
  int64_t Wo =
      (W + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1;

  c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Do, Ho, Wo};

  return outputSize;
}

at::Tensor &conv3d_out_npu_nocheck(at::Tensor &result, const at::Tensor &input,
                                   const at::Tensor &weight, const at::Tensor &bias,
                                   at::IntArrayRef stride, at::IntArrayRef padding,
                                   at::IntArrayRef dilation, int64_t groups) {
  at::Tensor filter = weight.to(input.dtype());
  c10::SmallVector<at::Tensor, N> inputTensor = {input, filter, bias};
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};

  OpCommand cmd;
  cmd.Name("Conv3D");
  cmd.Input(input, "x", ACL_FORMAT_NCDHW);
  cmd.Input(filter, "filter", ACL_FORMAT_NCDHW);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(result, "y", ACL_FORMAT_NCDHW);
  cmd.Attr("strides", stridesSize);
  cmd.Attr("pads", paddings);
  cmd.Attr("dilations", dilations);
  cmd.Attr("groups", groups);
  cmd.Attr("data_format", (string) "NCDHW");
  cmd.Run();

  return result;
}

at::Tensor& NPUNativeFunctions::npu_conv3d_out(const at::Tensor &input,
                                               const at::Tensor &weight,
                                               const c10::optional<at::Tensor> &bias_opt,
                                               at::IntArrayRef stride,
                                               at::IntArrayRef padding,
                                               at::IntArrayRef dilation,
                                               int64_t groups,
                                               at::Tensor &result) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = conv3d_npu_output_size(
      input, weight, bias, stride, padding, dilation, groups);
  OpPreparation::CheckOut(
      {input, weight, bias},
      result,
      input,
      outputSize);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({input, weight, bias}, {result})
             .Func([&input, &weight, &bias, stride, padding, dilation, groups](at::Tensor &result) {
                 conv3d_out_npu_nocheck(
                     result, input, weight, bias, stride, padding, dilation, groups);
              })
             .Call(result);
}

at::Tensor NPUNativeFunctions::npu_conv3d(const at::Tensor &input,
    const at::Tensor &weight, const c10::optional<at::Tensor> &bias_opt,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    int64_t groups) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});

  auto outputSize = conv3d_npu_output_size(
      input, weight, bias, stride, padding, dilation, groups);

  at::Tensor result = OpPreparation::ApplyTensor(input, outputSize);
  conv3d_out_npu_nocheck(result, input, weight, bias, stride, padding, dilation, groups);

  return result;
}

tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> slow_conv3d_npu_output_size(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t D = input.size(2);
  int64_t H = input.size(3);
  int64_t W = input.size(4);
  int64_t Co = weight.size(0);
  auto kernel_size = weight.sizes().slice(2);
  int64_t Do =
      (D + 2 * padding[0] - (kernel_size[0])) / stride[0] + 1;
  int64_t Ho =
      (H + 2 * padding[1] - (kernel_size[1])) / stride[1] + 1;
  int64_t Wo =
      (W + 2 * padding[2] - (kernel_size[2])) / stride[2] + 1;

  c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Do, Ho, Wo};
  c10::SmallVector<int64_t, SIZE> finputSize = {
    N, C * kernel_size[0] * kernel_size[1] * kernel_size[2], Do * Ho * Wo};

  return tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(outputSize, finputSize);
}

at::Tensor& slow_conv3d_forward_npu_nocheck(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor> & bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  at::Tensor filter = weight.to(input.dtype());
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1, 1};

  OpCommand cmd;
  cmd.Name("Conv3D");
  cmd.Input(input, "x", ACL_FORMAT_NCDHW);
  cmd.Input(filter, "filter", ACL_FORMAT_NCDHW);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(output, "y", ACL_FORMAT_NCDHW);
  cmd.Attr("strides", stridesSize);
  cmd.Attr("pads", paddings);
  cmd.Attr("dilations", dilations);
  cmd.Attr("data_format", (string) "NCDHW");
  cmd.Run();

  return output;
}

at::Tensor& NPUNativeFunctions::slow_conv3d_forward_out(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor> & bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = slow_conv3d_npu_output_size(
      input, weight, bias, stride, padding);
  OpPreparation::CheckOut(
      {input, weight, bias},
      output,
      input,
      std::get<0>(outputSize));
  slow_conv3d_forward_npu_nocheck(
      input,
      weight,
      kernel_size,
      bias_opt,
      stride,
      padding,
      output);
  return output;
}

at::Tensor& NPUNativeFunctions::slow_conv3d_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor> & bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& output) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = slow_conv3d_npu_output_size(
      self, weight, bias, stride, padding);
  at::Tensor finput = OpPreparation::ApplyTensor(self, std::get<1>(outputSize));
  at::Tensor fgrad_input = at::empty({0}, self.options());

  return slow_conv3d_forward_npu_nocheck(
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output);
}

at::Tensor NPUNativeFunctions::slow_conv3d_forward(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto outputSize = slow_conv3d_npu_output_size(
      self, weight, bias, stride, padding);
  auto output = OpPreparation::ApplyTensorWithFormat(self, std::get<0>(outputSize), ACL_FORMAT_NDC1HWC0);
  auto finput = OpPreparation::ApplyTensorWithSizes({0}, self.options());
  auto fgrad_input = OpPreparation::ApplyTensorWithSizes({0}, self.options());


  slow_conv3d_forward_npu_nocheck(
      self,
      weight,
      kernel_size,
      bias_opt,
      stride,
      padding,
      output);

  return output;
}

at::Tensor NPUNativeFunctions::slow_conv3d(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor> & bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  return NPUNativeFunctions::slow_conv3d_forward(
      self, weight, kernel_size, bias, stride, padding);
}

} // namespace native
} // namespace at_npu
