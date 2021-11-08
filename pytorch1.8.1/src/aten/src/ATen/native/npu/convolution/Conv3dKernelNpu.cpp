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

SmallVector<int64_t, SIZE>
conv3d_npu_output_size(const Tensor &input, const Tensor &weight,
                       const Tensor &bias, IntArrayRef stride,
                       IntArrayRef padding, IntArrayRef dilation,
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

  SmallVector<int64_t, SIZE> outputSize = {N, Co, Do, Ho, Wo};

  return outputSize;
}

Tensor &conv3d_out_npu_nocheck(Tensor &result, const Tensor &input,
                               const Tensor &weight, const Tensor &bias,
                               IntArrayRef stride, IntArrayRef padding,
                               IntArrayRef dilation, int64_t groups) {
  Tensor filter = weight.to(input.dtype());
  SmallVector<Tensor, N> inputTensor = {input, filter, bias};
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1], dilation[2]};

  OpCommand cmd;
  cmd.Name("Conv3D");
  cmd.Input(input);
  cmd.Input(filter);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(result);
  cmd.Attr("strides", stridesSize);
  cmd.Attr("pads", paddings);
  cmd.Attr("dilations", dilations);
  cmd.Attr("groups", groups);
  cmd.Attr("data_format", (string) "NCDHW");
  cmd.Run();

  return result;
}

Tensor &conv3d_out_npu(const Tensor &input,
                       const Tensor &weight,
                       const optional<Tensor> &bias_opt,
                       IntArrayRef stride,
                       IntArrayRef padding,
                       IntArrayRef dilation,
                       int64_t groups,
                       Tensor &result) {
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  auto outputSize = conv3d_npu_output_size(
      input, weight, bias, stride, padding, dilation, groups);
  OpPreparation::CheckOut(
      {input, weight, bias},
      result,
      input,
      outputSize);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({input, weight, bias}, {result})
             .Func([&input, &weight, &bias, stride, padding, dilation, groups](Tensor &result) {
                 conv3d_out_npu_nocheck(
                     result, input, weight, bias, stride, padding, dilation, groups);
              })
             .Call(result);
}

Tensor conv3d_npu(const Tensor &input, const Tensor &weight, const optional<Tensor> &bias_opt,
                  IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
                  int64_t groups) {
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

  auto outputSize = conv3d_npu_output_size(
      input, weight, bias, stride, padding, dilation, groups);

  Tensor result = OpPreparation::ApplyTensor(input, outputSize);
  conv3d_out_npu_nocheck(result, input, weight, bias, stride, padding, dilation, groups);

  return result;
}

tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>> slow_conv3d_npu_output_size(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    IntArrayRef stride,
    IntArrayRef padding) {
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

  SmallVector<int64_t, SIZE> outputSize = {N, Co, Do, Ho, Wo};
  SmallVector<int64_t, SIZE> finputSize = {
    N, C * kernel_size[0] * kernel_size[1] * kernel_size[2], Do * Ho * Wo};

  return tuple<SmallVector<int64_t, SIZE>, SmallVector<int64_t, SIZE>>(outputSize, finputSize);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv3d_forward_npu_nocheck(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const optional<Tensor> & bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output,
    Tensor& finput,
    Tensor& fgrad_input) {
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  Tensor filter = weight.to(input.dtype());
  SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1], stride[2]};
  SmallVector<int64_t, N> paddings = {padding[0], padding[0], padding[1],
                                      padding[1], padding[2], padding[2]};
  SmallVector<int64_t, N> dilations = {1, 1, 1, 1, 1};

  OpCommand cmd;
  cmd.Name("Conv3D");
  cmd.Input(input);
  cmd.Input(filter);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(output);
  cmd.Attr("strides", stridesSize);
  cmd.Attr("pads", paddings);
  cmd.Attr("dilations", dilations);
  cmd.Attr("data_format", (string) "NCDHW");
  cmd.Run();

  return std::tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv3d_forward_out_npu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const optional<Tensor> & bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output,
    Tensor& finput,
    Tensor& fgrad_input) {
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
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
      output,
      finput,
      fgrad_input);
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
}

Tensor& slow_conv3d_out_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const optional<Tensor> & bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  auto outputSize = slow_conv3d_npu_output_size(
      self, weight, bias, stride, padding);
  Tensor finput = OpPreparation::ApplyTensor(self, std::get<1>(outputSize));
  Tensor fgrad_input = at::empty({0}, self.options());

  return std::get<0>(slow_conv3d_forward_npu_nocheck(
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding,
      output,
      finput,
      fgrad_input));
}

std::tuple<Tensor, Tensor, Tensor> slow_conv3d_forward_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding) {
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  auto outputSize = slow_conv3d_npu_output_size(
      self, weight, bias, stride, padding);
  auto output = OpPreparation::ApplyTensor(self, std::get<0>(outputSize));
  auto finput = OpPreparation::ApplyTensor(self, {0});
  auto fgrad_input = OpPreparation::ApplyTensor(self, {0});


  slow_conv3d_forward_npu_nocheck(
      self,
      weight,
      kernel_size,
      bias_opt,
      stride,
      padding,
      output,
      finput,
      fgrad_input);

  return std::tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
}

Tensor slow_conv3d_npu(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const optional<Tensor> & bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  return std::get<0>(slow_conv3d_forward_npu(
      self, weight, kernel_size, bias, stride, padding));
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_conv3d", TORCH_FN(conv3d_npu));
  m.impl("npu_conv3d.out", TORCH_FN(conv3d_out_npu));
  m.impl("slow_conv3d.out", TORCH_FN(slow_conv3d_out_npu));
  m.impl("slow_conv3d", TORCH_FN(slow_conv3d_npu));
  m.impl("slow_conv3d_forward.output", TORCH_FN(slow_conv3d_forward_out_npu));
  m.impl("slow_conv3d_forward", TORCH_FN(slow_conv3d_forward_npu));
}

} // namespace native
} // namespace at
