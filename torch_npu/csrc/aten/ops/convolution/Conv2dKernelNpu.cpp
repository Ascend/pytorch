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


namespace {
bool isSpecialConv1d(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  if (stride[1] > 63 &&
      stride[1] == weight.size(3) &&
      padding[1] == 0 &&
      dilation[1] == 1 &&
      groups == 1 &&
      input.size(1) == 1) {
    return true;
  } else {
    return false;
  }
} // isSpecialConv1d
} // namespace

at::Tensor& NPUNativeFunctions::npu_conv2d_out(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& result) {

  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});

  // constructs the input and output NPUTensorDesc
  c10::SmallVector<int64_t, N> stridesSize = {1, 1, stride[0], stride[1]};
  c10::SmallVector<int64_t, N> paddings = { padding[0], padding[0], padding[1], padding[1]};
  c10::SmallVector<int64_t, N> dilations = {1, 1, dilation[0], dilation[1]};
  OpCommand cmd;
  cmd.Name("Conv2D")
      .Input(input, "x", ACL_FORMAT_NCHW)
      .Input(weight, "filter", ACL_FORMAT_NCHW);
  if (bias.defined()) {
      cmd.Input(bias);
  }
  cmd.Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("groups", groups)
      .Attr("data_format", (string)"NCHW")
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::npu_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  // support special scenario
  if (isSpecialConv1d(input,
                      weight,
                      stride,
                      padding,
                      dilation,
                      groups)) {
    at::Tensor mmInput = input.view({input.size(0), input.size(3)/weight.size(3), weight.size(3)});
    at::Tensor mmOther = weight.view({weight.size(0), weight.size(3)})
                           .permute({1, 0});
    at::Tensor mmResult = at::matmul(mmInput, mmOther);
    at::Tensor result = mmResult.permute({0, 2, 1});
    return result;
  }

  // calculate the output size
  int64_t N = input.size(0);
  int64_t H = input.size(2);
  int64_t W = input.size(3);
  int64_t Co = weight.size(0);
  auto kernel_size = weight.sizes().slice(2);

  int64_t Ho = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
  int64_t Wo = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
  c10::SmallVector<int64_t, SIZE> outputSize = {N, Co, Ho, Wo};
  // construct the output tensor of the NPU
  const int64_t result_format = CalcuOpUtil::judge_and_get_format_from_input(
      CalcuOpUtil::get_tensor_npu_format(weight) == ACL_FORMAT_FRACTAL_Z,
      input, ACL_FORMAT_NC1HWC0);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(input, outputSize, result_format);
  // calculate the output result of the NPU
  NPUNativeFunctions::npu_conv2d_out(input, weight, bias, stride, padding, dilation, groups, result);
  return result;
}

} // namespace native
} // namespace at_npu