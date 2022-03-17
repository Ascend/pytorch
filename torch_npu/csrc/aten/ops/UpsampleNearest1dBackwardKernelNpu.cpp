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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& upsample_nearest1d_backward_out_nocheck(
    const at::Tensor& grads,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& y) {
  OpCommand cmd;
  cmd.Name("UpsampleNearest1dGrad")
      .Input(grads)
      .Output(y)
      .Attr("output_size", output_size)
      .Attr("input_size", input_size);
      if (scales.has_value()) {
        cmd.Attr("scales", static_cast<float>(scales.value()));
      }
      cmd.Run();

   return y;
}

at::Tensor& NPUNativeFunctions::upsample_nearest1d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& y) {
  at::Tensor grads = grad_output;
  if (grad_output.scalar_type() != at::ScalarType::Float) {
    grads = grad_output.to(at::kFloat);
  }

  OpPreparation::CheckOut(
      {grad_output},
      y,
      CalcuOpUtil::get_tensor_npu_format(grad_output),
      grads.scalar_type(),
      input_size);

  if (!NpuUtils::check_match(&y)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(y);
    upsample_nearest1d_backward_out_nocheck(grads, output_size, input_size, scales, contiguousResult);
    NpuUtils::format_fresh_view(y, contiguousResult);
  } else {
    upsample_nearest1d_backward_out_nocheck(grads, output_size, input_size, scales, y);
  }

   return y;
}

at::Tensor NPUNativeFunctions::upsample_nearest1d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input_size, output_size, scale_factors);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 0);
  at::Tensor grads = grad_output;
  if (grad_output.scalar_type() != at::ScalarType::Float) {
    grads = grad_output.to(at::kFloat);
  }
  at::Tensor grad_input = OpPreparation::ApplyTensor(input_size, grads.options(), grad_output);
  upsample_nearest1d_backward_out_nocheck(grads, osize, input_size, scales_w, grad_input);
  return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_nearest1d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales) {
  at::Tensor grads = grad_output;
  if (grad_output.scalar_type() != at::ScalarType::Float) {
    grads = grad_output.to(at::kFloat);
  }

  at::Tensor grad_input = OpPreparation::ApplyTensor(input_size, grads.options(), grad_output);

  upsample_nearest1d_backward_out_nocheck(
      grads, output_size, input_size, scales, grad_input);
  return grad_input;
}
} // namespace native
} // namespace at_npu
