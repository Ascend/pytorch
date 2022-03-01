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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::upsample_nearest2d_backward_out(
    const at::Tensor& grads,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& y) {
  at::SmallVector<int64_t, N> outputSize = {input_size[2], input_size[3]};
  OpCommand cmd;
  cmd.Name("ResizeNearestNeighborV2Grad")
      .Input(grads)
      .Input(outputSize, at::kInt)
      .Output(y)
      .Attr("align_corners", false)
      .Attr("half_pixel_centers", false)
      .Run();

   return y;
}

at::Tensor NPUNativeFunctions::upsample_nearest2d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at::Tensor grads = grad_output;
  if (grad_output.scalar_type() != at::ScalarType::Float) {
    grads = grad_output.to(at::kFloat);
  }
  at::Tensor grad_input = OpPreparation::ApplyTensor(
      input_size, grads.options(), grad_output);
  NPUNativeFunctions::upsample_nearest2d_backward_out(
      grads, output_size, input_size, scales_h, scales_w, grad_input);
  return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_nearest2d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input_size, output_size, scale_factors);
  auto scales_h = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 1);
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output, input_size);
  NPUNativeFunctions::upsample_nearest2d_backward_out(grad_output, osize, input_size, scales_h, scales_w, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at_npu