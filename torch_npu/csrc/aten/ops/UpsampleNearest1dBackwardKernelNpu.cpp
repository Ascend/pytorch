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
c10::SmallVector<int64_t, SIZE> upsample_nearest1d_backward_output_size(at::IntArrayRef input_size) {
  c10::SmallVector<int64_t, SIZE> outputSize;
  int64_t N = input_size[0];
  int64_t C = input_size[1];
  int64_t W = input_size[2];
  outputSize = {N, C, 1, W};
  return outputSize;
}

at::Tensor& upsample_nearest1d_backward_out_nocheck(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& grad_input) {
  at::Tensor grad_cp = grad_output.unsqueeze(2);
  OpCommand cmd;
  if (grad_output.scalar_type() == at::kFloat || grad_output.scalar_type() == at::kHalf) {
    c10::SmallVector<int64_t, SIZE> result_size = {1, input_size[2]};
    cmd.Name("ResizeNearestNeighborV2Grad")
        .Input(grad_cp)
        .Input(result_size, at::kInt)
        .Output(grad_input)
        .Attr("align_corners", false)
        .Attr("half_pixel_centers", false)
        .Run();
  } else {
    c10::SmallVector<int64_t, SIZE> origin_size = upsample_nearest1d_backward_output_size(input_size);
    at::Scalar scales_cp = scales.has_value() ? scales.value() : input_size[0] / output_size[0];
    cmd.Name("ResizeGrad")
        .Input(grad_cp)
        .Input(scales_cp, at::kFloat)
        .Input(scales_cp, at::kFloat)
        .Input(origin_size, at::kLong)
        .Output(grad_input)
        // Default value of Resize
        .Attr("coordinate_transformation_mode", (string)"pytorch_half_pixel")
        .Attr("cubic_coeff_a", (float)-0.75)
        .Attr("exclude_outside", (int64_t)0)
        .Attr("extrapolation_value", (float)0.0)
        .Attr("mode", (string)"nearest")
        .Attr("nearest_mode", (string)"floor")
        .Run();
  }
  grad_input = grad_input.squeeze(2);
  return grad_input;
}

at::Tensor& NPUNativeFunctions::upsample_nearest1d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& grad_input) {
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_backward_output_size(input_size);
  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      CalcuOpUtil::get_tensor_npu_format(grad_output),
      grad_output.scalar_type(),
      outputSize);

  if (!NpuUtils::check_match(&grad_input)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(grad_input);
    upsample_nearest1d_backward_out_nocheck(grad_output, output_size, input_size, scales, contiguousResult);
    NpuUtils::format_fresh_view(grad_input, contiguousResult);
  } else {
    upsample_nearest1d_backward_out_nocheck(grad_output, output_size, input_size, scales, grad_input);
  }

   return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_nearest1d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input_size, output_size, scale_factors);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 0);
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_backward_output_size(input_size);
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output, outputSize);

  upsample_nearest1d_backward_out_nocheck(grad_output, osize, input_size, scales_w, grad_input);
  return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_nearest1d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales) {
  c10::SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_backward_output_size(input_size);
  at::Tensor grad_input = OpPreparation::ApplyTensor(grad_output, outputSize);

  upsample_nearest1d_backward_out_nocheck(
      grad_output, output_size, input_size, scales, grad_input);
  return grad_input;
}
} // namespace native
} // namespace at_npu
