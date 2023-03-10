// Copyright (c) 2020, Huawei Technologies.
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

at::Tensor& upsample_bicubic2d_backward_out_nocheck(
    const at::Tensor& grad_output, 
    at::IntArrayRef output_size, 
    at::IntArrayRef input_size, 
    bool align_corners, 
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  float temp_h = 0.0;
  float temp_w = 0.0;
  if(scales_h.has_value()) {
    temp_h = (float)scales_h.value();
  } 
  if(scales_w.has_value()) {
    temp_w = (float)scales_w.value();
  }
  c10::SmallVector<float,N> scales = {temp_h, temp_w};
  c10::SmallVector<float, N> roi = {};
  string coordinate_transformation_mode =
      align_corners ? "align_corners" : "half_pixel";

  float cu = -0.75;
  int64_t ex = 0;
  float ext = 0.0;
  string mode = "cubic";
  string ne = "round_prefer_floor";

  OpCommand cmd;
  cmd.Name("ResizeGradD")
      .Input(grad_output, "grads", ACL_FORMAT_NCHW)
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("scales", scales)
      .Attr("roi", roi)
      .Attr("original_size", input_size)
      .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
      .Attr("cubic_coeff_a", cu)
      .Attr("exclude_outside", ex)
      .Attr("extrapolation_value", ext)
      .Attr("mode", mode)
      .Attr("nearest_mode", ne)
      .Run();

  return grad_input;
}

at::Tensor& NPUNativeFunctions::upsample_bicubic2d_backward_out(
    const at::Tensor& grad_output, 
    at::IntArrayRef output_size, 
    at::IntArrayRef input_size, 
    bool align_corners, 
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {

  auto outputSize = upsample_bicubic2d_backward_npu_output_size(input_size);

  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      CalcuOpUtil::GetTensorNpuFormat(grad_output),
      grad_output.scalar_type(),
      outputSize);
  
  if (!NpuUtils::check_match(&grad_input)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(grad_input);
    upsample_bicubic2d_backward_out_nocheck(grad_output, output_size, input_size, align_corners, scales_h, scales_w, contiguousResult);
    NpuUtils::format_fresh_view(grad_input, contiguousResult);
  } else {
    upsample_bicubic2d_backward_out_nocheck(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }

  return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_bicubic2d_backward(
    const at::Tensor& grad_output, 
    at::IntArrayRef output_size, 
    at::IntArrayRef input_size, 
    bool align_corners, 
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // construct the output tensor of the NPU
  auto outputSize = upsample_bicubic2d_backward_npu_output_size(input_size);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, grad_output.options(), CalcuOpUtil::GetTensorNpuFormat(grad_output));
  // calculate the output result of the NPU
  return upsample_bicubic2d_backward_out_nocheck(grad_output, output_size, input_size, align_corners, scales_h, scales_w, result);
}

at::Tensor NPUNativeFunctions::upsample_bicubic2d_backward(
    const at::Tensor& grad_output, 
    at::OptionalIntArrayRef output_size, 
    at::IntArrayRef input_size, 
    bool align_corners, 
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::ComputeOutputSize(input_size, output_size, scale_factors);
  auto scales_h = CalcuOpUtil::GetScaleValue(scale_factors, 0);
  auto scales_w = CalcuOpUtil::GetScaleValue(scale_factors, 1);
  // construct the output tensor of the NPU
  auto outputSize = upsample_bicubic2d_backward_npu_output_size(input_size);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, grad_output.options(), CalcuOpUtil::GetTensorNpuFormat(grad_output));
  // calculate the output result of the NPU
  return upsample_bicubic2d_backward_out_nocheck(grad_output, osize, input_size, align_corners, scales_h, scales_w, result);
}

} // namespace native
} // namespace at_npu