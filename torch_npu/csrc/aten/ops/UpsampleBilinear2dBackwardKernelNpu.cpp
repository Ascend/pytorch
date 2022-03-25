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

bool upsample_bilinear2d_backward_check_is_aicore(
    const at::Tensor& grad_output) {
  int64_t H = grad_output.size(2);
  int64_t W = grad_output.size(3);
  // 判断H或W大于10000走ai_cpu算子
  if (H > 10000 || W > 10000) {
    return false;
  }
  return true;
}

at::Tensor& upsample_bilinear2d_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  bool isAicore = upsample_bilinear2d_backward_check_is_aicore(grad_output);
  OpCommand cmd;
  if (isAicore) {
    at::Tensor original_image = OpPreparation::ApplyTensor(grad_output, input_size);
    bool half_pixel_centers = !align_corners;
    cmd.Name("ResizeBilinearV2Grad")
      .Input(grad_output)
      .Input(original_image)
      .Output(grad_input)
      .Attr("align_corners", align_corners)
      .Attr("half_pixel_centers", half_pixel_centers)
      .Run();
  } else {
    cmd.Name("PTUpsampleBilinear2dGrad")
      .Input(grad_output)
      .Output(grad_input)
      .Attr("output_size", output_size)
      .Attr("input_size", input_size)
      .Attr("align_corners", align_corners);
    if (scales_h.has_value()) {
      cmd.Attr("scales_h", static_cast<float>(scales_h.value()));
    }
    if (scales_w.has_value()) {
      cmd.Attr("scales_w", static_cast<float>(scales_w.value()));
    }
    cmd.Run();
  }
  return grad_input;
}

at::Tensor& NPUNativeFunctions::upsample_bilinear2d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {

  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      grad_output,
      input_size);
  if (!NpuUtils::check_match(&grad_input)) {
    at::Tensor result_contiguous = NpuUtils::format_contiguous(grad_input);

    upsample_bilinear2d_backward_out_npu_nocheck(
        result_contiguous,
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w);
    NpuUtils::format_fresh_view(grad_input, result_contiguous);
  } else {
    upsample_bilinear2d_backward_out_npu_nocheck(
        grad_input,
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h,
        scales_w);
  }
  return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_bilinear2d_backward(
    const at::Tensor& grad_output_ex,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at::Tensor grad_output = grad_output_ex;
  bool isAicore = upsample_bilinear2d_backward_check_is_aicore(grad_output);
  if (!isAicore) {
    if (grad_output.scalar_type() != at::ScalarType::Float) {
      grad_output = NPUNativeFunctions::npu_dtype_cast(grad_output, at::ScalarType::Float);
    }
  }
  auto outputSize = upsample_bilinear2d_backward_npu_output_size(
      grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  aclFormat format = isAicore ? ACL_FORMAT_NC1HWC0 : (aclFormat)CalcuOpUtil::get_tensor_npu_format(grad_output);
  at::Tensor grad_input = OpPreparation::ApplyTensorWithFormat(grad_output, outputSize, format);

  upsample_bilinear2d_backward_out_npu_nocheck(
      grad_input,
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_h,
      scales_w);
  if (grad_input.dtype() != grad_output_ex.dtype()) {
    grad_input = grad_input.to(grad_output_ex.dtype());
  }
  return grad_input;
}

at::Tensor NPUNativeFunctions::upsample_bilinear2d_backward(
    const at::Tensor& grad_output_ex,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = CalcuOpUtil::compute_output_size(input_size, output_size, scale_factors);
  auto scales_h = CalcuOpUtil::get_scale_value(scale_factors, 0);
  auto scales_w = CalcuOpUtil::get_scale_value(scale_factors, 1);

  at::Tensor grad_output = grad_output_ex;
  bool isAicore = upsample_bilinear2d_backward_check_is_aicore(grad_output);
  if (!isAicore) {
    if (grad_output.scalar_type() != at::ScalarType::Float) {
      grad_output = NPUNativeFunctions::npu_dtype_cast(grad_output, at::ScalarType::Float);
    }
  }
  auto outputSize = upsample_bilinear2d_backward_npu_output_size(
      grad_output, osize, input_size, align_corners, scales_h, scales_w);
  aclFormat format = isAicore ? ACL_FORMAT_NC1HWC0 : (aclFormat)CalcuOpUtil::get_tensor_npu_format(grad_output);
  at::Tensor grad_input = OpPreparation::ApplyTensorWithFormat(grad_output, outputSize, format);

  upsample_bilinear2d_backward_out_npu_nocheck(
      grad_input,
      grad_output,
      osize,
      input_size,
      align_corners,
      scales_h,
      scales_w);
  if (grad_input.dtype() != grad_output_ex.dtype()) {
    grad_input = grad_input.to(grad_output_ex.dtype());
  }
  return grad_input;
}

} // namespace native
} // namespace at_npu
