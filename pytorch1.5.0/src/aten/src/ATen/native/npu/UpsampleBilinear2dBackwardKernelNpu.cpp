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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;
Tensor& upsample_bilinear2d_backward_out_npu_nocheck(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& original_image,
    bool align_corners) {
  bool half_pixel_centers = !align_corners;
  OpCommand cmd;
  cmd.Name("ResizeBilinearV2Grad")
    .Input(grad_output)
    .Input(original_image)
    .Output(grad_input)
    .Attr("align_corners", align_corners)
    .Attr("half_pixel_centers", half_pixel_centers)
    .Run();
  return grad_input;
}

Tensor& upsample_bilinear2d_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor original_image = OpPreparation::ApplyTensor(grad_output, input_size);

  OpPreparation::CheckOut(
      {grad_output},
      grad_input,
      grad_output);
  if (!NpuUtils::check_match(&grad_input)) {
    Tensor contiguous_result = NpuUtils::format_contiguous(grad_input);

    upsample_bilinear2d_backward_out_npu_nocheck(
        contiguous_result, grad_output, original_image, align_corners);
    NpuUtils::format_fresh_view(grad_input, contiguous_result);
  } else {
    upsample_bilinear2d_backward_out_npu_nocheck(
        grad_input, grad_output, original_image, align_corners);
  }
  return grad_input;
}

Tensor upsample_bilinear2d_backward_npu(
    const Tensor& grad_output_ex,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor grad_output = grad_output_ex;

  if (grad_output.scalar_type() != ScalarType::Float) {
    grad_output = grad_output.npu_dtype_cast(ScalarType::Float);
  }

  Tensor grad_input = OpPreparation::ApplyTensor(grad_output, input_size);
  Tensor original_image = OpPreparation::ApplyTensor(grad_output, input_size);

  upsample_bilinear2d_backward_out_npu_nocheck(
      grad_input, grad_output, original_image, align_corners);
  if (grad_input.dtype() != grad_output_ex.dtype()) {
    grad_input = grad_input.to(grad_output_ex.dtype());
  }
  return grad_input;
}

} // namespace native
} // namespace at
