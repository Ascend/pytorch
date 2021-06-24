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

bool upsample_bilinear2d_backward_check_is_aicore(
    const Tensor& grad_output) {
  int64_t H = grad_output.size(2);
  int64_t W = grad_output.size(3);

  if (H > 10000 || W > 10000) {
    return false;
  }
  return true;
}

Tensor& upsample_bilinear2d_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // check shape to judge use aicpu op or aicore op
  bool isAicore = upsample_bilinear2d_backward_check_is_aicore(grad_output);

  // executing the NPU operator
  OpCommand cmd;
  if (isAicore) {
    Tensor original_image = at::empty_with_format(
        input_size,
        grad_output.options(),
        CalcuOpUtil::get_tensor_npu_format(grad_output));

    bool half_pixel_centers = !align_corners;

    cmd.Name("ResizeBilinearV2Grad")
      .Input(grad_output)
      .Input(original_image)
      .Output(grad_input) 
      .Attr("align_corners", align_corners)
      .Attr("half_pixel_centers", half_pixel_centers)
      .Run();
  } else {
    cmd = cmd.Name("PTUpsampleBilinear2dGrad")
      .Input(grad_output)
      .Output(grad_input)
      .Attr("output_size", output_size)
      .Attr("input_size", input_size)
      .Attr("align_corners", align_corners);

    // optional attr
    if (scales_h.has_value()) {
      cmd = cmd.Attr("scales_h", static_cast<float>(scales_h.value()));
    }
    if (scales_w.has_value()) {
      cmd = cmd.Attr("scales_w", static_cast<float>(scales_w.value()));
    }
    cmd.Run();
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
  // check shape to judge use aicpu op or aicore op
  bool isAicore = upsample_bilinear2d_backward_check_is_aicore(grad_output);

  if (!isAicore) {
    if (grad_output.scalar_type() != ScalarType::Float) {
      grad_output = grad_output.npu_dtype_cast(ScalarType::Float);
    }
  }
  // calculate the output size
  auto outputSize = upsample_bilinear2d_backward_npu_output_size(
      grad_output, output_size, input_size, align_corners, scales_h, scales_w);

  // construct the output tensor of the NPU
  aclFormat format = isAicore ? ACL_FORMAT_NC1HWC0 : (aclFormat)CalcuOpUtil::get_tensor_npu_format(grad_output);
  Tensor grad_input = at::empty_with_format(outputSize, grad_output.options(), format);

  // calculate the output result of the NPU
  upsample_bilinear2d_backward_out_npu(
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

} // namespace native
} // namespace at
