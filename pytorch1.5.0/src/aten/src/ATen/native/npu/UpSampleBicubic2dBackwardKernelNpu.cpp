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

#include <vector>
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& upsample_bicubic2d_backward_out_npu(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
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
  if (scales_h.has_value()) {
    temp_h = (float)scales_h.value();
  }
  if (scales_w.has_value()) {
    temp_w = (float)scales_w.value();
  }
  SmallVector<float, N> scales = {temp_h, temp_w};
  SmallVector<float, N> roi = {};
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
      .Attr("original_size", input_size)
      .Attr("roi", roi)
      .Attr("scales", scales)
      .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
      .Attr("cubic_coeff_a", cu)
      .Attr("exclude_outside", ex)
      .Attr("extrapolation_value", ext)
      .Attr("mode", mode)
      .Attr("nearest_mode", ne)
      .Run();
  return grad_input;
}

Tensor upsample_bicubic2d_backward_npu(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // construct the output tensor of the NPU
  auto outputSize = upsample_bicubic2d_backward_npu_output_size(input_size);
  Tensor result = OpPreparation::ApplyTensor(grad_output, outputSize);
  // calculate the output result of the NPU
  return upsample_bicubic2d_backward_out_npu(
      result,
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_h,
      scales_w);
}
} // namespace native
} // namespace at