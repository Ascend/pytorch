// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

Tensor& upsample_bicubic2d_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  // calculate the output size
  int64_t N = self.size(0);
  int64_t C = self.size(1);
  int64_t H = output_size[0];
  int64_t W = output_size[1];

  SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};
  
  if (!result.sizes().equals(outputSize)) {
    result.resize_(outputSize);
  }

  TORCH_CHECK(
    output_size.size() == 2,
    "It is expected output_size equals to 2, but got size ",
    output_size.size());

  float temp_h = 0.0;
  float temp_w = 0.0;
  if (scales_h.has_value()) {
    temp_h = (float)scales_h.value();
  }
  if (scales_w.has_value()) {
    temp_w = (float)scales_w.value();
  }
  SmallVector<float, SIZE> scales = {temp_h, temp_w};
  SmallVector<float, SIZE> roi = {};
  string coordinate_transformation_mode = "half_pixel";
  if (align_corners == true) {
    coordinate_transformation_mode = "align_corners";
  }

  OpCommand cmd;
  cmd.Name("ResizeD")
      .Input(self)
      .Output(result)
      .Attr("sizes", output_size)
      .Attr("scales", scales)
      .Attr("roi", roi)
      .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
      .Attr("cubic_coeff_a", (float)-0.75)
      .Attr("exclude_outside", (int64_t)0)
      .Attr("extrapolation_value", (float)0.0)
      .Attr("mode", (string)"cubic")
      .Attr("nearest_mode", (string)"round_prefer_floor")
      .Run();
      
  return result;
}

Tensor upsample_bicubic2d_npu(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  
  // calculate the output size
  int64_t N = self.size(0);
  int64_t C = self.size(1);
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  SmallVector<int64_t, SIZE> outputSize = {N, C, H, W};

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  upsample_bicubic2d_out_npu(result, self, output_size, align_corners, scales_h, scales_w);
  
  return result;
}

} // namespace native
} // namespace at
