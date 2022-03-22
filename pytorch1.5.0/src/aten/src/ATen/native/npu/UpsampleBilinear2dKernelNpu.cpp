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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& upsample_bilinear2d_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  OpCommand cmd;
  bool half_pixel_centers = !align_corners;
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  SmallVector<int64_t, N> attr_size = {H, W};
  cmd.Name("ResizeBilinearV2")
    .Input(self)
    .Input(attr_size, at::kInt)
    .Output(result)
    .Attr("align_corners", align_corners)
    .Attr("half_pixel_centers", half_pixel_centers)
    .Run();
  return result;
}

Tensor& upsample_bilinear2d_out_npu(
    Tensor& result,
    const Tensor& self_ex,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w){
  Tensor self = self_ex;
  if (self.scalar_type() != ScalarType::Float) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);

  OpPreparation::CheckOut(
      {self},
      result,
      (aclFormat)CalcuOpUtil::get_tensor_npu_format(self),
      ScalarType::Float,
      outputSize);

  upsample_bilinear2d_out_npu_nocheck(
      result, self, output_size, align_corners, scales_h, scales_w);
  return result;
}

Tensor upsample_bilinear2d_npu(
    const Tensor& self_ex,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor self = self_ex;
  if (self.scalar_type() != ScalarType::Float) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);
  Tensor result = OpPreparation::ApplyTensor(outputSize, self.options(), self);

  upsample_bilinear2d_out_npu_nocheck(
      result, self, output_size, align_corners, scales_h, scales_w);
  if (result.dtype() != self_ex.dtype()) {
    result = result.to(self_ex.dtype());
  }
  return result;
}
} // namespace native
} // namespace at
