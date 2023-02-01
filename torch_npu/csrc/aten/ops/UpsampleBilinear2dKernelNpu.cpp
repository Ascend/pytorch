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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"

namespace at_npu {
namespace native {

at::Tensor& upsample_bilinear2d_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  OpCommand cmd;
  bool half_pixel_centers = !align_corners;
  int64_t H = output_size[0];
  int64_t W = output_size[1];
  at::SmallVector<int64_t, N> attr_size = {H, W};
  cmd.Name("ResizeBilinearV2")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(attr_size, at::kInt)
      .Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("align_corners", align_corners)
      .Attr("half_pixel_centers", half_pixel_centers)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::upsample_bilinear2d_out(
    const at::Tensor& self_ex,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result){
  at::Tensor self = self_ex;
  if (self.scalar_type() != at::ScalarType::Float) {
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);

  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor result_contiguous = NpuUtils::format_contiguous(result);

    upsample_bilinear2d_out_npu_nocheck(
        result_contiguous, self, output_size, align_corners, scales_h, scales_w);
    NpuUtils::format_fresh_view(result, result_contiguous);
  } else {
    upsample_bilinear2d_out_npu_nocheck(
        result, self, output_size, align_corners, scales_h, scales_w);
  }
  return result;
}

at::Tensor NPUNativeFunctions::upsample_bilinear2d(
    const at::Tensor& self_ex,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at::Tensor self = self_ex;
  if (self.scalar_type() != at::ScalarType::Float) {
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options(), self);

  upsample_bilinear2d_out_npu_nocheck(
      result, self, output_size, align_corners, scales_h, scales_w);
  if (result.dtype() != self_ex.dtype()) {
    result = NPUNativeFunctions::npu_dtype_cast(result, self_ex.scalar_type());
  }
  return result;
}

at::Tensor NPUNativeFunctions::upsample_bilinear2d(
    const at::Tensor& self_ex,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  at::Tensor self = self_ex;
  auto osize = CalcuOpUtil::ComputeOutputSize(self_ex.sizes(), output_size, scale_factors);
  auto scales_h = CalcuOpUtil::GetScaleValue(scale_factors, 0);
  auto scales_w = CalcuOpUtil::GetScaleValue(scale_factors, 1);

  if (self.scalar_type() != at::ScalarType::Float) {
    self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, osize, align_corners, scales_h, scales_w);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, self.options(), self);

  upsample_bilinear2d_out_npu_nocheck(
      result, self, osize, align_corners, scales_h, scales_w);
  if (result.dtype() != self_ex.dtype()) {
    result = NPUNativeFunctions::npu_dtype_cast(result, self_ex.scalar_type());
  }
  return result;
}

} // namespace native
} // namespace at_npu
