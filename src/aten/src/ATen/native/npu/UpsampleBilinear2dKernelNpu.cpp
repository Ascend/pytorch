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

namespace at {
namespace native {
using namespace at::native::npu;

bool upsample_bilinear2d_check_is_aicore(
    const Tensor& self,
    IntArrayRef output_size) {
  int64_t H = self.size(2);
  int64_t W = self.size(3);

  if (H > 2048 || W > 2048 || output_size[0] > 2048 || output_size[1] > 2048) {
    return false;
  }
  return true;
}

Tensor& upsample_bilinear2d_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // check shape to judge use aicpu op or aicore op
  bool isAicore = upsample_bilinear2d_check_is_aicore(self, output_size);

  OpCommand cmd;
  if (isAicore) {
    bool half_pixel_centers = !align_corners;

    int64_t H = output_size[0];
    int64_t W = output_size[1];
    SmallVector<int64_t, N> attr_size = {H, W};

    if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
      cmd.Name("ResizeBilinearV2")
        .Input(self)
        .Input(attr_size, at::kInt)
        .Output(result)
        .Attr("align_corners", align_corners)
        .Attr("half_pixel_centers", half_pixel_centers)
        .Run();
    } else {
      cmd.Name("ResizeBilinearV2D")
        .Input(self)
        .Output(result)
        .Attr("size", attr_size)
        .Attr("align_corners", align_corners)
        .Attr("half_pixel_centers", half_pixel_centers)
        .Run();    
    }
  } else {
    cmd = cmd.Name("PTUpsampleBilinear2D")
      .Input(self)
      .Output(result)
      .Attr("output_size", output_size)
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
  //check shape to judge use aicpu op or aicore op
  bool isAicore = upsample_bilinear2d_check_is_aicore(self, output_size);

  if (!isAicore) {
    if (self.scalar_type() != ScalarType::Float) {
      self = self.npu_dtype_cast(ScalarType::Float);
    }
  }
  aclFormat format = isAicore ? ACL_FORMAT_NC1HWC0 : (aclFormat)CalcuOpUtil::get_tensor_npu_format(self);
  // calculate the output size
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);

  OpPreparation::CheckOut(
    {self}, 
    result, 
    format, 
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
  //check shape to judge use aicpu op or aicore op
  bool isAicore = upsample_bilinear2d_check_is_aicore(self, output_size);

  if (!isAicore) {
    if (self.scalar_type() != ScalarType::Float) {
      self = self.npu_dtype_cast(ScalarType::Float);
    }
  }

  // calculate the output size
  auto outputSize = upsample_bilinear2d_npu_output_size(
      self, output_size, align_corners, scales_h, scales_w);

  // construct the output tensor of the NPU
  aclFormat format = isAicore ? ACL_FORMAT_NC1HWC0 : (aclFormat)CalcuOpUtil::get_tensor_npu_format(self);
  Tensor result = at::empty_with_format(outputSize, self.options().dtype(kFloat), format);

  // calculate the output result of the NPU
  upsample_bilinear2d_out_npu_nocheck(
      result, self, output_size, align_corners, scales_h, scales_w);

  if (result.dtype() != self_ex.dtype()) {
    result = result.to(self_ex.dtype());
  }

  return result;
}

} // namespace native
} // namespace at
