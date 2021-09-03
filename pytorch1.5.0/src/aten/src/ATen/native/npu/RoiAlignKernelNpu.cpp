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

SmallVector<int64_t, SIZE> roi_align_npu_output_size(
    const Tensor& self,
    const Tensor& rois,
    int64_t pooled_height,
    int64_t pooled_width) {
  return {
      rois.size(0),
      self.size(1),
      pooled_height,
      pooled_width}; //{N, C, H1, W1}
}

Tensor& roi_align_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sample_num,
    int64_t roi_end_mode) {
  OpCommand cmd;
  cmd.Name("ROIAlign")
      .Input(self)
      .Input(rois)
      .Output(result)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("pooled_height", pooled_height)
      .Attr("pooled_width", pooled_width)
      .Attr("sample_num", sample_num)
      .Attr("roi_end_mode", roi_end_mode)
      .Run();

  return result;
}

Tensor roi_align_npu(
    const Tensor& self,
    const Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sample_num,
    int64_t roi_end_mode) {
  Tensor selfCast = self;
  Tensor roisCast = rois;
  if (self.scalar_type() == kHalf || rois.scalar_type() == kHalf) {
    selfCast = self.to(kFloat);
    roisCast = rois.to(kFloat);
  }

  // calculate the output size
  auto outputSize =
      roi_align_npu_output_size(self, rois, pooled_height, pooled_width);

  // construct the output tensor of the NPU
  Tensor result =
      OpPreparation::ApplyTensorWithFormat(self, outputSize, ACL_FORMAT_NC1HWC0);

  // calculate the output result of the NPU
  roi_align_out_npu(
      result,
      self,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sample_num,
      roi_end_mode);

  if (self.scalar_type() == kHalf || rois.scalar_type() == kHalf) {
    result = result.to(kHalf);
  }

  return result;
}

} // namespace native
} // namespace at