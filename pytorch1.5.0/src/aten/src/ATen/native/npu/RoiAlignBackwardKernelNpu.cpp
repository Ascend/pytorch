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

Tensor& roi_align_backward_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& rois,
    IntArrayRef xdiff_shape,
    int64_t pooled_width,
    int64_t pooled_height,
    double spatial_scale,
    int64_t sample_num,
    optional<int64_t> roi_end_mode) {
  OpCommand cmd;
  cmd.Name("ROIAlignGrad")
      .Input(self)
      .Input(rois)
      .Output(result)
      .Attr("xdiff_shape", xdiff_shape)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("pooled_height", pooled_height)
      .Attr("pooled_width", pooled_width)
      .Attr("sample_num", sample_num);
  if (roi_end_mode.has_value()) {
    cmd.Attr("roi_end_mode", roi_end_mode.value());
  }
  cmd.Run();

  return result;
}

Tensor roi_align_backward_npu(
    const Tensor& self,
    const Tensor& rois,
    IntArrayRef xdiff_shape,
    int64_t pooled_width,
    int64_t pooled_height,
    double spatial_scale,
    int64_t sample_num,
    optional<int64_t> roi_end_mode) {
  // construct the output tensor of the NPU
  Tensor result =
      OpPreparation::ApplyTensorWithFormat(self, xdiff_shape, ACL_FORMAT_NC1HWC0);

  //Check the self empty
  for (int i = 0; i < self.dim(); i++) {
      if (self.size(i) == 0) {
          result.fill_(0);
          return result;
      }
  }

  // calculate the output result of the NPU
  roi_align_backward_out_npu(
      result,
      self,
      rois,
      xdiff_shape,
      pooled_width,
      pooled_height,
      spatial_scale,
      sample_num,
      roi_end_mode);

  return result;
}

} // namespace native
} // namespace at