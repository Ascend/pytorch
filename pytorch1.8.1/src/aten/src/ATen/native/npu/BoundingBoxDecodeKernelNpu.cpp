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

Tensor& bounding_box_decode_npu_nocheck(
    Tensor& result,
    const Tensor& rois,
    const Tensor& deltas,
    SmallVector<float, SIZE> means,
    SmallVector<float, SIZE> stds,
    IntArrayRef max_shape,
    double wh_ratio_clip) {
  OpCommand cmd;
  cmd.Name("BoundingBoxDecode")
       .Input(rois)
       .Input(deltas)
       .Output(result)
       .Attr("means", means)
       .Attr("stds", stds)
       .Attr("max_shape", max_shape)
       .Attr("wh_ratio_clip", static_cast<float>(wh_ratio_clip))
       .Run();

  return result;
}

Tensor bounding_box_decode_npu(
    const Tensor& rois,
    const Tensor& deltas,
    double means0,
    double means1,
    double means2,
    double means3,
    double stds0,
    double stds1,
    double stds2,
    double stds3,
    IntArrayRef max_shape,
    double wh_ratio_clip) {
  SmallVector<int64_t, SIZE> outputSize = {rois.size(0), 4};
  Tensor result = OpPreparation::ApplyTensor(rois, outputSize);
  SmallVector<float, SIZE> means = {
      static_cast<float>(means0),
      static_cast<float>(means1),
      static_cast<float>(means2),
      static_cast<float>(means3)};
  SmallVector<float, SIZE> stds = {
      static_cast<float>(stds0),
      static_cast<float>(stds1),
      static_cast<float>(stds2),
      static_cast<float>(stds3)};

  bounding_box_decode_npu_nocheck(
      result, rois, deltas, means, stds, max_shape, wh_ratio_clip);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_bounding_box_decode", TORCH_FN(bounding_box_decode_npu));
}

} // namespace native
} // namespace at
