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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bounding_box_encode_out_npu(
    Tensor& delats,
    const Tensor& anchor_box,
    const Tensor& ground_truth_box,
    SmallVector<float, SIZE> means,
    SmallVector<float, SIZE> stds) {
  OpCommand cmd;
  cmd.Name("BoundingBoxEncode")
       .Input(anchor_box)
       .Input(ground_truth_box)
       .Output(delats)
       .Attr("means", means)
       .Attr("stds", stds)
       .Run();

  return delats;
}

Tensor bounding_box_encode_npu(
    const Tensor& anchor_box,
    const Tensor& ground_truth_box,
    double means0,
    double means1,
    double means2,
    double means3,
    double stds0,
    double stds1,
    double stds2,
    double stds3) {
  // construct the output tensor of the NPU
  Tensor delats = at::empty_with_format(
      {anchor_box.size(0), 4},
      anchor_box.options(),
      CalcuOpUtil::get_tensor_npu_format(anchor_box));

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

  bounding_box_encode_out_npu(
      delats, anchor_box, ground_truth_box, means, stds);

  return delats;
}

} // namespace native
} // namespace at
