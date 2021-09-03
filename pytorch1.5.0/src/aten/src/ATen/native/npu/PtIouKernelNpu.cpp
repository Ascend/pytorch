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

Tensor ptiou_npu(
    const Tensor& bboxes,
    const Tensor& gtboxes,
    int64_t mode) {
  auto outputSize = {gtboxes.size(0), bboxes.size(0)};
  Tensor overlap = OpPreparation::ApplyTensor(bboxes, outputSize);
  string modeStr = "iou";
  if (mode == 1) {
    modeStr = "iof";
  }
  OpCommand cmd;
  cmd.Name("PtIou")
      .Input(bboxes)
      .Input(gtboxes)
      .Output(overlap)
      .Attr("mode", modeStr)
      .Run();

  return overlap;
}

} // namespace native
} // namespace at
