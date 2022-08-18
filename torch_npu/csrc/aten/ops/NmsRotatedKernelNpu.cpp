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

namespace at_npu {
namespace native {
tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_nms_rotated(const at::Tensor& dets, const at::Tensor& scores,
                                                                  double iouThreshold, double scoreThreshold,
                                                                  int64_t maxOutputSize, int64_t mode) {
  // the Op only support fp32 currently!
  auto originDtype = dets.scalar_type();
  at::Tensor detsCast = dets;
  at::Tensor scoresCast = scores;
  at::Tensor labels = at::zeros({}, scores.options().dtype(at::kInt));
  if (originDtype != at::ScalarType::Float) {
    detsCast = NPUNativeFunctions::npu_dtype_cast(dets, at::kFloat);
    scoresCast = NPUNativeFunctions::npu_dtype_cast(scores, at::kFloat);
  }
  c10::SmallVector<int64_t, SIZE> selectedIndexSize = {dets.size(0)};
  at::Tensor selectedBox = OpPreparation::ApplyTensor(dets);
  at::Tensor selectedIndex = OpPreparation::ApplyTensor(selectedIndexSize, dets.options().dtype(at::kInt), dets);

  c10::SmallVector<int64_t, N> output_sync_idx = {0, 1};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("RotatedNMS")
      .Input(detsCast)
      .Input(scoresCast)
      .Input(labels)
      .Output(selectedBox)
      .Output(selectedIndex)
      .Attr("iou_threshold", (float)iouThreshold)
      .Attr("score_threshold", (float)scoreThreshold)
      .Attr("max_output_size", maxOutputSize)
      .Attr("mode", mode)
      .Run();

  at::Tensor selectedNum =
      OpPreparation::ApplyTensor({1}, scores.options().dtype(at::kInt), scores).fill_(selectedIndex.size(0));
  return std::tie(selectedIndex, selectedNum);
}

}  // namespace native
}  // namespace at_npu