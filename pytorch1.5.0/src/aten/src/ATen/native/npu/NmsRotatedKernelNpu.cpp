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

tuple<Tensor, Tensor> nms_rotated_npu(
    const Tensor& dets,
    const Tensor& scores,
    double iouThreshold,
    double scoreThreshold,
    int64_t maxOutputSize,
    int64_t mode) {  
  SmallVector<int64_t, SIZE> selectedIndexSize = {dets.size(0)};
  SmallVector<int64_t, SIZE> selectedNumSize = {1};
   
  Tensor selectedIndex = OpPreparation::ApplyTensor(selectedIndexSize, dets.options().dtype(at::kInt), dets);
  Tensor selectedNum = OpPreparation::ApplyTensor(selectedNumSize, dets.options().dtype(at::kInt), dets);
  
  // the Op only support fp32 currently!
  auto originDtype = dets.scalar_type();
  Tensor detsCast = dets;
  Tensor scoresCast = scores;
  if(originDtype != at::ScalarType::Float){
    detsCast = dets.npu_dtype_cast(at::kFloat);
    scoresCast = scores.npu_dtype_cast(at::kFloat);
  }
 
  OpCommand cmd;
  cmd.Name("PolyNMS")
      .Input(detsCast)
      .Input(scoresCast)
      .Output(selectedIndex)
      .Output(selectedNum)
      .Attr("iou_threshold", (float)iouThreshold)
      .Attr("score_threshold", (float)scoreThreshold)
      .Attr("max_output_size", maxOutputSize)
      .Attr("mode", mode)
      .Run();
  
  Tensor selectedInd = selectedIndex.slice(0, 0, selectedNum.item().toLong());
  return std::tie(selectedInd, selectedNum);
}

} // namespace native
} // namespace at
