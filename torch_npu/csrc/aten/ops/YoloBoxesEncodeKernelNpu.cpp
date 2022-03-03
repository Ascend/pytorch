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

static inline void yolo_boxes_encode_check(
    const at::Tensor& anchor_boxes,
    const at::Tensor& gt_bboxes,
    const at::Tensor& stride){
  TORCH_CHECK(
      anchor_boxes.dim() == 2 && anchor_boxes.size(1) == 4,
      "Non-empty 2D anchor_boxes tensor expected but got a tensor with sizes ",
      anchor_boxes.sizes());
  TORCH_CHECK(
      anchor_boxes.size(0) <= 20480,
      "anchor_boxes only support max [20480] num, but got num ",
      anchor_boxes.size(0));
  TORCH_CHECK(
      gt_bboxes.dim() == 2 && gt_bboxes.size(1) == 4,
      "Non-empty 2D gt_bboxes tensor expected but got a tensor with sizes ",
      gt_bboxes.sizes());
  TORCH_CHECK(
      stride.dim() == 1,
      "Non-empty 1D stride tensor expected but got a tensor with sizes ",
      stride.sizes());
  TORCH_CHECK(
      stride.size(0) == gt_bboxes.size(0),
      "stride's length should be equal gt_bboxes' num, but got stride length ",
      stride.size(0),
      "gt_bboxes num ",
      gt_bboxes.size(0));
  TORCH_CHECK(
      at::isIntegralType(stride.scalar_type(), true) && stride.scalar_type() != at::ScalarType::Long,
      "int32 strdie tensor expected but got a tensor with dtype: ",
      stride.scalar_type());
}

at::Tensor NPUNativeFunctions::npu_yolo_boxes_encode(
    const at::Tensor& anchor_boxes, 
    const at::Tensor& gt_bboxes, 
    const at::Tensor& stride,
    bool performance_mode){
  yolo_boxes_encode_check(anchor_boxes, gt_bboxes, stride);
  at::Tensor result = OpPreparation::ApplyTensor(gt_bboxes);
  string implModeStr = performance_mode ? "high_performance" : "high_precision";
  at::Tensor strideCp = NPUNativeFunctions::npu_dtype_cast(stride, at::ScalarType::Int);
  OpCommand cmd;
  cmd.Name("YoloBoxesEncode")
      .Input(anchor_boxes)
      .Input(gt_bboxes)
      .Input(strideCp)
      .Output(result)
      .Attr("performance_mode", implModeStr)
      .Run();
  return result;
}

} // namespace native
} // namespace at_npu