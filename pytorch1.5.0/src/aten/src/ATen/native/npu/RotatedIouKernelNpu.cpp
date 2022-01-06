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

Tensor& rotated_iou_npu_nocheck(
    Tensor& iou,
    const Tensor& boxes,
    const Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold) {
  string mode_str = (mode == 0) ? "iou" : "iof";   

  OpCommand cmd;
  cmd.Name("RotatedIou")
      .Input(boxes)
      .Input(query_boxes)
      .Output(iou)
      .Attr("trans", trans)
      .Attr("mode", mode_str)
      .Attr("is_cross", is_cross)
      .Attr("value", static_cast<float>(v_threshold))
      .Attr("value", static_cast<float>(e_threshold))
      .Run();
  return iou;
}

Tensor rotated_iou_npu(
    const Tensor& boxes,
    const Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold) {
  TORCH_CHECK(boxes.ndimension() == 3 && query_boxes.ndimension() == 3);
      
  auto origin_dtype = boxes.scalar_type();
 
  Tensor boxesOk = boxes.permute({0, 2, 1});
  if (boxesOk.scalar_type() == at::kHalf){
    boxesOk = boxesOk.npu_dtype_cast(at::kFloat);
  }
  Tensor query_boxesOk = query_boxes.permute({0, 2, 1});
  if (query_boxesOk.scalar_type() == at::kHalf){
    query_boxesOk = query_boxesOk.npu_dtype_cast(at::kFloat);
  }

  int64_t B = boxesOk.size(0);
  int64_t N = boxesOk.size(-1);
  int64_t K = query_boxesOk.size(-1);
 
  SmallVector<int64_t, SIZE> output_size({B, N, K});
  Tensor iou = OpPreparation::ApplyTensor(boxesOk, output_size);
 
  rotated_iou_npu_nocheck(iou, boxesOk, query_boxesOk, trans, mode, is_cross, v_threshold, e_threshold);
  iou = iou.npu_dtype_cast(origin_dtype);
  return iou;
} 

} // namespace native
} // namespace at
