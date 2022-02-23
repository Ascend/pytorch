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
#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using namespace torch::autograd;

static inline void grid_assign_positive_check(
    const at::Tensor& argmax_overlaps,
    const at::Tensor& gt_argmax_overlaps){
  TORCH_CHECK(
      at::isIntegralType(argmax_overlaps.scalar_type(), true) && argmax_overlaps.scalar_type() != at::ScalarType::Long,
      "int32 argmax_overlaps tensor expected but got a tensor with dtype: ",
      argmax_overlaps.scalar_type());
  TORCH_CHECK(
      at::isIntegralType(gt_argmax_overlaps.scalar_type(), true) && gt_argmax_overlaps.scalar_type() != at::ScalarType::Long,
      "int32 gt_argmax_overlaps tensor expected but got a tensor with dtype: ",
      gt_argmax_overlaps.scalar_type());
}

at::Tensor NPUNativeFunctions::npu_grid_assign_positive(
    const at::Tensor& assigned_gt_inds,
    const at::Tensor& overlaps,
    const at::Tensor& box_responsible_flags,
    const at::Tensor& max_overlaps,
    const at::Tensor& argmax_overlaps,
    const at::Tensor& gt_max_overlaps,
    const at::Tensor& gt_argmax_overlaps,
    int64_t num_gts,
    double pos_iou_thr,
    double min_pos_iou,
    bool gt_max_assign_all){
  grid_assign_positive_check(argmax_overlaps, gt_argmax_overlaps);
  at::Tensor result = OpPreparation::ApplyTensor(assigned_gt_inds);
  auto option = assigned_gt_inds.options().dtype(at::kInt);
  at::Scalar s(num_gts);
  at::Tensor numOfGts = at::empty({}, option).fill_(s);

  at::Tensor argmaxOverLaps = NPUNativeFunctions::npu_dtype_cast(argmax_overlaps, at::ScalarType::Int);
  at::Tensor gtArgmaxOverLaps = NPUNativeFunctions::npu_dtype_cast(gt_argmax_overlaps, at::ScalarType::Int);
  
  OpCommand cmd;
  cmd.Name("GridAssignPositive")
      .Input(assigned_gt_inds)
      .Input(overlaps)
      .Input(box_responsible_flags)
      .Input(max_overlaps)
      .Input(argmaxOverLaps)
      .Input(gt_max_overlaps)
      .Input(gtArgmaxOverLaps)
      .Input(numOfGts)
      .Output(result)
      .Attr("pos_iou_thr", (float) pos_iou_thr)
      .Attr("min_pos_iou", (float) min_pos_iou)
      .Attr("gt_max_assign_all", gt_max_assign_all)
      .Run();
  return result;
}

} // namespace native
} // namespace at_npu
