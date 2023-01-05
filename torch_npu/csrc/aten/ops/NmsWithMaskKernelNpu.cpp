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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&, at::Tensor&> nms_with_mask_npu_nocheck(
    const at::Tensor& input,
    at::Scalar iou_threshold,
    at::Tensor& boxes,
    at::Tensor& idx,
    at::Tensor& mask) {
  float iouThresholdValue = CalcuOpUtil::get_scalar_float_value(iou_threshold);
  OpCommand cmd;
  cmd.Name("NMSWithMask")
      .Input(input)
      .Output(boxes)
      .Output(idx)
      .Output(mask)
      .Attr("iou_threshold", iouThresholdValue)
      .Run();
  return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(boxes, idx, mask);
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_nms_with_mask(
    const at::Tensor& input,
    const at::Scalar& iou_threshold) {
  auto outputSizes = nms_with_mask_npu_output_size(input);
  at::Tensor boxes = OpPreparation::ApplyTensor(input, std::get<0>(outputSizes));
  at::Tensor idx = OpPreparation::ApplyTensor(std::get<1>(outputSizes), input.options().dtype(at::kInt), input);
  at::Tensor mask = OpPreparation::ApplyTensor(std::get<2>(outputSizes), input.options().dtype(at::kByte), input);
  nms_with_mask_npu_nocheck(input, iou_threshold, boxes, idx, mask);
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(boxes, idx, mask);
}

} // namespace native
} // namespace at_npu
