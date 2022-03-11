// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

tuple<at::Tensor, at::Tensor> nms_v4_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& scores,
    at::Scalar max_output_size,
    const at::Tensor& iou_threshold,
    const at::Tensor& scores_threshold,
    bool pad_to_max_output_size,
    at::Tensor& selected_indices,
    at::Tensor& valid_outputs) {
  at::Tensor max_output_size_tensor = OpPreparation::ApplyTensor(
      {}, self.options().dtype(at::kInt), self).fill_(max_output_size);
  OpCommand cmd;
  cmd.Name("NonMaxSuppressionV4")
      .Input(self)
      .Input(scores)
      .Input(max_output_size_tensor)
      .Input(iou_threshold)
      .Input(scores_threshold)
      .Output(selected_indices)
      .Output(valid_outputs)
      .Attr("pad_to_max_output_size", pad_to_max_output_size)
      .Run();

  return std::tuple<at::Tensor, at::Tensor>(selected_indices, valid_outputs);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_nms_v4(
    const at::Tensor& self,
    const at::Tensor& scores,
    at::Scalar max_output_size,
    const at::Tensor& iou_threshold,
    const at::Tensor& scores_threshold,
    bool pad_to_max_output_size) {
  auto outputSizes = nms_v4_npu_output_size(max_output_size);

  at::Tensor selected_indices = OpPreparation::ApplyTensor(
      std::get<0>(outputSizes),
      self.options().dtype(at::kInt),
      self);
  at::Tensor valid_outputs = OpPreparation::ApplyTensor(
      std::get<1>(outputSizes),
      self.options().dtype(at::kInt),
      self);

  nms_v4_npu_nocheck(
      self,
      scores,
      max_output_size,
      iou_threshold,
      scores_threshold,
      pad_to_max_output_size,
      selected_indices,
      valid_outputs);

  return std::tuple<at::Tensor, at::Tensor>(selected_indices, valid_outputs);
}

} // namespace native
} // namespace at