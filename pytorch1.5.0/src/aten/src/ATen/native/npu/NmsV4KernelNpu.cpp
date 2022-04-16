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

#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor, Tensor> nms_v4_out_npu(
    Tensor& selected_indices,
    Tensor& valid_outputs,
    const Tensor& self,
    const Tensor& scores,
    Scalar max_output_size,
    const Tensor& iou_threshold,
    const Tensor& scores_threshold,
    bool pad_to_max_output_size) {
  Tensor max_output_size_tensor = OpPreparation::ApplyTensorWithFormat(
      {}, self.options().dtype(at::kInt), CalcuOpUtil::get_tensor_npu_format(self))
      .fill_(max_output_size);
          
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
      
  // return std::make_tuple(selected_indices, valid_outputs)
  return std::tuple<Tensor, Tensor>(selected_indices, valid_outputs);
}

tuple<Tensor, Tensor> nms_v4_npu(
    const Tensor& self,
    const Tensor& scores,
    Scalar max_output_size,
    const Tensor& iou_threshold,
    const Tensor& scores_threshold,
    bool pad_to_max_output_size) {
  // calculate the output size
  auto outputSizes = nms_v4_npu_output_size(max_output_size);

  // construct the output tensor of the NPU
  Tensor selected_indices = OpPreparation::ApplyTensorWithFormat(
      std::get<0>(outputSizes),
      self.options().dtype(at::kInt),
      CalcuOpUtil::get_tensor_npu_format(self));

  Tensor valid_outputs = OpPreparation::ApplyTensorWithFormat(
      std::get<1>(outputSizes),
      self.options().dtype(at::kInt),
      CalcuOpUtil::get_tensor_npu_format(self));

  nms_v4_out_npu(
      selected_indices,
      valid_outputs,
      self,
      scores,
      max_output_size,
      iou_threshold,
      scores_threshold,
      pad_to_max_output_size);

  return std::tuple<Tensor, Tensor>(selected_indices, valid_outputs);
}

} // namespace native
} // namespace at
