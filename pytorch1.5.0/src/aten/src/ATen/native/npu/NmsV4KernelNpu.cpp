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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> nms_v4_npu_input(
    const Tensor& self,
    const Tensor& scores,
    Scalar max_output_size,
    const Tensor& iou_threshold,
    const Tensor& scores_threshold) {
  SmallVector<NPUTensorDesc, N> inputs;

  Tensor max_output_size_tensor = at::empty_with_format(
      {}, self.options().dtype(at::kInt), CalcuOpUtil::get_tensor_npu_format(self))
      .fill_(max_output_size);
  return CalcuOpUtil::create_npu_input_tensor_desc({self, scores, max_output_size_tensor, iou_threshold, scores_threshold});
}

SmallVector<NPUTensorDesc, N> nms_v4_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> nms_v4_npu_attr(bool pad_to_max_output_size) {
  NPUAttrDesc npuAttrPadToMaxOutputSize =
      NPUAttrDesc("pad_to_max_output_size", pad_to_max_output_size);

  SmallVector<NPUAttrDesc, N> attrs = {npuAttrPadToMaxOutputSize};
  return attrs;
}

tuple<Tensor, Tensor> nms_v4_out_npu(
    Tensor& selected_indices,
    Tensor& valid_outputs,
    const Tensor& self,
    const Tensor& scores,
    Scalar max_output_size,
    const Tensor& iou_threshold,
    const Tensor& scores_threshold,
    bool pad_to_max_output_size) {
  // constructs the input and output NPUTensorDesc
  auto inputs = nms_v4_npu_input(self, scores, max_output_size, iou_threshold, scores_threshold);
  auto outputs = nms_v4_npu_output({selected_indices, valid_outputs});

  // constructs the attr of the NPUAttrDesc
  auto attrs = nms_v4_npu_attr(pad_to_max_output_size);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("NonMaxSuppressionV4", inputs, outputs, attrs);

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
  Tensor selected_indices = at::empty_with_format(
      std::get<0>(outputSizes),
      self.options().dtype(at::kInt),
      CalcuOpUtil::get_tensor_npu_format(self));

  Tensor valid_outputs = at::empty_with_format(
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
