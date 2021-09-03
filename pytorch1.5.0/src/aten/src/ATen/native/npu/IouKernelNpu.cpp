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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> iou_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> iou_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> iou_npu_attr(int64_t mode) {
  string modeStr = "iou";
  if (mode == 1) {
    modeStr = "iof";
  }
  NPUAttrDesc npuAttrIou = NPUAttrDesc("mode", modeStr);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrIou};
  return attrs;
}

Tensor& iou_out_npu(
    Tensor& overlap,
    const Tensor& bboxes,
    const Tensor& gtboxes,
    int64_t mode) {
  // constructs the input and output NPUTensorDesc
  auto inputs = iou_npu_input({bboxes, gtboxes});
  auto outputs = iou_npu_output({overlap});

  // constructs the attr of the NPUAttrDesc
  auto attrs = iou_npu_attr(mode);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Iou", inputs, outputs, attrs);

  // return std::make_tuple(boxes, idx, mask);
  return overlap;
}

Tensor iou_npu(
    const Tensor& bboxes,
    const Tensor& gtboxes,
    int64_t mode) {
  // calculate the output size
  auto outputSize = iou_npu_output_size(bboxes, gtboxes);

  Tensor bboxesFP16 = bboxes;
  if (bboxes.scalar_type() != at::ScalarType::Half) {
    bboxesFP16 = bboxes.to(at::kHalf);
  }
  Tensor gtboxesFP16 = gtboxes;
  if (gtboxes.scalar_type() != at::ScalarType::Half) {
    gtboxesFP16 = gtboxes.to(at::kHalf);
  }

  // construct the output tensor of the NPU
  Tensor overlap = at::empty_with_format(outputSize, bboxesFP16.options(), CalcuOpUtil::get_tensor_npu_format(bboxes));

  iou_out_npu(overlap, bboxesFP16, gtboxesFP16, mode);

  if (overlap.scalar_type() != bboxes.scalar_type()) {
    overlap = overlap.to(bboxes.scalar_type());
  }

  return overlap;
}

} // namespace native
} // namespace at
