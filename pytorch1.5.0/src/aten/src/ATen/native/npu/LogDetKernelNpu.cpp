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
#include "c10/npu/npu_log.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> logdet_npu_output_size(const Tensor& self) {
  c10::SmallVector<int64_t, SIZE> dimVec;
  if (self.dim() > 2) {
    for (int i = 0; i < self.dim() - 2; i++) {
      dimVec.push_back(self.size(i));
    }
  }
  return dimVec;
}

SmallVector<NPUTensorDesc, N> logdet_npu_input(
  const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> logdet_npu_output(
  const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> logdet_npu_attr(const Tensor& self) {
  SmallVector<NPUAttrDesc, N> attrs = {};
  return attrs;
}

Tensor& logdet_out_npu(Tensor& result, const Tensor& self) {
  // constructs the input and output NPUTensorDesc
  auto inputs = logdet_npu_input({self});
  auto outputs = logdet_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = logdet_npu_attr(self);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("LogDet", inputs, outputs, attrs);

  return result;
}

Tensor logdet_npu(const Tensor& self) {
  // calculate the output size
  auto outputSize = logdet_npu_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
  outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  // calculate the output result of the NPU
  logdet_out_npu(result, self);
  return result;
}
} // namespace native
} // namespace at
