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

SmallVector<NPUTensorDesc, N> take_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  Tensor contiguousTensor;
  SmallVector<NPUTensorDesc, N> inputs;
    
  for (int i = 0; i < inputTensor.size(); i++) {
    if (i == 0) {
      int64_t input_size = 1;
      Tensor input_tensor = inputTensor[i].reshape(-1);
      contiguousTensor = NpuUtils::format_contiguous(input_tensor);
    } else {
       contiguousTensor = NpuUtils::format_contiguous(inputTensor[i]);
    }
    inputs.emplace_back(NPUTensorDesc(contiguousTensor));
  }
  return inputs;
}

SmallVector<NPUTensorDesc, N> take_npu_output(const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> take_npu_attr(const Tensor& self) {
  NPUAttrDesc npuAttrValidateIndices = NPUAttrDesc("validate_indices", false);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrValidateIndices};
  return attrs;
}

Tensor& take_out_npu(Tensor& result, const Tensor& self, const Tensor& index) {
  // constructs the input and output NPUTensorDesc
  auto inputs = take_npu_input({self,index});
  auto outputs = take_npu_output({result});
  // constructs the attr of the NPUAttrDesc
  auto attrs = take_npu_attr(self);
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Gather", inputs, outputs, attrs);
  return result;
}

Tensor take_npu(const Tensor& self, const Tensor& index) {
  // calculate the output size
  auto outputSize = input_same_output_size(index);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));
  take_out_npu(result, self, index);
  return result;
}
} // namespace native
} // namespace at
