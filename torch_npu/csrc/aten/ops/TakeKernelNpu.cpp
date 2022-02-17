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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<NPUTensorDesc, N> take_npu_input(
    const c10::SmallVector<at::Tensor, N>& inputTensor) {
  at::Tensor contiguousTensor;
  c10::SmallVector<NPUTensorDesc, N> inputs;
    
  for (int i = 0; i < inputTensor.size(); i++) {
    if (i == 0) {
      int64_t input_size = 1;
      at::Tensor input_tensor = inputTensor[i].reshape(-1);
      contiguousTensor = NpuUtils::format_contiguous(input_tensor);
    } else {
       contiguousTensor = NpuUtils::format_contiguous(inputTensor[i]);
    }
    inputs.emplace_back(NPUTensorDesc(contiguousTensor));
  }
  return inputs;
}

c10::SmallVector<NPUTensorDesc, N> take_npu_output(const c10::SmallVector<at::Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

c10::SmallVector<NPUAttrDesc, N> take_npu_attr(const at::Tensor& self) {
  NPUAttrDesc npuAttrValidateIndices = NPUAttrDesc("validate_indices", false);
  c10::SmallVector<NPUAttrDesc, N> attrs = {npuAttrValidateIndices};
  return attrs;
}

at::Tensor& NPUNativeFunctions::take_out(const at::Tensor& self, const at::Tensor& index, at::Tensor& result) {
  // constructs the input and output NPUTensorDesc
  auto inputs = take_npu_input({self,index});
  auto outputs = take_npu_output({result});
  // constructs the attr of the NPUAttrDesc
  auto attrs = take_npu_attr(self);
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Gather", inputs, outputs, attrs);
  return result;
}

at::Tensor NPUNativeFunctions::take(const at::Tensor& self, const at::Tensor& index) {
  // calculate the output size
  auto outputSize = input_same_output_size(index);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(
      outputSize,
      self.options());

  NPUNativeFunctions::take_out(self, index, result);
  return result;
}
} // namespace native
} // namespace at_npu