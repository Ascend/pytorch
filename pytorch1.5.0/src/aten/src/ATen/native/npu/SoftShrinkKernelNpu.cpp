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

SmallVector<NPUTensorDesc, N> softshrink_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> softshrink_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> softshrink_npu_attr(Scalar lambd) {
  float lambd_value = CalcuOpUtil::get_scalar_float_value(lambd);
  NPUAttrDesc npuAttrScalarLambd = NPUAttrDesc("lambd", lambd_value);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrScalarLambd};
  return attrs;
}

Tensor& softshrink_out_npu(   
    Tensor& result, 
    const Tensor& self,
    Scalar lambd) {
  TORCH_CHECK(lambd.toFloat() > 0, "lambd should be greater than 0");
  // constructs the input and output NPUTensorDesc
  auto inputs = softshrink_npu_input({self});
  auto outputs = softshrink_npu_output({result});
  // constructs the attr of the NPUAttrDesc
  auto attrs = softshrink_npu_attr(lambd);
  
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("SoftShrink", inputs, outputs, attrs);
  
  return result;
}

Tensor softshrink_npu(const Tensor& self, Scalar lambd) {
  TORCH_CHECK(lambd.toFloat() > 0, "lambd should be greater than 0");
  // calculate the output size  
  auto outputSize = input_same_output_size(self);

  Tensor result = at::empty_with_format(
      outputSize,
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));

  softshrink_out_npu(result, self, lambd);

  return result;
}

} // namespace native
} // namespace at