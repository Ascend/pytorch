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

SmallVector<NPUTensorDesc, N> matrix_power_npu_input(const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> matrix_power_npu_output(const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> matrix_power_npu_attr(int64_t n) {
  NPUAttrDesc npuAttrN = NPUAttrDesc("n", n);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrN};
  return attrs;
}

Tensor& matrix_power_out_npu_3d(Tensor& result, const Tensor& self, int64_t n) {
  // constructs the input and output NPUTensorDesc
  auto inputs = matrix_power_npu_input({self});
  auto outputs = matrix_power_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = matrix_power_npu_attr(n);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("MatrixPower", inputs, outputs, attrs);
  
  return result;
}

Tensor& matrix_power_out_npu(Tensor& result, const Tensor& self, int64_t n) {
  TORCH_CHECK(self.dim() >= 2 && (at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type())),
              "matrix_power(", self.scalar_type(), "{", self.sizes(), "}): expected a tensor "
              "of floating types with dim at least 2");

  if (n == 1) {
    result = self.clone(at::MemoryFormat::Contiguous);
  } else if (self.dim() == 2) {
    // 2D (M*M) tensor reshape to 3D (1*M*M) tensor
    auto shape = array_to_small_vector(self.sizes());
    shape.insert(shape.begin(), 1);

    Tensor input = self.reshape(shape);
    Tensor output = at::empty_with_format(
        shape, result.options(), CalcuOpUtil::get_tensor_npu_format(result));
    
    matrix_power_out_npu_3d(output, input, n);

    result = output.reshape(self.sizes());
  } else {
    matrix_power_out_npu_3d(result, self, n);
  }

  return result;
}

Tensor matrix_power_npu(const Tensor& self, int64_t n) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  matrix_power_out_npu(result, self, n);

  return result;
}

} // namespace native
} // namespace at