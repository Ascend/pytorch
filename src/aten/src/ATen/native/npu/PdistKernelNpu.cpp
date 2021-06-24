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

SmallVector<NPUTensorDesc, N> pdist_npu_input(
    const SmallVector<Tensor, N>& inputTensor){
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> pdist_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N>  pdist_npu_attr(float p_value) {
  NPUAttrDesc P = NPUAttrDesc("p", p_value);
  SmallVector<NPUAttrDesc, N> attrs = {P};
  return attrs;
}

Tensor& pdist_out_npu(   
    Tensor& result, 
    const Tensor& self,
    float p) {
  // constructs the input and output NPUTensorDesc
  auto inputs = pdist_npu_input({self});
  auto outputs = pdist_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = pdist_npu_attr(p);
  
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("Pdist", inputs, outputs, attrs);
  
  return result;
}

Tensor pdist_npu(const Tensor& self, double p) {
  TORCH_CHECK(self.dim() == 2,
      "pdist only supports 2D tensors, got: ", self.dim(), "D");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "pdist only supports floating-point dtypes");
  TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
  return at::_pdist_forward(self, p);
}

Tensor _pdist_forward_npu(const Tensor& self, double p) {
  Tensor result;
  if (self.size(0) <= 1) {
    result = at::empty_with_format(
        {0},
        self.options(),
        CalcuOpUtil::get_tensor_npu_format(self));   
  } else {
    // double is not supported in NPU,  type of P needs to be converted from double to float.
    float p_float;
    if (std::isinf(p)) {
      p_float = std::numeric_limits<float>::infinity();
    } else {
      TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu dose not support float64" );
      p_float = (float) p;
    }
    auto outputSize =  pdist_npu_output_size(self, p_float);
    result = at::empty_with_format(
        outputSize,
        self.options(),
        CalcuOpUtil::get_tensor_npu_format(self));
    if(self.size(1) == 0){
      result.fill_(0);
    } else {
      pdist_out_npu(result, self, p_float);
    }  
  }
  return result;
}

} // namespace native
} // namespace at