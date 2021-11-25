// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;


Tensor& dropout_v2_backward_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask,
    double p) {
  
  OpCommand cmd;
  cmd.Name("MaskedScale")
      .Input(self)
      .Input(mask)
      .Output(result)
      .Attr("value",static_cast<float>(1./(1-p)))
      .Run();
  return result;
}

Tensor dropout_v2_backward_npu(const Tensor& grad_output, const Tensor& mask, double p){
  TORCH_CHECK(grad_output.scalar_type() == ScalarType::Half ||
              grad_output.scalar_type() == ScalarType::Float,
              "grad_output's dtype only support fp16 or fp32 current");
  TORCH_CHECK(mask.scalar_type() == ScalarType::Half ||
              mask.scalar_type() == ScalarType::Float ||
              mask.scalar_type() == ScalarType::Char || 
              mask.scalar_type() == ScalarType::Byte,
              "mask's dtype should be float32, float16, or int8 and uint8" );
  TORCH_CHECK(grad_output.sizes() == mask.sizes(),
              "grad_output must be the same shape with mask");

  Tensor maskCopy = mask;
  if (maskCopy.scalar_type() == ScalarType::Byte){
    maskCopy = maskCopy.to(ScalarType::Half);
  }
  auto result = OpPreparation::ApplyTensor(grad_output);
  dropout_v2_backward_out_npu(result, grad_output, maskCopy, p);

  return result;

}

} // namespace native
} // namespace at

