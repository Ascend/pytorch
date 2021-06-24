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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at { 
namespace native {
using namespace at::native::npu;

namespace {

Tensor& isclose_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    double rtol,
    double atol,
    bool equal_nan) {
  auto rtol1 = static_cast<float>(rtol);
  auto atol1 = static_cast<float>(atol);

  OpCommand cmd;
  cmd.Name("IsClose")
      .Input(self)
      .Input(other)
      .Attr("rtol", rtol1)
      .Attr("atol", atol1)
      .Attr("equal_nan", equal_nan)
      .Output(result)
      .Run();

  return result;
}
} // namespace

Tensor isclose_npu(
    const Tensor& self,
    const Tensor& other,
    double rtol, 
    double atol, 
    bool equal_nan) {

  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type());
    
  //calculate the output size
  auto outputSize = input_same_output_size(self);
        
  //construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(outputSize, self.options().dtype(kBool), self);
  // constructs the attr of the NPUAttrDesc
  result = isclose_nocheck(result, self, other, rtol, atol, equal_nan);
    
  return result;  
}

}}  // namespace at::native


