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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& __rshift___out_npu_nocheck(
    const Tensor& self,
    Scalar other,
    Tensor& result) {
  OpCommand cmd;
  cmd.Name("RightShift")
     .Input(self)
     .Input(other,self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

Tensor& __rshift___out_npu_nocheck(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
    OpCommand cmd;
    cmd.Name("RightShift")
       .Input(self)
       .Input(other)
       .Output(result)
       .Run(); 

  return result;
}

Tensor __rshift___tensor_npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);
  __rshift___out_npu_nocheck( self, other,result);

  return result;
}

Tensor __rshift___scalar_npu(const Tensor& self, Scalar other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);

  __rshift___out_npu_nocheck( self, other,result);

  return result;
}
TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("__rshift__.Tensor", TORCH_FN(__rshift___tensor_npu));
  m.impl("__rshift__.Scalar", TORCH_FN(__rshift___scalar_npu));
}
} // namespace native
} // namespace at