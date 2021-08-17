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

Tensor& __lshift___out_npu(
    Tensor& result,
    const Tensor& self,
    Scalar other) {
  OpCommand cmd;
  cmd.Name("LeftShift")
     .Input(self)
     .Input(other,self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

Tensor& __lshift___out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
    OpCommand cmd;
    cmd.Name("LeftShift")
       .Input(self)
       .Input(other)
       .Output(result)
       .Run(); 

  return result;
}

Tensor __lshift___npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  __lshift___out_npu(result, self, other);

  return result;
}

Tensor __lshift___npu(const Tensor& self, Scalar other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  __lshift___out_npu(result, self, other);

  return result;
}

} // namespace native
} // namespace at
