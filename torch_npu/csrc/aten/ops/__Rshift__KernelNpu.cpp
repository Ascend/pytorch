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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& rshift_out_npu_nocheck(
    const at::Tensor& self,
    at::Scalar other,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("RightShift")
     .Input(self)
     .Input(other, self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

at::Tensor& rshift_out_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
    OpCommand cmd;
    cmd.Name("RightShift")
       .Input(self)
       .Input(other)
       .Output(result)
       .Run(); 

  return result;
}

at::Tensor NPUNativeFunctions::__rshift__(const at::Tensor& self, const at::Tensor& other) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);
  rshift_out_npu_nocheck(self, other,result);

  return result;
}

at::Tensor NPUNativeFunctions::__rshift__(const at::Tensor& self, const at::Scalar& other) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);

  rshift_out_npu_nocheck(self, other, result);

  return result;
}

} // namespace native
} // namespace at