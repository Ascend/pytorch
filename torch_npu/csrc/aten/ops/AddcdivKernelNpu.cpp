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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& addcdiv_npu_nocheck(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar value,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Addcdiv")
    .Input(self)
    .Input(tensor1)
    .Input(tensor2)
    .Input(value, self.scalar_type())
    .Output(result)
    .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::addcdiv_out(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar value,
    at::Tensor& result) {
  auto divOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), divOutputSize);
  at::Tensor temp = OpPreparation::ApplyTensor(self, outputSize);
  addcdiv_npu_nocheck(self, tensor1, tensor2, value, temp);
  OpPreparation::CheckOut(
      {temp},
      result,
      temp);
  result.copy_(temp);
  return result;
}

at::Tensor NPUNativeFunctions::addcdiv(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar value) {
  auto divOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), divOutputSize);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  addcdiv_npu_nocheck(self, tensor1, tensor2, value, result);
  return result;
}

at::Tensor& NPUNativeFunctions::addcdiv_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    at::Scalar value) {
  OpPreparation::CheckMemory({self, tensor1, tensor2}, {self});
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = NPUNativeFunctions::addcdiv_out(contiguousSelf, tensor1, tensor2, value, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::addcdiv_out(self, tensor1, tensor2, value, self);
  }
  return self;
}

} // namespace native
} // namespace at_npu
