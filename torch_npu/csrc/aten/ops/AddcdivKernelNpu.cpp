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
  at::Tensor selfCp = self.scalar_type() == at::kFloat ? self : NPUNativeFunctions::npu_dtype_cast(self, at::kFloat);
  at::Tensor tensor1Cp = tensor1.scalar_type() == at::kFloat ? tensor1 : NPUNativeFunctions::npu_dtype_cast(tensor1, at::kFloat);
  at::Tensor tensor2Cp = tensor2.scalar_type() == at::kFloat ? tensor2 : NPUNativeFunctions::npu_dtype_cast(tensor2, at::kFloat);
  OpCommand cmd;
  cmd.Name("Addcdiv")
    .Input(selfCp)
    .Input(tensor1Cp)
    .Input(tensor2Cp)
    .Input(value, selfCp.scalar_type())
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
  bool isFp32 = self.scalar_type() == at::kFloat && tensor1.scalar_type() == at::kFloat && tensor2.scalar_type() == at::kFloat;
  at::Tensor temp = isFp32 ? OpPreparation::ApplyTensor(self, outputSize)
                      : OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::kFloat), self);
  addcdiv_npu_nocheck(self, tensor1, tensor2, value, temp);
  temp = isFp32 ? temp : NPUNativeFunctions::npu_dtype_cast(temp, self.scalar_type());
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
  bool isFp32 = self.scalar_type() == at::kFloat && tensor1.scalar_type() == at::kFloat && tensor2.scalar_type() == at::kFloat;
  at::Tensor result = isFp32 ? OpPreparation::ApplyTensor(self, outputSize)
                      : OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::kFloat), self);
  addcdiv_npu_nocheck(self, tensor1, tensor2, value, result);
  result = isFp32 ? result : NPUNativeFunctions::npu_dtype_cast(result, self.scalar_type());
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
