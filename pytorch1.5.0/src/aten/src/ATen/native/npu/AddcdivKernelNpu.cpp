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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& addcdiv_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  bool isFp32 = self.scalar_type() == at::kFloat && tensor1.scalar_type() == at::kFloat && tensor2.scalar_type() == at::kFloat;
  Tensor selfCp = isFp32 ? self : self.npu_dtype_cast(at::kFloat);
  Tensor tensor1Cp = isFp32 ? tensor1 : tensor1.npu_dtype_cast(at::kFloat);
  Tensor tensor2Cp = isFp32 ? tensor2 : tensor2.npu_dtype_cast(at::kFloat);
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

Tensor& addcdiv_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  auto divOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), divOutputSize);
  bool isFp32 = self.scalar_type() == at::kFloat && tensor1.scalar_type() == at::kFloat && tensor2.scalar_type() == at::kFloat;
  Tensor temp = isFp32 ? OpPreparation::ApplyTensor(self, outputSize)
                      : OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::kFloat), self);
  addcdiv_npu_nocheck(temp, self, tensor1, tensor2, value);
  temp = isFp32 ? temp : temp.npu_dtype_cast(self.scalar_type());
  OpPreparation::CheckOut(
      {temp},
      result,
      temp);
  result.copy_(temp);
  return result;
}

Tensor addcdiv_npu(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {

  auto divOutputSize = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto outputSize = broadcast_ops_npu_output_size(self.sizes(), divOutputSize);
  bool isFp32 = self.scalar_type() == at::kFloat && tensor1.scalar_type() == at::kFloat && tensor2.scalar_type() == at::kFloat;
  Tensor result = isFp32 ? OpPreparation::ApplyTensor(self, outputSize)
                      : OpPreparation::ApplyTensor(outputSize, self.options().dtype(at::kFloat), self);
  addcdiv_npu_nocheck(result, self, tensor1, tensor2, value);
  result = isFp32 ? result : result.npu_dtype_cast(self.scalar_type());
  return result;
}

Tensor& addcdiv_npu_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar value) {
  addcdiv_out_npu(self, self, tensor1, tensor2, value);
  return self;
}

} // namespace native
} // namespace at