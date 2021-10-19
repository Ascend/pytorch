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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

// pow.Tensor_Tensor_out
Tensor& pow_tensor_tensor_out_npu_nocheck(const Tensor& self, const Tensor& exp, Tensor& result) {
  OpCommand cmd;
  cmd.Name("Pow")
    .Input(self)
    .Input(exp)
    .Output(result)
    .Run();

  return result;
}

// pow.Tensor_Scalar_out
Tensor& pow_tensor_scalar_out_npu_nocheck(const Tensor& self, Scalar exp, Tensor& result) {
  OpCommand cmd;
  cmd.Name("Pow")
    .Input(self)
    .Input(exp,self.scalar_type())
    .Output(result)
    .Run();

  return result;
}

// pow.Scalar_out
Tensor& pow_scalar_out_npu_nocheck(Scalar self, const Tensor& exp, Tensor& result) {
  OpCommand cmd;
  cmd.Name("Pow")
    .Input(self,exp.scalar_type())
    .Input(exp)
    .Output(result)
    .Run();

  return result;
}

// pow.Tensor_Tensor_out
Tensor& pow_tensor_tensor_out_npu(const Tensor& self, const Tensor& exp, Tensor& result) {
  auto outputSize = broadcast_ops_npu_output_size(self, exp);
  OpPreparation::CheckOut(
      {self, exp},
      result,
      self,
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self, exp}, {result})
   .Func([&self, &exp](Tensor& result){pow_tensor_tensor_out_npu_nocheck(self, exp, result);})
   .Call(result);
}

// pow.Tensor_Scalar_out
Tensor& pow_tensor_scalar_out_npu(const Tensor& self, Scalar exp, Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &exp](Tensor& result){pow_tensor_scalar_out_npu_nocheck(self, exp, result);})
   .Call(result);
}

// pow.Scalar_out
Tensor& pow_scalar_out_npu(Scalar self, const Tensor& exp, Tensor& result) {
  OpPreparation::CheckOut(
      {exp},
      result,
      exp);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({exp}, {result})
   .Func([&self, &exp](Tensor& result){pow_scalar_out_npu_nocheck(self, exp, result);})
   .Call(result);
}

Tensor pow_tensor_tensor_npu(const Tensor& self, const Tensor& exp) {
  // calculate the output size
  auto outputSize = broadcast_ops_npu_output_size(self, exp);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  pow_tensor_tensor_out_npu_nocheck(self, exp, result);
  return result;
}

Tensor pow_tensor_scalar_npu(const Tensor& self, Scalar exp) {
  Tensor result = OpPreparation::ApplyTensor(self);
  pow_tensor_scalar_out_npu_nocheck(self, exp, result);
  return result;
}

Tensor pow_scalar_npu(Scalar self, const Tensor& exp) {
  // calculate the output size
  auto outputSize = input_same_output_size(exp);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, exp.options());

  // calculate the output result of the NPU
  pow_scalar_out_npu_nocheck(self, exp, result);
  return result;
}

Tensor& pow_tensor_npu_(Tensor& self, const Tensor& exp) {
  pow_tensor_tensor_out_npu(self, exp, self);
  return self;
}

Tensor& pow_scalar_npu_(Tensor& self, Scalar exp) {
  pow_tensor_scalar_out_npu(self, exp, self);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("pow_.Scalar", TORCH_FN(pow_scalar_npu_));
  m.impl("pow_.Tensor", TORCH_FN(pow_tensor_npu_));
  m.impl("pow.Tensor_Tensor_out", TORCH_FN(pow_tensor_tensor_out_npu));
  m.impl("pow.Tensor_Tensor", TORCH_FN(pow_tensor_tensor_npu));
  m.impl("pow.Scalar_out", TORCH_FN(pow_scalar_out_npu));
  m.impl("pow.Scalar", TORCH_FN(pow_scalar_npu));
  m.impl("pow.Tensor_Scalar_out", TORCH_FN(pow_tensor_scalar_out_npu));
  m.impl("pow.Tensor_Scalar", TORCH_FN(pow_tensor_scalar_npu));
}

} // namespace native
} // namespace at
