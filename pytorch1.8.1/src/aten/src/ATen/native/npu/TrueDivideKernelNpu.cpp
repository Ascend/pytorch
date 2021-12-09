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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor true_divide_dest_output(const Tensor& self, const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  if (not isSelfWrapped) {
    return self;
  } else {
    return other;
  }
}

Tensor& true_divide_Tensor_out_npu_nocheck(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  OpCommand cmd;
  cmd.Name("Div")
     .Input(self)
     .Input(other)
     .Output(result)
     .Run();

  return result;
}

Tensor& true_divide_Tensor_out_npu(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  Tensor selfTemp=self;
  Tensor otherTemp = other;
  if (self.scalar_type() == ScalarType::Int){
    selfTemp = self.npu_dtype_cast(ScalarType::Float);
    result = result.npu_dtype_cast(ScalarType::Float);
  }

  if(other.scalar_type() == ScalarType::Int||other.scalar_type() == ScalarType::Bool){
    otherTemp = other.npu_dtype_cast(ScalarType::Float);
  }

  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(selfTemp);
  bool isOtherWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(otherTemp);
  if (isSelfWrapped && (!isOtherWrapped)) {
    selfTemp = selfTemp.npu_dtype_cast(otherTemp.scalar_type());
  } else if (isOtherWrapped && (!isSelfWrapped)) {
    otherTemp = otherTemp.npu_dtype_cast(selfTemp.scalar_type());
  }

  Tensor outputTensor = true_divide_dest_output(selfTemp, otherTemp);
  auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);

  OpPreparation::CheckOut(
      {outputTensor},
      result,
      CalcuOpUtil::get_tensor_npu_format(outputTensor),
      outputTensor.scalar_type(),
      outputSize);

  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({selfTemp, otherTemp}, {result})
   .Func([&selfTemp, &otherTemp](Tensor& result){true_divide_Tensor_out_npu_nocheck(selfTemp, otherTemp, result);})
   .Call(result);

  if (self.scalar_type() == ScalarType::Int){
    result = result.npu_dtype_cast(ScalarType::Int);
  }
  return result;
}

Tensor& true_divide_Scalar_out_npu_nocheck(Tensor& result, const Tensor& self, const Scalar other) {
  Tensor selfTemp = self;
  if (self.scalar_type() == ScalarType::Int){
    selfTemp = self.npu_dtype_cast(ScalarType::Float);
    result = result.npu_dtype_cast(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("Div")
     .Input(selfTemp)
     .Input(other, selfTemp.scalar_type())
     .Output(result)
     .Run();

  if (self.scalar_type() == ScalarType::Int){
    result = result.npu_dtype_cast(ScalarType::Int);
  }
  return result;
}

Tensor true_divide_Tensor_npu(const Tensor& self, const Tensor& other) {
  Tensor  selfTemp=self;
  Tensor  otherTemp = other;
  if (self.scalar_type() == ScalarType::Int || self.scalar_type() == ScalarType::Bool){
    selfTemp = self.npu_dtype_cast(ScalarType::Float);
  }
  if (other.scalar_type() == ScalarType::Int || other.scalar_type() == ScalarType::Bool){
    otherTemp = other.npu_dtype_cast(ScalarType::Float);
  }

  Tensor outputTensor = true_divide_dest_output(selfTemp, otherTemp);
  auto outputSize = broadcast_ops_npu_output_size(selfTemp, otherTemp);

  Tensor result = OpPreparation::ApplyTensor(outputTensor, outputSize);

  true_divide_Tensor_out_npu_nocheck(selfTemp, otherTemp, result);

  return result;
}

Tensor true_divide_Scalar_npu(const Tensor& self, Scalar other) {
  Tensor selfTemp =self;
  if (self.scalar_type() == ScalarType::Int || self.scalar_type() == ScalarType::Bool){
    selfTemp = self.npu_dtype_cast(ScalarType::Float);
  }

  Tensor result = OpPreparation::ApplyTensor(self);

  true_divide_Scalar_out_npu_nocheck(result, selfTemp, other);

  return result;
}

Tensor& true_divide_Tensor_npu_(Tensor& self, const Tensor& other) {
  true_divide_Tensor_out_npu(self, other, self);

  return self;
}

Tensor& true_divide_Scalar_npu_(Tensor& self, Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = true_divide_Scalar_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    true_divide_Scalar_out_npu_nocheck(self, self, other);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("true_divide.Tensor", TORCH_FN(true_divide_Tensor_npu));
  m.impl("true_divide_.Tensor", TORCH_FN(true_divide_Tensor_npu_));
  m.impl("true_divide.out", TORCH_FN(true_divide_Tensor_out_npu));
  m.impl("true_divide.Scalar", TORCH_FN(true_divide_Scalar_npu));
  m.impl("true_divide_.Scalar", TORCH_FN(true_divide_Scalar_npu_));
}
} // namespace native
} // namespace at
